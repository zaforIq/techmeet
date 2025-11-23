# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import json
import fitz  # pymupdf
from difflib import SequenceMatcher
from sklearn.preprocessing import OrdinalEncoder
import traceback

app = FastAPI(title="Matching ML Service - Full Pipeline")

# ---------------------------
# Config / thresholds / weights
# ---------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
ROLE_MATCH_THRESHOLD = 0.60   # keep if role similarity >= this (stage1)
SKILL_MATCH_THRESHOLD = 0.75  # keep if resume->job skill containment >= this (stage1)
EXP_MATCH_THRESHOLD = 0.60    # keep if experience score >= this (stage2)
TEXT_MATCH_THRESHOLD = 0.40   # keep if embedding similarity >= this (stage3)

WEIGHTS = {
    "role": 0.30,
    "skill": 0.40,
    "experience": 0.15,
    "text": 0.15
}

EXPERIENCE_LEVELS = [
    "Fresher", "Entry-Level", "Junior", "Mid-Level",
    "Mid-Senior", "Senior", "Lead"
]

# Ordinal encoder for experience levels
enc = OrdinalEncoder()

# ---------------------------
# Load model once (singleton)
# ---------------------------
model = SentenceTransformer(MODEL_NAME)
EMB_DIM = model.get_sentence_embedding_dimension()

# ---------------------------
# Pydantic models for requests
# ---------------------------
class JobItem(BaseModel):
    id: str
    title: Optional[str] = ""
    skills: Optional[List[str]] = []
    responsibilities: Optional[str] = ""
    experiencelevel: Optional[str] = ""
    years_of_experience: Optional[str] = None
    # any other fields allowed

class ResumeItem(BaseModel):
    id: str
    role: Optional[str] = ""
    skills: Optional[List[str]] = []
    summary: Optional[str] = ""
    projects: Optional[str] = ""
    experiencelevel: Optional[str] = ""
    years_of_experience: Optional[str] = None
    resume_url: Optional[str] = None  # local path or s3 url (optional)

class GenerateRequest(BaseModel):
    jobs: List[JobItem]
    resumes: List[ResumeItem]

# ---------------------------
# Utility helpers
# ---------------------------
def normalize_skill_token(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def normalize_skills_list(raw) -> List[str]:
    if not raw:
        return []
    # If skills already a list, normalize tokens
    if isinstance(raw, list):
        toks = [normalize_skill_token(x) for x in raw if x]
        return sorted(list(set(toks)))
    # if string, split common separators
    s = str(raw)
    s = s.replace(",", ";").replace("|", ";")
    toks = [normalize_skill_token(x) for x in s.split(";") if x.strip()]
    return sorted(list(set(toks)))

def parse_years_of_experience(raw):
    """Return numeric float approximating years (average of range, handle 5+ -> 5, '3' -> 3, '0-1' -> 0.5)"""
    if raw is None:
        return 0.0
    s = str(raw).lower().strip()
    s = s.replace("years", "").replace("year", "").strip()
    s = s.replace("–", "-").replace("—", "-")
    # handle '5+' or '5 +'
    plus = re.search(r"(\d+)\s*\+", s)
    if plus:
        return float(int(plus.group(1)))
    # single number
    m = re.match(r"^\d+(\.\d+)?$", s)
    if m:
        return float(m.group(0))
    # range like 3-5
    if "-" in s:
        parts = [p for p in s.split("-") if re.search(r"\d+", p)]
        nums = []
        for p in parts:
            nm = re.search(r"\d+(\.\d+)?", p)
            if nm:
                nums.append(float(nm.group(0)))
        if nums:
            return sum(nums) / len(nums)
    # fallback: extract first number
    nm = re.search(r"\d+(\.\d+)?", s)
    if nm:
        return float(nm.group(0))
    return 0.0

def role_similarity(job_title: str, resume_role: str, role_map: Dict[str, List[str]] = None) -> float:
    jt = (job_title or "").lower().strip()
    rr = (resume_role or "").lower().strip()
    if not jt or not rr:
        return 0.0
    # if mapping provided, try match against mapped roles
    if role_map:
        for k, mapped in role_map.items():
            if k.lower().strip() == jt:
                best = 0.0
                for m in mapped:
                    best = max(best, SequenceMatcher(None, rr, m.lower().strip()).ratio())
                return best
    # fallback fuzzy ratio
    return SequenceMatcher(None, jt, rr).ratio()

# Example base role_map — extend as needed
BASE_ROLE_MAP = {
    "full stack developer": ["full stack developer", "fullstack developer"],
    "react developer": ["react developer", "frontend developer"],
    "backend developer": ["backend developer", "server developer"],
    ".net developer": [".net developer", "dotnet developer", "backend developer"],
    "machine learning engineer": ["machine learning engineer", "ml engineer"],
    "devops engineer": ["devops engineer", "site reliability engineer"],
    "android developer": ["android developer", "mobile developer", "react native developer"],
    "cloud engineer": ["cloud engineer", "aws engineer", "azure engineer"],
}

def reversed_skill_containment(job_skills: List[str], resume_skills: List[str]) -> float:
    """
    Count how many resume skills are present in job skills divided by total resume skills.
    (resume -> job containment)
    """
    if not resume_skills:
        return 0.0
    job_set = set([normalize_skill_token(s) for s in (job_skills or [])])
    resume_set = set([normalize_skill_token(s) for s in (resume_skills or [])])
    matches = sum(1 for s in resume_set if s in job_set)
    return matches / len(resume_set)

def encode_texts(texts: List[str]) -> np.ndarray:
    """Return numpy array of embeddings (float32)"""
    if not texts:
        return np.zeros((0, EMB_DIM), dtype="float32")
    try:
        vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype="float32")
    except Exception:
        print("Error while computing embeddings for texts:")
        traceback.print_exc()
        raise
    # normalize rows for cosine similarity using inner product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    return vecs

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between a (m x d) and b (n x d) -> m x n
    Assumes rows are L2-normalized.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype="float32")
    return np.matmul(a, b.T)

# ---------------------------
# Experience ordinal conversion
# ---------------------------
def encode_experience_levels(jobs: List[Dict], resumes: List[Dict]):
    # Build arrays of experiencelevel strings for encoder fit/transform
    # If any experiencelevel is missing or not in canonical list, we coerce to nearest fallback
    all_levels = []
    for j in jobs:
        lvl = j.get("experiencelevel") or "Mid-Level"
        all_levels.append(lvl)
    for r in resumes:
        lvl = r.get("experiencelevel") or "Mid-Level"
        all_levels.append(lvl)
    # Fit ordinal encoder on canonical list to ensure correct mapping
    try:
        # ensure encoder is fitted to canonical EXPERIENCE_LEVELS
        enc.fit(np.array(EXPERIENCE_LEVELS).reshape(-1, 1))
    except Exception:
        print("Error fitting OrdinalEncoder for experience levels:")
        traceback.print_exc()
        raise
    # Now transform job/resume arrays safely
    def safe_transform_list(items):
        out = []
        for it in items:
            lvl = it.get("experiencelevel") or "Mid-Level"
            # fallback mapping if value unknown
            if lvl not in EXPERIENCE_LEVELS:
                # try approximate mapping by substring
                lvl_lower = lvl.lower()
                if "fresher" in lvl_lower or "0" in lvl_lower:
                    lvl = "Fresher"
                elif "entry" in lvl_lower:
                    lvl = "Entry-Level"
                elif "junior" in lvl_lower:
                    lvl = "Junior"
                elif "mid-senior" in lvl_lower or "mid senior" in lvl_lower:
                    lvl = "Mid-Senior"
                elif "senior" in lvl_lower and "mid" not in lvl_lower:
                    lvl = "Senior"
                elif "lead" in lvl_lower:
                    lvl = "Lead"
                elif "mid" in lvl_lower:
                    lvl = "Mid-Level"
                else:
                    lvl = "Mid-Level"
            out.append([lvl])
        arr = np.array(out)
        transformed = enc.transform(arr).astype(int).reshape(-1)
        return transformed
    job_exp_nums = safe_transform_list(jobs)
    resume_exp_nums = safe_transform_list(resumes)
    return job_exp_nums, resume_exp_nums

# ---------------------------
# Main pipeline endpoint
# ---------------------------
@app.post("/generate-report")
def generate_report(payload: GenerateRequest):
    try:
        jobs_in = [dict(j) for j in payload.jobs]
        resumes_in = [dict(r) for r in payload.resumes]

        print(f"Received {len(jobs_in)} jobs and {len(resumes_in)} resumes for processing.")

        # 0) Normalize skills lists & basic fields
        for j in jobs_in:
            j["clean_skills"] = normalize_skills_list(j.get("skills", []))
            j["clean_title"] = (j.get("title") or "").strip()
            j["responsibilities_text"] = (j.get("responsibilities") or "").strip()
            j["years_numeric"] = parse_years_of_experience(j.get("years_of_experience") or j.get("years_of_experience_raw"))
        for r in resumes_in:
            # if skills provided as CSV string or list, normalize
            r["clean_skills"] = normalize_skills_list(r.get("skills", []))
            r["combined_text"] = ((r.get("summary") or "") + " " + (r.get("projects") or "")).strip()
            # If resume_url provided and combined_text empty, extract pdf text
            if (not r["combined_text"]) and r.get("resume_url"):
                try:
                    doc = fitz.open(r["resume_url"])
                    pages = [p.get_text("text") for p in doc]
                    r["combined_text"] = "\n".join(pages)
                except Exception:
                    r["combined_text"] = ""
            r["years_numeric"] = parse_years_of_experience(r.get("years_of_experience") or r.get("years_of_experience_raw"))

        # 1) Step 1: Role + Skill matching (filter)
        step1 = []  # list of dicts: job_idx, resume_idx, role_score, skill_score
        for ji, j in enumerate(jobs_in):
            for ri, r in enumerate(resumes_in):
                role_score = role_similarity(j.get("clean_title", ""), r.get("role", ""), BASE_ROLE_MAP)
                skill_score = reversed_skill_containment(j.get("clean_skills", []), r.get("clean_skills", []))
                if role_score >= ROLE_MATCH_THRESHOLD and skill_score >= SKILL_MATCH_THRESHOLD:
                    step1.append({
                        "job_index": ji,
                        "resume_index": ri,
                        "role_match_score": round(float(role_score), 3),
                        "skill_match_score": round(float(skill_score), 3)
                    })

        # Build quick lookup for stage1 survivors
        survivors_stage1 = set((row["job_index"], row["resume_index"]) for row in step1)

        # 2) Step 2: Experience matching (apply only to survivors)
        # Encode experience levels ordinal
        job_level_nums, resume_level_nums = encode_experience_levels(jobs_in, resumes_in)
        step2 = []
        survivors_stage2 = set()
        MAX_GAP = len(EXPERIENCE_LEVELS) - 1  # 6
        for row in step1:
            ji = row["job_index"]; ri = row["resume_index"]
            job_num = int(job_level_nums[ji])
            res_num = int(resume_level_nums[ri])
            diff = abs(job_num - res_num)
            exp_score = 1.0 - (diff / MAX_GAP)
            exp_score = round(float(exp_score), 3)
            if exp_score >= EXP_MATCH_THRESHOLD:
                newrow = dict(row)
                newrow["experience_match_score"] = exp_score
                step2.append(newrow)
                survivors_stage2.add((ji, ri))

        # 3) Prepare embeddings for all relevant jobs and resumes (we'll compute once)
        # We'll only compute job embeddings for jobs that have at least one survivor in stage2
        job_indices_needed = sorted(set([ji for ji, ri in survivors_stage2]))
        resume_indices_needed = sorted(set([ri for ji, ri in survivors_stage2]))

        # If no survivors, return empty reports
        if not job_indices_needed or not resume_indices_needed:
            # Prepare analytics and return
            analytics = {
                "jobs_count": len(jobs_in),
                "resumes_count": len(resumes_in),
                "stage1_pairs": len(step1),
                "stage2_pairs": len(step2),
                "stage3_pairs": 0,
                "final_pairs": 0
            }
            return {
                "step1_role_skill": step1,
                "step2_experience": step2,
                "step3_text_similarity": [],
                "final_ranking": [],
                "analytics": analytics
            }

        job_texts = [jobs_in[i]["responsibilities_text"] or jobs_in[i].get("clean_title","") for i in job_indices_needed]
        resume_texts = [resumes_in[i]["combined_text"] or resumes_in[i].get("role","") for i in resume_indices_needed]

        job_embs = encode_texts(job_texts)
        resume_embs = encode_texts(resume_texts)

        # Map from global index to local embedding row index
        job_idx_to_emb_row = {job_idx: row for row, job_idx in enumerate(job_indices_needed)}
        res_idx_to_emb_row = {res_idx: row for row, res_idx in enumerate(resume_indices_needed)}

        # 4) Step 3: Text similarity (compute pairwise only for survivors_stage2)
        sim_matrix = cosine_similarity_matrix(job_embs, resume_embs)  # shape (len(job_indices), len(resume_indices))
        step3 = []
        survivors_stage3 = set()
        for ji, ri in survivors_stage2:
            jr = job_idx_to_emb_row[ji]
            rr = res_idx_to_emb_row[ri]
            score = float(sim_matrix[jr, rr])
            if score >= TEXT_MATCH_THRESHOLD:
                step3.append({
                    "job_index": ji,
                    "resume_index": ri,
                    "text_similarity_score": round(score, 3)
                })
                survivors_stage3.add((ji, ri))

        # 5) Final ranking: merge scores and compute weighted final_score for survivors_stage3
        final_rows = []
        # build quick maps for scores
        role_skill_map = {(r["job_index"], r["resume_index"]): r for r in step1}
        exp_map = {(r["job_index"], r["resume_index"]): r for r in step2}
        text_map = {(r["job_index"], r["resume_index"]): r for r in step3}

        for pair in sorted(survivors_stage3):
            ji, ri = pair
            rs = role_skill_map.get(pair)
            es = exp_map.get(pair)
            ts = text_map.get(pair)
            # safety check
            if not rs or not es or not ts:
                continue
            role_score = float(rs["role_match_score"])
            skill_score = float(rs["skill_match_score"])
            exp_score = float(es["experience_match_score"])
            text_score = float(ts["text_similarity_score"])
            final_score = (
                role_score * WEIGHTS["role"]
                + skill_score * WEIGHTS["skill"]
                + exp_score * WEIGHTS["experience"]
                + text_score * WEIGHTS["text"]
            )
            final_rows.append({
                "job_index": ji,
                "resume_index": ri,
                "role_match_score": round(role_score, 3),
                "skill_match_score": round(skill_score, 3),
                "experience_match_score": round(exp_score, 3),
                "text_similarity_score": round(text_score, 3),
                "final_score": round(float(final_score), 3)
            })

        # Sort and group final_rows per job (return top N per job)
        final_rows_sorted = sorted(final_rows, key=lambda x: (x["job_index"], -x["final_score"]))
        final_ranked_by_job = {}
        TOP_K = 10
        for row in final_rows_sorted:
            ji = row["job_index"]
            final_ranked_by_job.setdefault(str(ji), []).append(row)
        # trim to top K
        for k in final_ranked_by_job:
            final_ranked_by_job[k] = final_ranked_by_job[k][:TOP_K]

        # Analytics
        analytics = {
            "jobs_count": len(jobs_in),
            "resumes_count": len(resumes_in),
            "stage1_pairs": len(step1),
            "stage2_pairs": len(step2),
            "stage3_pairs": len(step3),
            "final_pairs": sum(len(v) for v in final_ranked_by_job.values())
        }

        return {
            "step1_role_skill": step1,
            "step2_experience": step2,
            "step3_text_similarity": step3,
            "final_ranking": final_ranked_by_job,
            "analytics": analytics
        }

    except Exception as e:
        # Log full traceback to server logs for debugging
        print("Unhandled exception in generate_report:")
        traceback.print_exc()
        # Return a generic error message to client
        raise HTTPException(status_code=500, detail="Internal server error - see server logs for details")
