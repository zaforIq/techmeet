import path from "path";
import fs from "fs";
import pdfParse from "pdf-parse";
import { parseResumeText } from "../services/resumeParser.js";

export const getresumeData = async (req, res) => {};

export const postresumeData = async (req, res) => {
  try {
    const filePath = path.join(process.cwd(), "uploads", req.file.filename);
    const buffer = fs.readFileSync(filePath);

    const parsed = await pdfParse(buffer);
    const resumeText = parsed.text;

    const json = parseResumeText(resumeText);


    const resumeData = {
      name: req.body.name || json.name,
      designation: req.body.designation || json.designation,
      skills: json.skills,
      summary: json.summary,
      projects: json.projects,
      experience: json.experience,
      resume_url: `/uploads/${req.file.filename}`,
    };

    console.log("Extracted Resume Data:", resumeData);

    return res.json({
      success: true,
      data: json,
    });
  } catch (error) {
    console.log(error);
    return res.status(500).json({ success: false, error: error.message });
  }
};
