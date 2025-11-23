

export function parseResumeText(text) {
  const resume = {
    name: "",
    email: "",
    phone: "",
    location: "",
    portfolio: "",
    education: [],
    experience: [],
    skills: [],
    summary: "",
  };

  // Clean text
  const clean = text.replace(/\r/g, "").trim();

  // ----------------------------------------
  // 1. Extract basic info (name, email, phone)
  // ----------------------------------------
  resume.email =
    clean.match(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}/)?.[0] || "";

  resume.phone = clean.match(/(\+?\d{10,15})/)?.[0] || "";

  // Name assumption: First line or before "PROFILE" section
  const lines = clean
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 1);
  resume.name = lines[0];

  // ----------------------------------------
  // 2. Extract Skills
  // ----------------------------------------
  const skillsIdx = clean.indexOf("Skills");
  if (skillsIdx !== -1) {
    const skillBlock = clean
      .slice(skillsIdx)
      .split(/Work Highlight|Experience|Professional/i)[0];

    resume.skills = skillBlock
      .replace("Skills", "")
      .split("\n")
      .map((s) => s.trim())
      .filter((s) => s.length > 1);
  }

  // ----------------------------------------
  // 3. Extract Education Section
  // ----------------------------------------
  const eduMatch = clean.match(
    /Education([\s\S]*?)(Experience|Professional|Skills)/i
  );
  if (eduMatch) {
    const eduText = eduMatch[1].trim();
    const eduLines = eduText
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);

    // Simple split logic
    for (let i = 0; i < eduLines.length; i++) {
      if (
        eduLines[i].includes("University") ||
        eduLines[i].includes("College") ||
        eduLines[i].includes("Institute")
      ) {
        resume.education.push({
          degree: eduLines[i - 1] || "",
          institution: eduLines[i],
          duration: eduLines[i + 1] || "",
        });
      }
    }
  }

  // ----------------------------------------
  // 4. Extract Experience Section
  // ----------------------------------------
  const expMatch = clean.match(
    /Experience([\s\S]*?)(Skills|Projects|Work Highlight)/i
  );
  if (expMatch) {
    const expText = expMatch[1].trim();
    const expLines = expText
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);

    let current = { company: "", role: "", duration: "", responsibilities: [] };

    expLines.forEach((line) => {
      // Company detection
      if (line.match(/[A-Za-z].+ (Ltd|Solution|IT|Agency)/i)) {
        if (current.company) resume.experience.push(current);
        current = {
          company: line,
          role: "",
          duration: "",
          responsibilities: [],
        };
      }
      // Role
      else if (line.toLowerCase().includes("developer")) {
        current.role = line;
      }
      // Duration
      else if (line.match(/\d{4}/)) {
        current.duration = line;
      }
      // Responsibility item
      else {
        current.responsibilities.push(line);
      }
    });

    if (current.company) resume.experience.push(current);
  }

  // ----------------------------------------
  // 5. Extract Summary
  // ----------------------------------------
  const summaryMatch = clean.match(/Summary([\s\S]*?)(Skills|Experience)/i);
  if (summaryMatch) {
    resume.summary = summaryMatch[1].trim();
  }

  return resume;
}
