import { generateReport } from "../services/mlClient.js";

export const runMatching = async (req, res) => {
  try {
    const jobs= [
    {
      "id": "job_001",
      "title": "Full Stack Developer",
      "skills": ["React", "Node.js", "Express", "MongoDB"],
      "responsibilities": "Build scalable MERN applications, REST APIs, dashboards.",
      "experiencelevel": "Mid-Senior",
      "years_of_experience": "4-7"
    },
    {
      "id": "job_002",
      "title": "React Developer",
      "skills": ["React", "Redux", "TypeScript", "Tailwind"],
      "responsibilities": "Develop frontend components and integrate with APIs.",
      "experiencelevel": "Junior",
      "years_of_experience": "1-2"
    }
  ]
  const resumes = [
    {
      id: "resume_001",
      role: "Full Stack Developer",
      skills: ["React", "Next.js", "Express", "Node.js", "MongoDB"],
      summary: "Build scalable MERN applications, REST APIs, dashboards.",
      projects:
        " chile na ekta project banaiechi jeita MERN stack use kore ecommerce website.",
      experiencelevel: "Mid-Senior",
      years_of_experience: "5+",
      resume_url: "/mnt/data/myresume.pdf",
    },
    {
      id: "resume_002",
      role: "React Developer",
      skills: ["React", "Redux", "JavaScript", "CSS"],
      summary: "Frontend developer skilled in building modern UI.",
      projects: "Created a real-time dashboard with React and Redux.",
      experiencelevel: "Junior",
      years_of_experience: "1",
      resume_url: null,
    },
  ];


    // 2. Send to ML service
    const report = await generateReport(jobs, resumes);

    console.log("Generated Report:", report);
    // 3. Send final result to frontend
    res.json({
      success: true,
      report,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({ error: "Matching failed" });
  }
};
