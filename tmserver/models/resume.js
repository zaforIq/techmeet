import mongoose from "mongoose";

const ResumeSchema = new mongoose.Schema({
  user_id: mongoose.Schema.Types.ObjectId,
  name: String,
  email: String,
  phone: String,
  education: String,
  certifications: [String],
  achievements: [String],
  designation: String,
  skills: [String],
  summary: String,
  projects: String,
  resume_url: String,
  experience: String,
  embedding_vector: [Number],
  offered_interviews: [{ type: mongoose.Schema.Types.ObjectId, ref: "Job" }],
  created_at: { type: Date, default: Date.now },
});

module.exports = mongoose.model("Resume", ResumeSchema);
