import mongoose from "mongoose";

const ResumeSchema = new mongoose.Schema({
  user_id: mongoose.Schema.Types.ObjectId,
  name: String,
  designation: String,
  skills: [String],
  summary: String,
  projects: String,
  resume_url: String,
  experience: String,
  embedding_vector: [Number],
});

module.exports = mongoose.model("Resume", ResumeSchema);
