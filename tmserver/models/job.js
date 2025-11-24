import mongoose from "mongoose";

const JobSchema = new mongoose.Schema({
  recruiter_id: mongoose.Schema.Types.ObjectId,
  title: String,
  company: String,
  skills: [String],
  responsibilities: String,
  experiencelevel: String,
  keyWorkds: [String],
  location: String,
  salary: String,
  job_type: String,
  description: String,
  applied_candidates: [{ type: mongoose.Schema.Types.ObjectId, ref: "Resume" }],
  created_at: { type: Date, default: Date.now },
});

module.exports = mongoose.model("Job", JobSchema);
