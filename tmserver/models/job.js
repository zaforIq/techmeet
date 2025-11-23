import mongoose from "mongoose";

const JobSchema = new mongoose.Schema({
  title: String,
  company: String,
  skills: [String],
  responsibilities: String,
  experiencelevel: String,
  keyWorkds: [String],
  location: String,
  salary: String,
  job_type: String,
  created_at: { type: Date, default: Date.now },
});

module.exports = mongoose.model("Job", JobSchema);
