import express from "express"
import { getJobs, createJob } from "../controllers/jobController.js"

const router = express.Router()

router.route("/")
  .get(getJobs)
  .post(createJob)

export default router