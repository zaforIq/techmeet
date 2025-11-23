import express from "express";
import { getresumeData,postresumeData } from "../controllers/resumeController.js";
import { upload } from "../middlewares/multer.js";

const router = express.Router();

router.route("/")
  .get(getresumeData)
  .post(upload.single('resume'),postresumeData);

export default router;