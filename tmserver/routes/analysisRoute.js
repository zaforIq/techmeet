import { runMatching } from "../controllers/analysisController.js";
import express from "express";

const router = express.Router();

router.route("/run-matching").get(runMatching);

export default router;