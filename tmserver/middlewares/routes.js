import jobRoute from "../routes/jobRoute.js";
import resumeRoute from "../routes/resumeRoute.js";
import analysisRoute from "../routes/analysisRoute.js";

export const setupRoutes = (app) => {
    app.get("/", (req, res) => {
        res.send("Welcome to the API");
    });
    app.use("/api/job",jobRoute)
    app.use("/api/resume",resumeRoute)
    app.use("/api/analysis",analysisRoute)
}