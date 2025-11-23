import express from "express";
import { setupMiddlewares } from "./middlewares/index.js";
import { setupRoutes } from "./middlewares/routes.js";

const app = express();

setupMiddlewares(app);
setupRoutes(app);

export default app;
