import express from "express";
import cors from "cors";
import morgan from "morgan";
import path from "path";

export const setupMiddlewares = (app) => {
  app.use(cors());
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  if (process.env.NODE_ENV === "development") {
    app.use(morgan("dev"));
  }
};
