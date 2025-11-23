import mongoose from "mongoose";
import app from "./app.js";
import { configDotenv } from "dotenv";

configDotenv();

const DB = process.env.MONGODB_URI;

mongoose
  .connect(DB)
  .then(() => console.log("DB connection successful"))
  .catch((err) => console.log("DB connection error:", err));

const port = process.env.PORT || 3001;
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
