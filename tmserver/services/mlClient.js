import axios from "axios";

const ML_URL = "http://localhost:8000";

export const generateReport = async (jobs, resumes) => {
    try {
      console.log("Sending data to ML service:", { jobs, resumes }); 
      const response = await axios.post(`${ML_URL}/generate-report`, {
        jobs,
        resumes,
      });

      return response.data;
    } catch (error) {
      console.error("ML Service Error:", error.response?.data || error.message);
      throw new Error("ML_GENERATE_REPORT_FAILED");
    }
  }