import fetch from 'node-fetch';

export const getJobs = async (req, res) => {

}

export const createJob = async (req, res) => {
    try {
        const jobData = req.body;

        return res.status(201).json({
            success: true,
            message: "Job created successfully",
            data: jobData
        });
    } catch (error) {
        console.error("Error creating job:", error);
        return res.status(500).json({ success: false, error: error.message });
    }
    
}