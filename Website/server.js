const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
const app = express();
const port = 3000;
// Add to store-data endpoint
const { exec } = require('child_process');

// Path to your dataset
const datasetPath = 'C:/Users/HP/Desktop/27 03 - Sign language/dataset';

// Middleware
app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// Serve videos from external dataset
app.use('/dataset', express.static(datasetPath));

app.get('/api/videos', (req, res) => {
    try {
        const page = parseInt(req.query.page) || 1;
        const limit = 3;
        const allFolders = fs.readdirSync(datasetPath)
            .filter(folder => fs.statSync(path.join(datasetPath, folder)).isDirectory());

        const startIndex = (page - 1) * limit;
        const endIndex = page * limit;
        const videoFolders = allFolders.slice(startIndex, endIndex);

        const videos = videoFolders.map(folder => {
            const videoFiles = fs.readdirSync(path.join(datasetPath, folder))
                .filter(file => file.endsWith('.mp4'));
            
            if(videoFiles.length === 0) return null;

            return {
                title: folder.split(/[-_]/).map(word => 
                    word[0].toUpperCase() + word.slice(1)
                ).join(' '),
                path: `/dataset/${folder}/${videoFiles[0]}`
            };
        }).filter(Boolean);

        res.json({
            currentPage: page,
            totalPages: Math.ceil(allFolders.length / limit),
            videos
        });
    } catch (error) {
        console.error('Error loading videos:', error);
        res.status(500).json({ error: 'Error loading videos' });
    }
});

// New endpoint for practice questions
app.get('/api/practice-video', (req, res) => {
    try {
        const allFolders = fs.readdirSync(datasetPath)
            .filter(folder => fs.statSync(path.join(datasetPath, folder)).isDirectory());

        // Get random folder
        const randomFolder = allFolders[Math.floor(Math.random() * allFolders.length)];
        const videoFiles = fs.readdirSync(path.join(datasetPath, randomFolder))
            .filter(file => file.endsWith('.mp4'));

        // Generate wrong answers
        const wrongAnswers = allFolders
            .filter(f => f !== randomFolder)
            .sort(() => 0.5 - Math.random())
            .slice(0, 3);

        res.json({
            videoPath: `/dataset/${randomFolder}/${videoFiles[0]}`,
            correctAnswer: randomFolder.split(/[-_]/).map(word => 
                word[0].toUpperCase() + word.slice(1)
            ).join(' '),
            options: [
                randomFolder.split(/[-_]/).map(word => 
                    word[0].toUpperCase() + word.slice(1)
                ).join(' '),
                ...wrongAnswers.map(folder => folder.split(/[-_]/).map(word => 
                    word[0].toUpperCase() + word.slice(1)
                ).join(' '))
            ].sort(() => 0.5 - Math.random())
        });
    } catch (error) {
        console.error('Error loading practice video:', error);
        res.status(500).json({ error: 'Error loading practice video' });
    }
});

const ffmpeg = require('fluent-ffmpeg');
app.post('/api/store-data', async (req, res) => {
    try {
        const { videoData, correctSign, predictedSign } = req.body;
        const targetFolder = path.join(datasetPath, correctSign);
        
        if (!fs.existsSync(targetFolder)) {
            fs.mkdirSync(targetFolder, { recursive: true });
        }

        // Create temporary file name
        const tempFileName = `temp-${Date.now()}.webm`;
        const outputFileName = `user-${Date.now()}.mp4`;
        
        // Write raw video data
        await fs.promises.writeFile(tempFileName, Buffer.from(videoData, 'base64'));

        // Convert to MP4 using ffmpeg
        await new Promise((resolve, reject) => {
            ffmpeg(tempFileName)
                .outputOptions('-c:v copy')
                .save(path.join(targetFolder, outputFileName))
                .on('end', resolve)
                .on('error', reject);
        });

        // Cleanup temp file
        fs.unlinkSync(tempFileName);

        feedbackCount++;
        if(feedbackCount >= 100) { // Retrain every 100 corrections
            await retrain_model();
        }

        res.json({ success: true });
    } catch (error) {
        console.error('Error storing data:', error);
        res.status(500).json({ error: 'Error storing data' });
    }
});

// Oxford Dictionary validation
const axios = require('axios');
app.get('/api/validate-word', async (req, res) => {
    const { word } = req.query;
    try {
        const response = await axios.get(`https://api.dictionaryapi.dev/api/v2/entries/en/${word}`);
        res.json({ valid: response.data.length > 0 });
    } catch {
        res.json({ valid: false });
    }
});

const { spawn } = require('child_process');

// Prediction endpoint
app.post('/api/predict', (req, res) => {
    const { videoData } = req.body;
    const tempVideoPath = path.join(__dirname, 'temp.mp4'); // Changed to .mp4
    fs.writeFileSync(tempVideoPath, Buffer.from(videoData, 'base64'));

    // Use absolute path to model_inference.py
    const inferenceScript = path.join(
        __dirname, 
        '../Model/Model Development/model_inference.py'
    );
    
    const pythonProcess = spawn('python', [
        inferenceScript,
        tempVideoPath
    ], {
        cwd: path.join(__dirname, '../Model/Model Development')  // Set working directory
    });

    let prediction = '';
    pythonProcess.stdout.on('data', (data) => {
        prediction += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python error: ${data}`);
    });

    pythonProcess.on('close', () => {
        fs.unlinkSync(tempVideoPath);
        res.json({ prediction: prediction.trim() });
    });
});

let feedbackCount = 0;

async function retrain_model() {
    feedbackCount = 0;
    const retrainProcess = spawn('python', ['model_training.py']);
    
    retrainProcess.stdout.on('data', (data) => {
        console.log(`Retraining: ${data}`);
    });

    retrainProcess.stderr.on('data', (data) => {
        console.error(`Retraining error: ${data}`);
    });
}

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});