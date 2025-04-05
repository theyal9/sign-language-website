# main.py
# FastAPI application for Sign Language Recognition System.
# Provides endpoints for learning videos, practice, real-time model predictions, and feedback collection.

# Import necessary libraries
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import random
import subprocess
import shutil
import requests
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from typing import List
from pydantic import BaseModel
import uvicorn
import tempfile
import time
from fastapi import BackgroundTasks
import subprocess

# Initialize FastAPI app
app = FastAPI()

# Paths and Configuration
DATASET_PATH = "../Model/Data Preprocessing/reduced_dataset"
MODEL_PATH = "../Model/Model Development/trained_model_reduced_dataset.h5"
CLASS_NAMES_PATH = "../Model/Data Preprocessing/class_names_reduced_dataset.txt"

# MediaPipe and feedback
mp_holistic = mp.solutions.holistic

# Load model and class names on startup
@app.on_event("startup")
async def load_model():
    global model, actions
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'r') as f:
        actions = [line.strip() for line in f.readlines()]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class VideoResponse(BaseModel):
    currentPage: int
    totalPages: int
    videos: List[dict]

class PracticeVideoResponse(BaseModel):
    videoPath: str
    correctAnswer: str
    options: List[str]

# Serve landing page HTML
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("public/index.html", encoding="utf-8") as f:
        return f.read()

# Fetch paginated list of learning videos
# Returns a list of video titles and paths for the current page.
# Each page contains a 3 videos.
@app.get("/api/videos", response_model=VideoResponse)
async def get_videos(page: int = 1):
    try:
        limit = 3
        all_folders = [f for f in os.listdir(DATASET_PATH) 
                      if os.path.isdir(os.path.join(DATASET_PATH, f))]
        
        start_index = (page - 1) * limit
        end_index = page * limit
        video_folders = all_folders[start_index:end_index]

        videos = []
        for folder in video_folders:
            folder_path = os.path.join(DATASET_PATH, folder)
            video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
            if video_files:
                title = ' '.join([word.capitalize() for word in folder.replace('_', '-').split('-')])
                videos.append({
                    "title": title,
                    "path": f"/dataset/{folder}/{video_files[0]}"
                })

        return {
            "currentPage": page,
            "totalPages": (len(all_folders) + limit - 1) // limit,
            "videos": videos
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Fetch a random practice video
# Returns a video, a correct answer and 3 wrong answers.
@app.get("/api/practice-video", response_model=PracticeVideoResponse)
async def get_practice_video():
    try:
        all_folders = [f for f in os.listdir(DATASET_PATH) 
                      if os.path.isdir(os.path.join(DATASET_PATH, f))]
        random_folder = random.choice(all_folders)
        video_files = [f for f in os.listdir(os.path.join(DATASET_PATH, random_folder)) 
                      if f.endswith(".mp4")]
        
        wrong_answers = random.sample(
            [f for f in all_folders if f != random_folder], 
            3
        )

        title_formatter = lambda f: ' '.join(
            [word.capitalize() for word in f.replace('_', '-').split('-')]
        )

        return {
            "videoPath": f"/dataset/{random_folder}/{video_files[0]}",
            "correctAnswer": title_formatter(random_folder),
            "options": random.sample([
                title_formatter(random_folder),
                *[title_formatter(f) for f in wrong_answers]
            ], 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Validate a word using an external dictionary API
# Returns a boolean indicating if the word is valid or not.
@app.get("/api/validate-word")
async def validate_word(word: str):
    try:
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
        return {"valid": response.status_code == 200}
    except:
        return {"valid": False}

# Process video to extract keypoints for model prediction
# This function reads a video file, processes it using MediaPipe to extract hand landmarks
# and prepares the data for model input.
# It returns a numpy array of shape (1, 30, 9, 14, 1) for model prediction.
def process_video(video_path: str):
    sequence = []
    cap = cv2.VideoCapture(video_path)
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        # Extract keypoints
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        keypoints = np.concatenate([lh, rh]).reshape(9, 14, 1)
        
        sequence.append(keypoints)
        sequence = sequence[-30:] 

    cap.release()
    holistic.close()
    
    if len(sequence) < 30:
        return None
    
    # Prepare input for model
    input_data = np.array(sequence[-30:]).reshape(1, 30, 9, 14, 1)
    return input_data

# Predict the sign language from the video
# This endpoint accepts a video file, processes it to extract keypoints
# and uses the trained model to predict the sign language.
# It returns the predicted sign and its confidence score.
@app.post("/api/predict")
async def predict(video: UploadFile = File(...)):
    try:
        # Save uploaded file to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await video.read()
            tmp.write(content)
            temp_path = tmp.name

        # Process video
        input_data = process_video(temp_path)
        if input_data is None:
            raise HTTPException(status_code=400, detail="Video too short")

        # Make prediction
        res = model.predict(input_data)[0]
        pred_index = np.argmax(res)
        confidence = res[pred_index]

        if confidence > 0.3:
            prediction = actions[pred_index]
        else:
            prediction = "Uncertain"

        # Cleanup
        os.unlink(temp_path)
        
        return JSONResponse(content={"prediction": prediction, "confidence": float(confidence)})

    except Exception as e:
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

# Store feedback data
# This endpoint accepts a video file and feedback data (correct sign and predicted sign).
@app.post("/api/store-data")
async def store_data(
    video: UploadFile = File(...),
    correct_sign: str = Form(...),
    predicted_sign: str = Form(...)
):
    try:
        # Create target directory
        target_dir = os.path.join(DATASET_PATH, correct_sign.lower().replace(" ", "_"))
        os.makedirs(target_dir, exist_ok=True)
        
        # Generate unique filename
        filename = f"{int(time.time())}_{correct_sign}.mp4"
        file_path = os.path.join(target_dir, filename)
        
        # Save video
        with open(file_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
            
        return JSONResponse(content={"status": "success"})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Launches Python script for interpreting sign language in a new console window using subprocess.
# A background task that runs independently of the main FastAPI application.
@app.get("/api/start-interpreter")
async def start_interpreter(background_tasks: BackgroundTasks):
    try:
        def run_script():
            try:
                script_path = os.path.abspath("../Model/Model Development/model_testing.py")
                result = subprocess.run(
                    ["python", script_path],
                    shell=True,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    text=True
                )
                # Log outputs for debugging
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                if result.returncode != 0:
                    print(f"Script failed with code {result.returncode}")
            except Exception as e:
                print(f"Subprocess error: {str(e)}")

        background_tasks.add_task(run_script)
        return {"status": "Interpreter launched"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static content and dataset
app.mount("/dataset", StaticFiles(directory=DATASET_PATH), name="dataset")
app.mount("/static", StaticFiles(directory="public"), name="static")
app.mount("/", StaticFiles(directory="public", html=True), name="static")

# Development entry point
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=3000, reload=True)