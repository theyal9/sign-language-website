# main.py
# FastAPI application for Sign Language Recognition System.
# Provides endpoints for learning videos, practice, real-time model predictions, and feedback collection.

# Import necessary libraries
import os
import random
import shutil
import requests
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form 
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect
from typing import List
from pydantic import BaseModel
import uvicorn
import tempfile
import time
import asyncio
import base64
import pyttsx3
import threading

# Initialize FastAPI app
app = FastAPI()

# Paths and Configuration
DATASET_PATH = "../Model/Data Preprocessing/reduced_dataset"
MODEL_PATH = "../Model/Model Development/trained_model_reduced_dataset.h5"
CLASS_NAMES_PATH = "../Model/Data Preprocessing/class_names_reduced_dataset.txt"

# MediaPipe and feedback
mp_holistic = mp.solutions.holistic

# Global flag to track if TTS is active
is_speaking = threading.Lock()

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
# Reads a video file, processes it using MediaPipe to extract hand landmarks
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
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
        
        # Face coordinates (use nose as reference point)
        if results.face_landmarks:
            face = results.face_landmarks.landmark[1]
            face_coords = np.array([face.x, face.y, face.z])
        else:
            face_coords = np.zeros(3)

        # Calculate Euclidean distance of each hand landmark from face
        lh_dists = np.linalg.norm(lh - face_coords, axis=1) if results.left_hand_landmarks else np.zeros(21)
        rh_dists = np.linalg.norm(rh - face_coords, axis=1) if results.right_hand_landmarks else np.zeros(21)

        lh = lh.flatten()
        rh = rh.flatten()
        
        keypoints = np.concatenate([lh, rh, lh_dists, rh_dists]).reshape(12, 14, 1)
        
        sequence.append(keypoints)
        sequence = sequence[-30:] 

    cap.release()
    holistic.close()
    
    if len(sequence) < 30:
        return None
    
    # Prepare input for model
    input_data = np.array(sequence[-30:]).reshape(1, 30, 12, 14, 1)
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

# WebSocket endpoint for real-time sign language recognition
# This endpoint accepts a video stream, processes it using MediaPipe to extract keypoints,
# and uses the trained model to predict the sign language in real-time.
# It returns the predicted sign and its confidence score.
@app.websocket("/ws/interpreter")
async def websocket_interpreter(websocket: WebSocket):
    await websocket.accept()
    holistic = mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.7)

    sequence = []

    try:
        while True:
            try:
                # Receive video frame from client
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=5.0)
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                keypoints = extract_keypoints(results)

                if np.all(keypoints == 0):
                    await websocket.send_json({
                        "prediction": "No sign detected",
                        "confidence": 0.0
                    })
                    continue

                sequence.append(keypoints)

                # Limit sequence length to 30 frames
                if len(sequence) == 30:
                    smoothed = smooth_sequence(np.array(sequence))
                    input_data = process_sequence(smoothed)[np.newaxis, ...]
                    res = model.predict(input_data)[0]
                    pred_index = np.argmax(res)
                    confidence = float(res[pred_index])
                    if confidence > 0.3:
                        prediction = actions[pred_index] 
                    else: 
                        prediction = "Uncertain"

                    # Send prediction and confidence back to client
                    await websocket.send_json({"prediction": prediction, "confidence": confidence})
                    speak_action(prediction)

                    await asyncio.sleep(2)  # Wait before next round
                    sequence.clear()
            
            # Handle exceptions
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except Exception as e:
                print(f"Processing error: {e}")
                break
    finally:
        holistic.close()

# Speak the predicted action using TTS
# Uses pyttsx3 to convert text to speech.
def speak_action(action):
    def run_tts():
        global is_speaking
        if not is_speaking.acquire(blocking=False):  
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  
            engine.say(action)
            engine.runAndWait()
            engine.stop()  # Ensure the engine is stopped after use
        except Exception as e:
            print(f"Error in TTS thread: {str(e)}")
        finally:
            is_speaking.release()  # Release the lock when done

    if action != "Uncertain":
        # Start a daemon thread to prevent blocking
        thread = threading.Thread(target=run_tts, daemon=True)
        thread.start()

# Smooth the sequence of keypoints using exponential smoothing
# Takes a sequence of keypoints and applies exponential smoothing to it.
def smooth_sequence(sequence, alpha=0.3):
    smoothed = [sequence[0]]
    for i in range(1, len(sequence)):
        smoothed.append(alpha * sequence[i] + (1 - alpha) * smoothed[i-1])
    return np.array(smoothed)

# Process the sequence of keypoints for model input
# Reshapes the sequence to match the model input shape.
def process_sequence(sequence):
    return sequence.reshape(30, 12, 14, 1)

# Extract keypoints from MediaPipe results
# Extracts left and right hand landmarks, face coordinates,
def extract_keypoints(results):
    # Explicit fallbacks for missing landmarks
    lh = np.zeros(21*3)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    
    rh = np.zeros(21*3)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    
    # Handle missing face landmarks
    face_coords = np.zeros(3)
    if results.face_landmarks:
        face = results.face_landmarks.landmark[1]
        face_coords = np.array([face.x, face.y, face.z])
    
    # Calculate distances safely
    lh_dists = np.zeros(21)
    if results.left_hand_landmarks:
        lh_dists = np.linalg.norm(lh.reshape(-1, 3) - face_coords, axis=1)
    
    rh_dists = np.zeros(21)
    if results.right_hand_landmarks:
        rh_dists = np.linalg.norm(rh.reshape(-1, 3) - face_coords, axis=1)
    
    return np.concatenate([lh, rh, lh_dists, rh_dists])

# Serve static content and dataset
app.mount("/dataset", StaticFiles(directory=DATASET_PATH), name="dataset")
app.mount("/static", StaticFiles(directory="public"), name="static")
app.mount("/", StaticFiles(directory="public", html=True), name="static")

# Development entry point
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=3000, reload=True)