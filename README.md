# sign-language-website
The AI Sign Language Interpreter website 
This is a real-time AI-powered sign language recognition system with a browser-based interface. It uses a ConvLSTM2D deep learning model integrated via a FastAPI backend to interpret American Sign Language (ASL) gestures from webcam input.

Git Repository Link: https://github.com/theyal9/sign-language-website

Dataset link: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed/data?select=WLASL_v0.3.json

Data preprocessing:
1.  Run download_videos.py to download videos from dataset.
2.  Run remove_non_mp4.py to remove videos that are not in mp4 format.
3.  Run remove_corrupted_mp4.py to delete corrupted videos.
4.  Run classes.py to extract classes from cleaned dataset.

To run website:
1.  Start the server:
    python main.py
2.  Open in browser:
    Navigate to: http://localhost:3000/

Project Structure
Model/ - Root directory for the model development. 
    Data Preprocessing/ - Includes all Python scripts responsible for cleaning, organizing and processing raw video data. Scripts here manage tasks like removing corrupted files, normalizing landmarks, extracting class labels and augmenting video samples. 

    Model Development/ - Contains the core deep learning scripts for training, evaluating, and saving the ConvLSTM2D model. Training logs, loss curves and performance reports are also generated from this folder. 

Website/ -  Root directory for the web application. 
public/ - The main landing page that connects users to the modules. 
    css/ - Contains styling sheets for page layouts and responsiveness. 
    js/ -  Contains the main JavaScript files that handle client-side interactivity, including WebSocket communications and UI logic. 

main.py - The FastAPI entry point that handles backend routes, video preprocessing endpoints, model inference logic and feedback storage.

Requirements
- Python 3.8+
- Node.js 
- pip packages:
  - tensorflow
  - fastapi
  - uvicorn
  - mediapipe
  - opencv-python
  - pyttsx3

Model Architecture
- Type: ConvLSTM2D (spatio-temporal)
- Input: Sequences of MediaPipe keypoints (30 frames)
- Output: Predicted sign and confidence score
- Trained on: WLASL dataset (processed)
- Accuracy: ~95%

Features
- Real-time ASL gesture recognition via webcam
- Learning module with categorized instructional videos
- Practice mode with multiple-choice quizzes and feedback
- Prediction correction with user feedback storage
- Responsive UI (HTML/CSS/JS)
- FastAPI backend with WebSocket-based real-time inference