# File to test the model using ConvLSTM2D
# This code loads the trained model, processes the video stream and predicts actions in real-time.

# Import necessary libraries
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import time
from collections import deque, Counter
import pyttsx3
import threading

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Window setup
cv2.namedWindow('Gesture Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Gesture Detection', 800, 600)

# Get the current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load actions from text file
def load_actions(class_file):
    with open(class_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

# MediaPipe detection function
# This function processes the image and converts it to RGB format for MediaPipe processing.
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to extract keypoints from MediaPipe results
# This function extracts the x, y, z coordinates of the left and right hand landmarks and calculates the distance from the face landmarks.
def extract_keypoints(results):
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
    
    return np.concatenate([lh, rh, lh_dists, rh_dists])

# Function to draw landmarks on the image
# This function draws the detected landmarks on the image using MediaPipe drawing utilities.
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Function to reshape the sequence of keypoints to match the input shape of the model.
def process_sequence(sequence):
    return sequence.reshape(30, 12, 14, 1)

# Function to smooth the sequence of keypoints using exponential smoothing to reduce noise and improve prediction accuracy.
def smooth_sequence(sequence, alpha=0.3):
    smoothed = [sequence[0]]
    for i in range(1, len(sequence)):
        smoothed.append(alpha * sequence[i] + (1 - alpha) * smoothed[i - 1])
    return np.array(smoothed)

# Function to load training metadata (landmarks and labels) from numpy files.
def load_training_metadata(dataset_path):
    landmarks = np.load(os.path.join(dataset_path, 'landmarks.npy'))
    labels = np.load(os.path.join(dataset_path, 'labels.npy'))
    return landmarks, labels

# Function to validate the predicted class using training data.
# This function checks if the predicted class is valid by comparing the current sequence with training data using a similarity measure.
def validate_with_training_data(current_sequence, predicted_class, training_landmarks, training_labels):
    class_examples = training_landmarks[training_labels == predicted_class]
    if len(class_examples) == 0:
        return False
    similarities = [np.linalg.norm(example.flatten() - current_sequence.flatten()) for example in class_examples]
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    current_sim = np.min(similarities)
    return current_sim < mean_sim + 2 * std_sim

# Function to speak the predicted action using text-to-speech engine.
def speak_action(action, engine):
    engine.say(action)
    engine.runAndWait()

# Function for real-time gesture detection using the trained model.
# This function captures video from the webcam, processes each frame and predicts the action using the trained model.
def real_time_detection(model_path, actions, dataset_path):
    training_landmarks, training_labels = load_training_metadata(dataset_path)
    model = load_model(model_path)
    sequence = []
    predictions_deque = deque(maxlen=10)

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    cap = cv2.VideoCapture(0)

    cooldown = 0
    start_time = time.time()
    displayed_action = None
    display_start_time = 0
    display_duration = 3

    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.7) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            current_time = time.time()

            # Stabilizing period (ignore gestures)
            if current_time - start_time < 2:
                cv2.putText(image, "Stabilizing camera...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.imshow('Gesture Detection', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                continue

            status = "No detection"
            color = (0, 0, 255)

            if len(sequence) == 30 and cooldown == 0:
                smoothed_sequence = smooth_sequence(np.array(sequence))
                processed_sequence = process_sequence(smoothed_sequence)

                input_data = processed_sequence[np.newaxis, ...]
                res = model.predict(input_data)[0]
                pred_index = np.argmax(res)
                confidence = res[pred_index]

                if confidence > 0.5:
                    predictions_deque.append(pred_index)
                    most_common = Counter(predictions_deque).most_common(1)
                    if most_common[0][1] > 6:
                        validated_class = most_common[0][0]
                        is_valid = validate_with_training_data(processed_sequence, validated_class, training_landmarks, training_labels)
                        if is_valid:
                            displayed_action = actions[validated_class]
                            display_start_time = current_time
                            cooldown = 15
                            threading.Thread(target=speak_action, args=(displayed_action, engine), daemon=True).start()
                else:
                    predictions_deque.clear()

            # Handle action display
            if displayed_action and (current_time - display_start_time < display_duration):
                status = f"{displayed_action}"
                color = (0, 255, 0)

            cv2.putText(image, status, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            if cooldown > 0:
                cooldown -= 1

            cv2.imshow('Gesture Detection', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Main to run the real-time detection
# Loads the actions, model and dataset path and calls the real_time_detection function.
if __name__ == "__main__":
    dataset_path = os.path.join(BASE_DIR, '../Data Preprocessing/reduced_dataset/')
    actions_path = os.path.join(BASE_DIR, '../Data Preprocessing/class_names_reduced_dataset.txt')
    model_path = os.path.join(BASE_DIR, 'trained_model_reduced_dataset.h5')
    actions = load_actions(actions_path)
    real_time_detection(model_path, actions, dataset_path)