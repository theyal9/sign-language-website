import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize MediaPipe Hands model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def load_actions(class_file):
    with open(class_file, 'r') as f:
        actions = [line.strip() for line in f.readlines()]
    return actions

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    keypoints = np.concatenate([lh, rh])
    
    # Reshape for ConvLSTM2D (9x14 grid as used in training)
    return keypoints.reshape(9, 14, 1)  # Changed from 6x21 to 9x14

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def real_time_detection(model_path, actions):
    model = load_model(model_path)
    sequence = []
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Maintain 30-frame sequence

            if len(sequence) == 30:
                # Reshape for model expecting (None, 30, 9, 14, 1)
                input_data = np.array(sequence).reshape(1, 30, 9, 14, 1)  # Updated shape
                
                res = model.predict(input_data)[0]
                pred_index = np.argmax(res)
                confidence = res[pred_index]

                # Confidence threshold
                if confidence > 0.1:
                    predicted_action = actions[pred_index]
                else:
                    predicted_action = "Uncertain"

                # Display info
                cv2.putText(image, f"{predicted_action} ({confidence:.2f})", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Gesture Detection', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # actions = load_actions('../Data Preprocessing/classes_names.txt')
    # real_time_detection('convlstm2d_model_smaller_dataset.h5', actions)
    actions = load_actions('../Data Preprocessing/class_names.txt')
    real_time_detection('convlstm2d_model_larger_dataset.h5', actions)