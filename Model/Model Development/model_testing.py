# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import os

# # Initialize MediaPipe Hands model
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# # At the top of real_time_detection function
# cv2.namedWindow('Gesture Detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Gesture Detection', 800, 600)

# # Get the current script directory
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# def load_actions(class_file):
#     with open(class_file, 'r') as f:
#         actions = [line.strip() for line in f.readlines()]
#     return actions

# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results

# def extract_keypoints(results):
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
#     keypoints = np.concatenate([lh, rh])
    
#     # Reshape for ConvLSTM2D (9x14 grid as used in training)
#     return keypoints.reshape(9, 14, 1)

# def draw_landmarks(image, results):
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# def real_time_detection(model_path, actions):
#     model = load_model(model_path)
#     sequence = []
#     cap = cv2.VideoCapture(0)
    
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             image, results = mediapipe_detection(frame, holistic)
#             draw_landmarks(image, results)
#             keypoints = extract_keypoints(results)
#             sequence.append(keypoints)
#             sequence = sequence[-30:]  # Maintain 30-frame sequence

#             if len(sequence) == 30:
#                 # Reshape for model
#                 input_data = np.array(sequence).reshape(1, 30, 9, 14, 1) 
                
#                 res = model.predict(input_data)[0]
#                 pred_index = np.argmax(res)
#                 confidence = res[pred_index]

#                 # Confidence threshold
#                 if confidence > 0.5:
#                     predicted_action = actions[pred_index]
#                 else:
#                     predicted_action = "Uncertain"

#                 # Display info
#                 cv2.putText(image, f"{predicted_action} ({confidence:.2f})", (10, 50), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
#             cv2.imshow('Gesture Detection', image)
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     actions_path = os.path.join(BASE_DIR, '../Data Preprocessing/class_names_reduced_dataset.txt')
#     model_path = os.path.join(BASE_DIR, 'trained_model_reduced_dataset.h5')
#     actions = load_actions(actions_path)
#     real_time_detection(model_path, actions)

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import time
from collections import deque, Counter

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Window setup
cv2.namedWindow('Gesture Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Gesture Detection', 800, 600)

# Get the current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_actions(class_file):
    with open(class_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

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
    return np.concatenate([lh, rh])

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def process_sequence(sequence):
    return sequence.reshape(30, 9, 14, 1)

def smooth_sequence(sequence, alpha=0.3):
    smoothed = [sequence[0]]
    for i in range(1, len(sequence)):
        smoothed.append(alpha * sequence[i] + (1 - alpha) * smoothed[i - 1])
    return np.array(smoothed)

def load_training_metadata(dataset_path):
    landmarks = np.load(os.path.join(dataset_path, 'landmarks.npy'))
    labels = np.load(os.path.join(dataset_path, 'labels.npy'))
    return landmarks, labels

def validate_with_training_data(current_sequence, predicted_class, training_landmarks, training_labels):
    class_examples = training_landmarks[training_labels == predicted_class]
    if len(class_examples) == 0:
        return False
    similarities = [np.linalg.norm(example.flatten() - current_sequence.flatten()) for example in class_examples]
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    current_sim = np.min(similarities)
    return current_sim < mean_sim + 2 * std_sim

def real_time_detection(model_path, actions, dataset_path):
    training_landmarks, training_labels = load_training_metadata(dataset_path)
    model = load_model(model_path)
    sequence = []
    predictions_deque = deque(maxlen=10)
    cap = cv2.VideoCapture(0)

    cooldown = 0
    start_time = time.time()
    displayed_action = None
    display_start_time = 0
    display_duration = 2

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
                            cooldown = 15  # optional cooldown to avoid repeats
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

if __name__ == "__main__":
    dataset_path = os.path.join(BASE_DIR, '../Data Preprocessing/reduced_dataset/')
    actions_path = os.path.join(BASE_DIR, '../Data Preprocessing/class_names_reduced_dataset.txt')
    model_path = os.path.join(BASE_DIR, 'trained_model_reduced_dataset.h5')
    actions = load_actions(actions_path)
    real_time_detection(model_path, actions, dataset_path)