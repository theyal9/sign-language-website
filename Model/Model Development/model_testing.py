import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Initialize MediaPipe Hands model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# At the top of real_time_detection function
cv2.namedWindow('Gesture Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Gesture Detection', 800, 600)

# Get the current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    return keypoints.reshape(9, 14, 1)

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
                # Reshape for model
                input_data = np.array(sequence).reshape(1, 30, 9, 14, 1) 
                
                res = model.predict(input_data)[0]
                pred_index = np.argmax(res)
                confidence = res[pred_index]

                # Confidence threshold
                if confidence > 0.5:
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
    actions_path = os.path.join(BASE_DIR, '../Data Preprocessing/class_names_reduced_dataset.txt')
    model_path = os.path.join(BASE_DIR, 'trained_model_reduced_dataset.h5')
    actions = load_actions(actions_path)
    real_time_detection(model_path, actions)

# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import os

# # Initialize MediaPipe Hands model
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# # Window setup
# cv2.namedWindow('Gesture Detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Gesture Detection', 800, 600)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# def load_actions(class_file):
#     with open(class_file, 'r') as f:
#         return [line.strip() for line in f.readlines()]

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
#     return np.concatenate([lh, rh])

# def draw_landmarks(image, results):
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# def process_sequence(sequence):
#     """Matches the training data preprocessing exactly"""
#     # Normalization (if you did any during training)
#     # sequence = (sequence - np.mean(sequence)) / np.std(sequence) 
#     return sequence.reshape(30, 9, 14, 1)

# def load_training_metadata(dataset_path):
#     """Load the saved landmarks and labels from training"""
#     landmarks = np.load(os.path.join(dataset_path, 'landmarks.npy'))
#     labels = np.load(os.path.join(dataset_path, 'labels.npy'))
#     return landmarks, labels

# def validate_with_training_data(current_sequence, predicted_class, training_landmarks, training_labels):
#     """Compare with actual training examples"""
#     class_examples = training_landmarks[training_labels == predicted_class]
    
#     if len(class_examples) == 0:
#         return False
    
#     # Calculate similarity with class prototypes
#     similarities = []
#     for example in class_examples:
#         # Use DTW or simple Euclidean distance
#         similarity = np.linalg.norm(example.flatten() - current_sequence.flatten())
#         similarities.append(similarity)
    
#     # Check if within 2 standard deviations of mean similarity
#     mean_sim = np.mean(similarities)
#     std_sim = np.std(similarities)
#     current_sim = np.min(similarities)
    
#     return current_sim < mean_sim + 2*std_sim

# def real_time_detection(model_path, actions, dataset_path):
#     # Load training metadata
#     training_landmarks, training_labels = load_training_metadata(dataset_path)
    
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
#                 # Process exactly like training data
#                 processed_sequence = process_sequence(np.array(sequence))
#                 input_data = processed_sequence[np.newaxis, ...]
                
#                 res = model.predict(input_data)[0]
#                 pred_index = np.argmax(res)
#                 confidence = res[pred_index]

#                 validation_result = False
#                 if confidence > 0.5:
#                     # Validate against training data
#                     validation_result = validate_with_training_data(
#                         processed_sequence, 
#                         pred_index,
#                         training_landmarks,
#                         training_labels
#                     )
                
#                 if validation_result:
#                     status = f"Confirmed {actions[pred_index]} ({confidence:.2f})"
#                     color = (0, 255, 0)
#                 else:
#                     status = f"Uncertain {actions[pred_index]} ({confidence:.2f})" if confidence > 0.5 else "No detection"
#                     color = (0, 0, 255)
                
#                 cv2.putText(image, status, (10, 100), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
#             cv2.imshow('Gesture Detection', image)
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     DATASET_PATH = '../Data Preprocessing/reduced_dataset/'
#     actions_path = os.path.join(BASE_DIR, '../Data Preprocessing/class_names_reduced_dataset.txt')
#     model_path = os.path.join(BASE_DIR, 'trained_model_reduced_dataset.h5')
#     actions = load_actions(actions_path)
#     real_time_detection(model_path, actions, DATASET_PATH)