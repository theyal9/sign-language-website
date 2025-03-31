# import cv2
# import numpy as np
# import mediapipe as mp
# from tensorflow.keras.models import load_model

# mp_holistic = mp.solutions.holistic

# def load_actions(class_file):
#     with open(class_file, 'r') as f:
#         return [line.strip() for line in f.readlines()]

# def process_frame(image, holistic):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = holistic.process(image)
#     return results

# def extract_keypoints(results):
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
#     return np.concatenate([lh, rh]).reshape(9, 14, 1)

# def predict_sign(video_path):
#     model = load_model('convlstm2d_model.h5')
#     actions = load_actions('../Data Preprocessing/class_names.txt')
#     sequence = []
    
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return "Error: Could not open video"
    
#     with mp_holistic.Holistic(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     ) as holistic:
#         frame_count = 0
#         while cap.isOpened() and frame_count < 30:
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             results = process_frame(frame, holistic)
#             keypoints = extract_keypoints(results)
#             sequence.append(keypoints)
#             frame_count += 1

#     cap.release()
    
#     # Pad sequence if needed
#     while len(sequence) < 30:
#         sequence.append(np.zeros((9, 14, 1)))
    
#     input_data = np.array(sequence).reshape(1, 30, 9, 14, 1)
#     prediction = model.predict(input_data)
#     return actions[np.argmax(prediction)]

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 2:
#         print("Usage: python model.py <video_path>")
#         sys.exit(1)
#     print(predict_sign(sys.argv[1]))
# model.py
import sys
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic

def load_actions(class_file):
    with open(class_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def process_frame(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    return results

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh]).reshape(9, 14, 1)

def predict_sign(video_path):
    model = load_model('convlstm2d_model.h5')
    actions = load_actions('../Data Preprocessing/class_names.txt')
    sequence = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video"
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        frame_count = 0
        while cap.isOpened() and frame_count < 30:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = process_frame(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            frame_count += 1

    cap.release()
    
    # Pad sequence if needed
    while len(sequence) < 30:
        sequence.append(np.zeros((9, 14, 1)))
    
    input_data = np.array(sequence).reshape(1, 30, 9, 14, 1)
    prediction = model.predict(input_data)
    return actions[np.argmax(prediction)]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python model.py <video_path>")
        sys.exit(1)
    
    try:
        result = predict_sign(sys.argv[1])
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)