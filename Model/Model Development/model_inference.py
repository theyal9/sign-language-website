# import cv2
# import numpy as np
# import os
# import mediapipe as mp
# from tensorflow.keras.models import load_model

# mp_holistic = mp.solutions.holistic

# def process_video_for_inference(video_path):
#     sequence = []
#     sequence_length = 30
    
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         cap = cv2.VideoCapture(video_path)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             image, results = mediapipe_detection(frame, holistic)
#             keypoints = extract_keypoints(results)
#             sequence.append(keypoints)
            
#             if len(sequence) == sequence_length:
#                 break
                
#         cap.release()
    
#     # Pad sequence if needed
#     while len(sequence) < sequence_length:
#         sequence.append(np.zeros(126))  # 126 = 21*3*2 hands
    
#     return np.array([sequence]).reshape(1, 30, 9, 14, 1)

# def predict_sign(video_path):
#     model = load_model('../Model/Model Development/convlstm2d_model.h5')
#     processed_video = process_video_for_inference(video_path)
#     prediction = model.predict(processed_video)
#     return np.argmax(prediction, axis=1)[0]

# if __name__ == "__main__":
#     import sys
#     video_path = sys.argv[1]
#     # To use absolute path for dataset
#     dataset_path = os.path.abspath('C:/Users/HP/Desktop/27 03 - Sign language/dataset') 
#     actions = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
#     prediction_idx = predict_sign(video_path)
#     print(actions[prediction_idx])
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import time

mp_holistic = mp.solutions.holistic

def load_actions():
    dataset_path = os.path.abspath('../../Model/Data preprocessing/dataset')
    return [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

def process_frame(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def predict_sign(video_path):
    model = load_model('convlstm2d_model.h5')
    actions = load_actions()
    sequence = []
    
    cap = cv2.VideoCapture(video_path)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            _, results = process_frame(frame, holistic)
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
            keypoints = np.concatenate([lh, rh]).reshape(9, 14, 1)
            
            sequence.append(keypoints)
            sequence = sequence[-30:]

    # Pad sequence if needed
    while len(sequence) < 30:
        sequence.append(np.zeros((9, 14, 1)))
    
    prediction = model.predict(np.array(sequence).reshape(1, 30, 9, 14, 1))
    return actions[np.argmax(prediction)]

if __name__ == "__main__":
    import sys
    print(predict_sign(sys.argv[1]))