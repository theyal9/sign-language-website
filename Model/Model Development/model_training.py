import cv2
import numpy as np
import os
import mediapipe as mp
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, GRU, ConvLSTM2D, MaxPooling3D, TimeDistributed, Dropout
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

# Initialize MediaPipe Hands model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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

def save_landmarks(sequences, labels, actions, dataset_path):
    np.save(os.path.join(dataset_path, 'landmarks.npy'), sequences)
    np.save(os.path.join(dataset_path, 'labels.npy'), labels)
    df = pd.DataFrame([{'action': actions[label], 'landmarks': sequence.tolist()} for sequence, label in zip(sequences, labels)])
    df.to_csv(os.path.join(dataset_path, 'landmarks.csv'), index=False)

def process_videos(dataset_path):
    actions = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    sequence_length = 30

    for action in actions:
        action_path = os.path.join(dataset_path, action)
        for video_file in os.listdir(action_path):
            video_path = os.path.join(action_path, video_file)
            cap = cv2.VideoCapture(video_path)
            sequence = []
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    image, results = mediapipe_detection(frame, holistic)
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    if len(sequence) == sequence_length:
                        sequences.append(sequence)
                        labels.append(label_map[action])
                        sequence = []
            cap.release()

    X = np.array(sequences)
    y = np.array(labels)

    # Reshape X for ConvLSTM2D
    X = X.reshape(X.shape[0], 30, 9, 14, 1) 

    save_landmarks(X, y, actions, dataset_path)
    return X, y, actions

def augment_data(X, y):
    augmented_X, augmented_y = [], []
    for i in range(len(X)):
        for _ in range(10): 
            sample = X[i].copy()
            
            # Apply random noise
            noise = np.random.normal(0, 0.02, sample.shape)
            augmented_sample = sample + noise

            # Apply slight scaling
            scale_factor = np.random.uniform(0.9, 1.1)
            augmented_sample *= scale_factor

            augmented_X.append(augmented_sample)
            augmented_y.append(y[i])

    return np.array(augmented_X), np.array(augmented_y)

def train_convlstm2d(X_train, y_train, actions):
    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(2,2), activation='tanh', padding='same', 
                         return_sequences=True, input_shape=(30, 9, 14, 1)))
    model.add(MaxPooling3D(pool_size=(1,2,2), padding='same'))
    model.add(TimeDistributed(Dropout(0.3)))

    model.add(ConvLSTM2D(filters=64, kernel_size=(2,2), activation='tanh', padding='same', return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1,2,2), padding='same'))
    model.add(TimeDistributed(Dropout(0.3)))

    model.add(ConvLSTM2D(filters=128, kernel_size=(2,2), activation='tanh', padding='same', return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1,2,2), padding='same'))
    model.add(TimeDistributed(Dropout(0.3)))

    model.add(ConvLSTM2D(filters=256, kernel_size=(1,1), activation='tanh', return_sequences=False))
    model.add(Flatten())
    model.add(Dense(len(actions), activation='softmax'))

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])  # Add validation split and early stopping

    return model

if __name__ == "__main__":
    DATASET_PATH = '../Data Preprocessing/dataset/'
    print("Processing videos and extracting keypoints...")
    X, y, actions = process_videos(DATASET_PATH)
    
    print("Augmenting data...")
    X, y = augment_data(X, y)
    
    print("Splitting data and training models...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    
    print("Training ConvLSTM2D model...")
    convlstm2d_model = train_convlstm2d(X_train, y_train, actions)
    convlstm2d_model.save('convlstm2d_model.h5')
    
    print("Evaluating ConvLSTM2D model...")
    y_pred = np.argmax(convlstm2d_model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"ConvLSTM2D Model Accuracy: {acc*100:.2f}%")