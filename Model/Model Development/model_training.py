# File to train the model using ConvLSTM2D
# This code processes the video dataset, extracts keypoints using MediaPipe, augments the data, and trains a ConvLSTM2D model.
# The model is then evaluated and the results are saved to a CSV file.
# The code includes functions for data augmentation and saving landmarks to a CSV file.

# Import necessary libraries
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
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Initialize MediaPipe Hands model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to detect keypoints using MediaPipe
def mediapipe_detection(image, model):
    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    # Process the image to detect keypoints
    results = model.process(image)
    image.flags.writeable = True
    # Convert image back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to extract keypoints from the detected hands
def extract_keypoints(results):
    # Extract landmarks
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
    
    # Face coordinates (use nose as reference point)
    if results.face_landmarks:
        face_point = face[1]
    else:
        face_point = np.zeros(3)

    # Compute distances from face point to each hand landmark
    lh_dist = np.linalg.norm(lh - face_point, axis=1) if lh.any() else np.zeros(21)
    rh_dist = np.linalg.norm(rh - face_point, axis=1) if rh.any() else np.zeros(21)

    # Flatten all features
    return np.concatenate([lh.flatten(), rh.flatten(), lh_dist, rh_dist])

# Function to save landmarks and labels to a CSV file
def save_landmarks(sequences, labels, actions, dataset_path):
    # Save sequences (landmarks) and labels as numpy arrays
    np.save(os.path.join(dataset_path, 'landmarks.npy'), sequences)
    np.save(os.path.join(dataset_path, 'labels.npy'), labels)
    # Create a dataframe to save a detailed csv file of landmarks and corresponding actions
    df = pd.DataFrame([{'action': actions[label], 'landmarks': sequence.tolist()} for sequence, label in zip(sequences, labels)])
    df.to_csv(os.path.join(dataset_path, 'landmarks.csv'), index=False)

# Function to process the video dataset and extract keypoints from each frame
def process_videos(dataset_path):
    # List all action folders in the dataset directory, create a label map
    # and initialize empty lists for sequences and labels and set length of each sequence of frames
    actions = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    sequence_length = 30

    # Loop through each action, process the videos and extract keypoints
    for action in actions:
        action_path = os.path.join(dataset_path, action)
        for video_file in os.listdir(action_path):
            video_path = os.path.join(action_path, video_file)
            cap = cv2.VideoCapture(video_path)
            sequence = []

            # Detect landmarks for each frame using MediaPipe Holistic model and extract keypoints
            # Append the keypoints to the sequence list
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

    # Reshape X for ConvLSTM2D input
    # The input shape for ConvLSTM2D is (samples, time_steps, rows, cols, channels)
    X = X.reshape(X.shape[0], 30, 12, 14, 1)

    save_landmarks(X, y, actions, dataset_path)
    return X, y, actions

# Function to augment the data by applying noise and scaling
def augment_data(X, y):
    augmented_X, augmented_y = [], []
    for i in range(len(X)):
        # Augment each sample by 10 times
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

# Function to build and train the ConvLSTM2D model 
# Model uses ConvLSTM2D layers for spatio-temporal feature extraction
def train_convlstm2d(X_train, y_train, actions):
    model = Sequential()

    # Add ConvLSTM2D layers with dropout for regularization and max pooling layer for downsampling
    model.add(ConvLSTM2D(filters=32, kernel_size=(2,2), activation='tanh', padding='same', 
                         return_sequences=True, input_shape=(30, 12, 14, 1)))
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

    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

    return model, history

if __name__ == "__main__":
    # Set the path to the dataset directory
    DATASET_PATH = '../Data Preprocessing/reduced_dataset/'
    print("Processing videos and extracting keypoints...")
    X, y, actions = process_videos(DATASET_PATH)
    
    print("Augmenting data...")
    X, y = augment_data(X, y)
    
    # Split the data into training and testing sets
    # 5% of the data is used for testing
    print("Splitting data and training models...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    print("Training ConvLSTM2D model...")
    convlstm2d_model, history = train_convlstm2d(X_train, y_train, actions)
    convlstm2d_model.save('trained_model_reduced_dataset.h5')
    
    print("\nEvaluating ConvLSTM2D model...")
    y_pred = np.argmax(convlstm2d_model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
      
    # Prediction Results
    results_df = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': y_pred,
        'True_Action': [actions[i] for i in y_test],
        'Predicted_Action': [actions[i] for i in y_pred],
        'Correct': y_test == y_pred
    })
    results_df.to_csv('prediction_details.csv', index=False)

    # Accuracy Report
    print(f"\nFinal Test Accuracy: {acc*100:.2f}%")

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_accuracy.png')
    plt.show()

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

    # Classification Report
    report_dict = classification_report(y_test, y_pred, target_names=actions, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv('classification_report.csv', index=True)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=actions))

    # Confusion Matrix
    # Extract true and predicted labels
    y_true = results_df['True_Action']
    y_pred = results_df['Predicted_Action']

    # Get all actions
    actions = sorted(results_df['True_Action'].unique())

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=actions)

    # Split the labels and matrix into two halves
    mid = len(actions) // 2
    actions1 = actions[:mid]
    actions2 = actions[mid:]

    cm1 = cm[:mid, :mid]
    cm2 = cm[mid:, mid:]

    # Plot for first 50 classes
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
                xticklabels=actions1, yticklabels=actions1,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix (Classes 1-60)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_1-60.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot for next 60 classes
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues',
                xticklabels=actions2, yticklabels=actions2,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix (Classes 61-120)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_61-120.png', dpi=300, bbox_inches='tight')
    plt.show()