# Remove corrupted videos from the dataset by using OpenCV to check if the video file is corrupted.

# Importing the required libraries
import os
import cv2

# Function to check if the video file is corrupted using OpenCV
def is_video_corrupted(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return True  
    
    # Read 10 frames from the video only to check if it is corrupted
    for _ in range(10): 
        ret, _ = cap.read()
        if not ret:
            cap.release()
            return True 

    cap.release()
    return False 

# Function to delete all corrupted videos from the dataset
def delete_corrupted_videos(directory):
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv')
    # Use os.walk to traverse directories
    for root, dirs, files in os.walk(directory):  
        for filename in files:
            if filename.lower().endswith(video_extensions):
                file_path = os.path.join(root, filename)
                if is_video_corrupted(file_path):
                    print(f"Deleting corrupted video: {file_path}")
                    os.remove(file_path)

# Directory path
directory_path = 'dataset/' 
delete_corrupted_videos(directory_path)