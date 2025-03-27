# Remove all non-MP4 videos from the dataset

# Importing the required libraries
import os

# Function to remove all non-MP4 videos from the dataset
def remove_non_mp4_videos(data_path):
    for action in os.listdir(data_path):
        action_path = os.path.join(data_path, action)
        if os.path.isdir(action_path):
            for video in os.listdir(action_path):
                video_path = os.path.join(action_path, video)
                # Check if the file is not an mp4 file
                if not video.lower().endswith('.mp4'):
                    print(f'Removing non-MP4 video: {video_path}')
                    os.remove(video_path)  # Remove the file

# Dataset path
DATASET_PATH = 'dataset/'

# Call function
remove_non_mp4_videos(DATASET_PATH)