import os
import shutil
from pathlib import Path

# Set your source and destination folders
SOURCE_FOLDER = "dataset"  
DESTINATION_FOLDER = "reduced_dataset" 
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')

def has_more_than_five_videos(folder_path):
    return len([
        file for file in os.listdir(folder_path)
        if file.lower().endswith(VIDEO_EXTENSIONS)
    ]) > 7

def copy_folder_with_videos(src_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for file_name in os.listdir(src_folder):
        if file_name.lower().endswith(VIDEO_EXTENSIONS):
            src_file = os.path.join(src_folder, file_name)
            dst_file = os.path.join(dest_folder, file_name)
            shutil.copy2(src_file, dst_file)

def main():
    for subfolder_name in os.listdir(SOURCE_FOLDER):
        subfolder_path = os.path.join(SOURCE_FOLDER, subfolder_name)
        if os.path.isdir(subfolder_path) and has_more_than_five_videos(subfolder_path):
            dest_subfolder_path = os.path.join(DESTINATION_FOLDER, subfolder_name)
            copy_folder_with_videos(subfolder_path, dest_subfolder_path)
            print(f"Copied: {subfolder_name}")

    print("Done!")

if __name__ == "__main__":
    main()