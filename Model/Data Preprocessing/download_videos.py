# Download the videos from the URLs provided in the dataset.json file. 
# The videos are stored in a dataset directory. 

# Importing the required libraries
import os
import requests
import json
import re
import time

# Load dataset from JSON file
with open("dataset.json", "r") as f:
    DATASET = json.load(f)

# Create dataset directory 
os.makedirs("dataset", exist_ok=True)

# Function to sanitize filenames to remove special characters
def sanitize_filename(url):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', url.split("/")[-1])

# Download videos with error handling and retries
def download_videos():
    for item in DATASET:
        gloss = item["gloss"]
        os.makedirs(f"dataset/{gloss}", exist_ok=True)
        for instance in item["instances"]:
            url = instance["url"]
            filename = os.path.join("dataset", gloss, sanitize_filename(url))

            # Skip .swf files
            if url.endswith(".swf"):
                print(f"Skipping Flash file: {url}")
                continue

            if not os.path.exists(filename):
                print(f"Downloading {url}...")
                retries = 3 
                for attempt in range(retries):
                    try:
                        r = requests.get(url, stream=True, timeout=10)
                        with open(filename, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=1024):
                                if chunk:
                                    f.write(chunk)
                        print(f"Downloaded: {filename}")
                        break
                    except requests.exceptions.RequestException as e:
                        print(f"Error downloading {url}: {e}")
                        if attempt < retries - 1:
                            print("Retrying...")
                            time.sleep(2) 
                        else:
                            print(f"Failed to download {url} after {retries} attempts.")

# Main script
if __name__ == "__main__":
    download_videos()
        