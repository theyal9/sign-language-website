# Extract class names from dataset directory structure and save them to a text file

# Import necessary libraries
import os

# Configuration
# DATASET_PATH = "dataset/"
# OUTPUT_FILE = "class_names.txt"
DATASET_PATH = "reduced_dataset/"
OUTPUT_FILE = "reduced_dataset_class_names.txt"

# Extract class names from dataset directory structure
def extract_class_names(dataset_path):
    classes = []
    
    for entry in os.listdir(dataset_path):
        full_path = os.path.join(dataset_path, entry)
        if os.path.isdir(full_path):
            classes.append(entry)
    
    # Sort alphabetically and clean names
    classes = sorted(classes)
    # Remove underscores
    return [cls.replace("_", " ").strip() for cls in classes]  

# Save class names to a text file
def save_class_names(class_list, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for cls in class_list:
            f.write(f"{cls}\n")

# Main script
if __name__ == "__main__":
    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")
    
    # Extract and save classes
    classes = extract_class_names(DATASET_PATH)
    save_class_names(classes, OUTPUT_FILE)
    
    # Print summary
    print(f"Successfully extracted {len(classes)} classes to {OUTPUT_FILE}")
    print("First 5 classes:", classes[:5])