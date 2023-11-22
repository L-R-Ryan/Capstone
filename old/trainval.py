import os
import shutil
from sklearn.model_selection import train_test_split

# Define constants
SPLIT_RATIO = 0.8
PROCESSED_DIR = 'processed'
TRAIN_DIR = 'train'
VAL_DIR = 'val'
CATEGORIES = [folder for folder in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, folder))]

import os
from PIL import Image
import shutil

def resize_image(img, target_size):
    """
    Resize the image directly to the target size without maintaining aspect ratio.
    """
    return img.resize((target_size, target_size), Image.LANCZOS)

source_dir = "no_bbox"
destination_dir = "images_for_detection"
os.makedirs(destination_dir, exist_ok=True)

# Loop through each category_id sub-directory inside the no_bbox directory
for category_id in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category_id)
    
    # Ensure we're working with directories (category_id folders)
    if os.path.isdir(category_path):
        # Copy each image from the category_id sub-directory to the images_for_detection directory
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            
            with Image.open(img_path) as img:
                img = resize_image(img, 640)  # Resize to 640x640
                img.save(os.path.join(destination_dir, img_name))

print("Finished processing images.")

