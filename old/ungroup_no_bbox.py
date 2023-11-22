import os
import shutil

# Define source and destination directories
source_dir = 'no_bbox'
destination_dir = 'images_for_detection'
os.makedirs(destination_dir, exist_ok=True)

# Loop through each category_id folder in no_bbox and copy images to the destination
for category_folder in os.listdir(source_dir):
    category_folder_path = os.path.join(source_dir, category_folder)
    
    # Ensure it's a directory before proceeding
    if os.path.isdir(category_folder_path):
        for image_file in os.listdir(category_folder_path):
            source_path = os.path.join(category_folder_path, image_file)
            destination_path = os.path.join(destination_dir, image_file)
            
            # Copy the image to the destination directory
            shutil.copy(source_path, destination_path)

print("Images copied successfully!")
