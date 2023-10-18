import os
import yaml
import re

# Step 1: Parse the log file to extract detection information
with open("predictres.txt", 'r') as f:
    logs = f.readlines()

# Regular expression pattern to extract details
pattern = r"/home/michael/animal/unlabeled/(?P<image_id>[a-z0-9]+).jpg: 640x640 (?P<detections>.+),"

# Extracting image IDs and their detections
detections_dict = {}
for log in logs:
    match = re.search(pattern, log)
    if match:
        image_id = match.group('image_id')
        detections = [det.split(" ")[1] for det in match.group('detections').split(",")]
        detections_dict[image_id] = detections

# Step 2: Load the YAML file to create mapping from class names to IDs
with open("data.yaml", 'r') as f:
    data = yaml.safe_load(f)

name_to_id = {name: idx for idx, name in enumerate(data['names'])}

# Step 3, 4: Find the image's species_id folder and compare with detections
correct_count = 0
incorrect_count = 0

# For individual animal accuracies
individual_counts = {name: {"correct": 0, "incorrect": 0} for name in data['names']}

for image_id, detected_classes in detections_dict.items():
    for root, dirs, files in os.walk("no_bbox"):
        if image_id + ".jpg" in files:
            species_id = int(os.path.basename(root))
            species_name = data['names'][species_id]
            if species_name in detected_classes:
                correct_count += 1
                individual_counts[species_name]["correct"] += 1
            else:
                incorrect_count += 1
                individual_counts[species_name]["incorrect"] += 1
            break

# Step 5: Calculate the accuracy and other metrics
accuracy = correct_count / (correct_count + incorrect_count)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print(f"Total Correct: {correct_count}")
print(f"Total Incorrect: {incorrect_count}")
print("\nIndividual Accuracies:")

for animal, counts in individual_counts.items():
    total = counts['correct'] + counts['incorrect']
    if total != 0:
        individual_accuracy = counts['correct'] / total
        print(f"{animal}: {individual_accuracy * 100:.2f}%")
    else:
        print(f"{animal}: No samples detected.")
