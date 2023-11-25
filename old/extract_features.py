import torch
import torch.nn as nn  
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from pathlib import Path
import json
from ultralytics import YOLO
import joblib

def load_simclr_model(model_path, device):
    # Initialize the ResNet model
    encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    encoder.fc = nn.Identity()  # Replace the fully connected layer
    encoder = encoder.to(device)

    # Load the trained weights
    encoder.load_state_dict(torch.load(model_path, map_location=device))
    encoder.eval()
    return encoder

def load_svm_models(svm_dir):
    svm_models = {}
    for svm_file in os.listdir(svm_dir):
        if svm_file.endswith('.joblib'):
            species = svm_file.split('_')[0]
            svm_models[species] = joblib.load(os.path.join(svm_dir, svm_file))
    return svm_models

def extract_resnet_features(resnet_model, image_path, bbox_data, device):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_id = Path(image_path).stem

    if image_id.startswith("anomaly"):
        return image_id, None, None

    bbox_info = bbox_data.get(image_id, None)

    with Image.open(image_path) as img:
        if bbox_info:
            x, y, width, height = bbox_info ########## IS THIS CODE REDUNDANT WITH THE BBOX CODE UP THERE????
            x_min, y_min = x, y
            x_max, y_max = x + width, y + height
            img = img.crop((x_min, y_min, x_max, y_max))
        img = transform(img).unsqueeze(0).to(device)  # Move the image tensor to the same device as the model

        with torch.no_grad():
            features = resnet_model(img)

    species = identify_species(image_path)  # Implement this function based on your data
    return image_id, features, bbox_info, species

def identify_species(image_path):
    # Implement your species identification logic here...

def load_yolo_model():
    # Load the YOLO model
   

    weights_path = '/home/michael/animal/runs/detect/train8/weights/best.pt'
    model = YOLO('/home/michael/animal/data.yaml')
    model.load(weights_path)
    
    model.eval()
    return model


def get_anomaly_bbox(anomaly_index, yolo_results):
    # Extract the bounding box from YOLO results for the given anomaly index
    # This function needs to be defined based on how you want to relate anomaly indices to bounding boxes
    pass


def is_anomaly_in_yolo_detections(detected_objects, anomaly_bbox, confidence_threshold=0.5):
    """
    Check if the anomaly index corresponds to any YOLO detection.
    
    :param detected_objects: Objects detected by YOLO.
    :param anomaly_bbox: Bounding box of the anomaly detected.
    :param confidence_threshold: Threshold for considering YOLO detection.
    :return: Boolean indicating if the anomaly overlaps with YOLO detection.
    """
    for obj in detected_objects:
        # Extract YOLO detection information
        x1, y1, x2, y2, conf, cls = obj[:6]
        if conf < confidence_threshold:
            continue  # Skip low confidence detections

        # Check for bounding box overlap
        if do_boxes_overlap(anomaly_bbox, (x1, y1, x2, y2)):
            return True
    return False


def bbox_overlap(bbox1, bbox2):
    """
    Check if two bounding boxes overlap.

    :param bbox1: First bounding box [x_min, y_min, x_max, y_max].
    :param bbox2: Second bounding box [x_min, y_min, x_max, y_max].
    :return: True if boxes overlap, False otherwise.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # If they do not overlap, return False
    if x_right < x_left or y_bottom < y_top:
        return False

    # Calculate the area of the intersection rectangleload_
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate the combined area
    total_area = bbox1_area + bbox2_area - intersection_area

    # Calculate the overlap ratio (i.e., intersection over union)
    overlap_ratio = intersection_area / total_area

    # Consider it an overlap if the ratio is above a threshold (e.g., 0.5)
    return overlap_ratio > 0.5


def detect_anomalies(resnet_model, svm_models, yolo_model, image_path, device):
    image_id, features, species = extract_resnet_features(resnet_model, image_path, device)
    if features is not None:
        svm_model = svm_models.get(species)
        if svm_model and svm_model.predict([features])[0] == -1:
            yolo_results = yolo_model.predict(image_path)
            # Further processing with YOLO results and anomaly detection...
            # ...

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models and SVMs
    resnet_model_path = '/home/michael/animal/simclr_runs/run_20231119-185055/encoder_epoch_97.pth'
    resnet_model = load_simclr_model(resnet_model_path, device)
    svm_models_dir = '/home/michael/animal/svm_models/'
    svm_models = load_svm_models(svm_models_dir)
    yolo_model = load_yolo_model()

    # Process images
    image_folder = Path('/home/michael/animal/unlabeled_anom_test/')
    for image_file in image_folder.glob('*.jpg'):
        detect_anomalies(resnet_model, svm_models, yolo_model, image_file, device)

if __name__ == '__main__':
    main()

