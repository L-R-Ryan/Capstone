import torch
import torch.nn as nn  
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from sklearn.svm import OneClassSVM
from pathlib import Path
import json
from ultralytics import YOLO

def load_simclr_model(model_path, device):
    # Initialize the ResNet model
    encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    encoder.fc = nn.Identity()  # Replace the fully connected layer
    encoder = encoder.to(device)

    # Load the trained weights
    encoder.load_state_dict(torch.load(model_path, map_location=device))
    encoder.eval()
    return encoder

def read_bounding_box_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    bbox_data = {}
    for item in data:
        if 'image_id' in item and 'bbox' in item and isinstance(item['bbox'], list):
            if all(v is not None for v in item['bbox']):  # Check if all bbox values are not None
                # Adjust image_id if needed
                image_id = item['image_id']
                bbox_data[image_id] = item['bbox']

    return bbox_data

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
            x, y, width, height = bbox_info
            x_min, y_min = x, y
            x_max, y_max = x + width, y + height
            img = img.crop((x_min, y_min, x_max, y_max))
        img = transform(img).unsqueeze(0).to(device)  # Move the image tensor to the same device as the model

        with torch.no_grad():
            features = resnet_model(img)

    return image_id, features, bbox_info


def apply_novelty_detection(features):
    # Initialize a One-Class SVM
    oc_svm = OneClassSVM(gamma='auto')
    oc_svm.fit(features)
    anomaly_scores = oc_svm.decision_function(features)
    # Set a threshold for anomaly detection (this needs to be tuned)
    threshold = -0.1 
    anomalies = [i for i, score in enumerate(anomaly_scores) if score < threshold]
    return anomalies

def load_yolo_model():
    # Load the YOLO model
   

    weights_path = '/home/michael/animal/runs/detect/train8/weights/best.pt'
    model = YOLO('/home/michael/animal/data.yaml')
    model.load(weights_path)
    
    model.eval()
    return model

def validate_with_yolo(yolo_model, image_path, anomaly_indices, features):
    # Convert features tensor to a list of lists for One-Class SVM
    features_list = features.cpu().detach().numpy().tolist()
    # Initialize a One-Class SVM and fit it on the features
    oc_svm = OneClassSVM(gamma='auto').fit(features_list)
    # Get the decision function scores
    decision_scores = oc_svm.decision_function(features_list)
    # Predict anomalies based on decision scores
    anomalies = [i for i, score in enumerate(decision_scores) if score < 0]  # Anomaly threshold

    # Run YOLO prediction on the image
    yolo_results = yolo_model.predict(image_path)

    # Compare YOLO predictions with anomalies detected by One-Class SVM
    validated_anomalies = []
    for anomaly_index in anomalies:
        # Retrieve the bounding box of the anomaly
        anomaly_bbox = get_anomaly_bbox(anomaly_index, yolo_results)
        # Check if the anomaly overlaps with YOLO detections
        if is_anomaly_in_yolo_detections(yolo_results['detections'], anomaly_bbox):
            validated_anomalies.append(anomaly_index)

    return validated_anomalies

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

def do_boxes_overlap(box1, box2):
    """
    Check if two bounding boxes overlap.
    
    :param box1: First bounding box (x1, y1, x2, y2).
    :param box2: Second bounding box (x1, y1, x2, y2).
    :return: Boolean indicating if boxes overlap.
    """
    # Check if boxes do not overlap
    if (box1[2] < box2[0] or box1[0] > box2[2] or
        box1[3] < box2[1] or box1[1] > box2[3]):
        return False
    return True

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


def detect_anomalies(resnet_model, yolo_model, image_path):
    features = extract_resnet_features(resnet_model, image_path)
    anomalies = apply_novelty_detection(features)
    validated_anomalies = validate_with_yolo(yolo_model, image_path, anomalies)
    return validated_anomalies

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    resnet_model_path = '/home/michael/animal/simclr_runs/run_20231119-185055/encoder_epoch_97.pth'
    resnet_model = load_simclr_model(resnet_model_path, device)
    yolo_model = load_yolo_model()
    

    # Paths
    image_folder = Path('/home/michael/animal/unlabeled_anom_test/')
    json_path = Path('/home/michael/animal/jldp-animl-cct.json')
    output_path = Path('/home/michael/animal/features.txt')

    # Read bounding box data
    bbox_data = read_bounding_box_data(json_path)
    bbox_data = bbox_data if bbox_data is not None else {}

    # Process each image
    for image_file in image_folder.glob('*.jpg'):
        image_id = image_file.stem

        # Skip JSON lookup for images starting with "anomaly"
        bbox_info = None if image_id.startswith("anomaly") else bbox_data.get(image_id, None)

        # Extract features
        image_id, features, bbox_info = extract_resnet_features(resnet_model, image_file, bbox_info, device)

        if features is not None:
            # Save extracted features, image ID, and bbox_info
            with open(output_path, 'a') as f:
                f.write(f"{image_id},{features.cpu().numpy().tolist()},{bbox_info}\n")

            # Detect anomalies
            anomalies = detect_anomalies(resnet_model, yolo_model, image_file)
            print(f"Detected anomalies for {image_id}: {anomalies}")
        else:
            print(f"Warning: No features extracted for image {image_id}")

if __name__ == '__main__':
    main()

