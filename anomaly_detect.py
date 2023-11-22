import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from ultralytics import YOLO
import joblib
from pathlib import Path
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import datetime


# Load SimCLR model
def load_simclr_model(model_path, device):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Load SVM models
def load_svm_models(svm_dir):
    svm_models = {}
    for svm_file in os.listdir(svm_dir):
        if svm_file.endswith('.joblib'):
            species = svm_file.split('_')[0]
            svm_models[species] = joblib.load(os.path.join(svm_dir, svm_file))
    return svm_models

# Extract features using SimCLR
def extract_features(simclr_model, img, device):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = simclr_model(img_t)
        print(features)
    return features.squeeze(0).cpu().numpy()

# Run YOLO model and detect anomalies
def detect_anomalies(yolo_model, image_path, simclr_model, svm_models, device, anomaly_dir):
    str_image_path = str(image_path)
    img = Image.open(str_image_path)

    # Run YOLO prediction and save output with bounding boxes
    yolo_results = yolo_model.predict(source=str_image_path, verbose=False, save=True, imgsz=640, conf=0.25)

    features_list = []
    labels_list = []
    anomaly_count = 0

    # Check if results have boxes and process each detection
    if yolo_results and hasattr(yolo_results[0], 'boxes') and yolo_results[0].boxes is not None:
        boxes = yolo_results[0].boxes.xywh.cpu()  # Extract bounding boxes in XYWH format
        for box in boxes:
            x, y, w, h = box[:4].numpy()  # Convert tensor to numpy array
            x, y, w, h = int(x), int(y), int(w), int(h)  # Convert to integers
            cropped_img = img.crop((x, y, x+w, y+h))
            features = extract_features(simclr_model, cropped_img, device)
            species = get_species_name(int(box[5]))  # Assuming the 6th element is the class index

            is_anomaly = species in svm_models and svm_models[species].predict([features])[0] == -1
            if is_anomaly:
                anomaly_count += 1

            features_list.append(features)
            labels_list.append(1 if is_anomaly else 0)

    print(f"Total features extracted: {len(features_list)}, Anomalies detected: {anomaly_count}")
    return features_list, labels_list


def get_species_name(cls_index): #ignore anomaly I thought I could make an anomaly class
    species_mapping = {
        0: 'anomaly',
        1: 'animal',
        2: 'bird',
        3: 'bobcat',
        4: 'boycot',
        5: 'c',
        6: 'coyote',
        7: 'dd',
        8: 'deer',
        9: 'dog',
        10: 'none',
        11: 'person',
        12: 'pig',
        13: 'pwe',
        14: 'raccoon',
        15: 'rodent',
        16: 'skunk',
    }
    return species_mapping.get(cls_index, 'unknown')



def detect_anomalies_and_visualize(resnet_model, svm_models, yolo_model, image_path, device, anomaly_dir, feature_file, tsne_features, tsne_labels):
    image_id, features, species = extract_resnet_features(resnet_model, image_path, device)
    is_anomaly = False
    if features is not None:
        svm_model = svm_models.get(species)
        if svm_model and svm_model.predict([features])[0] == -1:
            is_anomaly = True
            shutil.copy(image_path, os.path.join(anomaly_dir, os.path.basename(image_path)))
        with open(feature_file, 'a') as f:
            f.write(f"{image_id}: {features.tolist()}\n")

        tsne_features.append(features)
        tsne_labels.append(1 if is_anomaly else 0)

def plot_tsne_clusters(features, labels, output_dir):
    # Filter out None values from features and corresponding labels
    filtered_features = [feat for feat, label in zip(features, labels) if feat is not None]
    filtered_labels = [label for feat, label in zip(features, labels) if feat is not None]

    # Check if filtered_features has elements to process
    if filtered_features:
        # Convert list of arrays to a 2D NumPy array
        features_array = np.vstack(filtered_features)

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=0)
        transformed_features = tsne.fit_transform(features_array)

        # Plot
        plt.scatter(transformed_features[:, 0], transformed_features[:, 1], c=filtered_labels)
        plt.colorbar()
        plt.title("t-SNE Visualization of Features")
        # Save the plot in the specified output directory
        plt.savefig(os.path.join(output_dir, "tsne_plot.png"))
        plt.close()  # Close the plot to free up memory
    else:
        print("No valid features to visualize.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a unique run ID
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    resnet_model_path = '/home/michael/animal/simclr_runs/300_epoch/encoder_epoch_250.pth'
    svm_models_dir = '/home/michael/animal/svm_models/'
    yolo_weights_path = '/home/michael/animal/runs/detect/train8/weights/best.pt'
    image_folder = Path('/home/michael/animal/unlabeled_anom_test/')

    simclr_model = load_simclr_model(resnet_model_path, device)
    svm_models = load_svm_models(svm_models_dir)
    yolo_model = YOLO(yolo_weights_path)
    output_dir = f"/home/michael/animal/anom_runs/run-{run_id}/tse_output"
    os.makedirs(output_dir, exist_ok=True)

    
    anomaly_dir = f"/home/michael/animal/anom_runs/run-{run_id}/images"
    os.makedirs(anomaly_dir, exist_ok=True)

    tsne_features = []
    tsne_labels = []

    for image_file in image_folder.glob('*.jpg'):
        features, is_anomaly = detect_anomalies(yolo_model, image_file, simclr_model, svm_models, device, anomaly_dir)
        tsne_features.append(features)
        tsne_labels.append(1 if is_anomaly else 0)

    # Generate and save t-SNE plot
    plot_tsne_clusters(tsne_features, tsne_labels, output_dir)


if __name__ == '__main__':
    main()
