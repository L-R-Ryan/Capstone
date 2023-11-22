# Wildlife Image Dataset Preparation for Anomaly Detection

This script prepares a wildlife image dataset for anomaly detection by splitting it into training and validation sets.

## Overview

`simclr_get_trainval` organizes images into species-based subfolders, creating balanced datasets for training and validation. This step ensures the model learns from a variety of species, crucial for accurately detecting new or invasive species in wildlife images.

## Requirements

- Python 3.x
- os, shutil, random (standard libraries)

## Usage

1. **Set Directories**: 
   Define your source directory containing species subfolders, and the target directories for training and validation sets.

2. **Run Script**: 
   Execute the script to distribute images into training and validation sets based on the specified ratio.

3. **Output**: 
   Two directories are created, one each for training and validation, populated with the corresponding images.


# SIMCLR Anomaly Detection Training

The script "SimClr_Train" trains a model using Self-Supervised Learning (SIMCLR) and ResNet50 on wildlife images for anomaly detection.

## Overview

The SIMCLR approach adopts unsupervised learning techniques to train a robust model capable of identifying anomalies, such as unusual species patterns, in wildlife images. 

## Requirements

- Python 3.x
- PyTorch, torchvision
- PIL (Python Imaging Library)
- matplotlib, sklearn, NumPy
- CUDA (for GPU acceleration)

## Script Details

1. **Data Preparation** (`TripletDataset`):
   - Loads training and validation datasets.
   - Ensures each data point has an anchor, a positive sample (similar to anchor), and a negative sample (different from anchor).
   - Facilitates contrastive learning by teaching the model to differentiate between similar and dissimilar images.

2. **Model Initialization**:
   - Uses ResNet50 as an encoder within the SIMCLR framework. ResNet50 is a CNN that uses skip connections and commonly used in image classification problems. 
   ## Why ResNet50 in this Context?

1. **Feature Extraction**: ResNet50 is exceptionally good at extracting features from images, which is essential for identifying nuances in wildlife images.
   
2. **Handling Deep Networks**: Its architecture is designed to train deeper networks, which is generally challenging due to vanishing gradients. The residual connections in ResNet50 mitigate this issue.

3. **Pre-trained Models**: We can leverage pre-trained ResNet50 models, trained on large datasets like ImageNet, as a starting point. This transfer learning aspect is beneficial for improving model accuracy with limited wildlife data.

4. **Integration with SIMCLR**: ResNet50's feature extraction capabilities complement the contrastive learning approach of SIMCLR, enhancing the overall performance in distinguishing between similar and dissimilar images, crucial for anomaly detection.


3. **Contrastive Learning Process**:
   - Involves training the model to recognize and differentiate between images of the same and different species.
   - Enhances the model's capability to capture nuanced features in wildlife images.

4. **Validation and Evaluation**:
   - Assesses model performance on unseen data, ensuring real-world applicability.

5. **Model Saving**:
   - Stores trained model weights for future anomaly detection tasks.

## Running the Script

Set your training (`train_dir`) and validation (`val_dir`) dataset paths and adjust model parameters as needed.