import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from torch.nn import TripletMarginLoss
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
from simclr import SimCLR
from simclr.modules.transformations import TransformsSimCLR
import os
import random
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix #not used yet 
import numpy as np
from collections import defaultdict # not used yet 


torch.cuda.empty_cache()


class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = []
        self.transform = transform

        # Collect all image paths
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        anchor_image_path = self.image_paths[index]
        species_path = Path(anchor_image_path).parent

        # Get list of all image files in the same directory
        all_images = [img for img in os.listdir(species_path) if img.lower().endswith(('.jpg', '.jpeg', '.png')) and img != Path(anchor_image_path).name]

        # Choose a positive image (different from the anchor)
        positive_image = random.choice(all_images) if all_images else anchor_image_path

        # Dynamically determine available species for negative sample
        all_species_folders = [sp for sp in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, sp))]
        available_species = [sp for sp in all_species_folders if sp != species_path.name]

        if available_species:
            negative_species = random.choice(available_species)
            negative_species_path = os.path.join(self.root_dir, negative_species)
            negative_images = [img for img in os.listdir(negative_species_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            negative_image = random.choice(negative_images) if negative_images else Path(anchor_image_path).name
        else:
            # Handle case where no other species available
            negative_species_path = species_path
            negative_image = Path(anchor_image_path).name

        # Load and transform the images
        anchor = self.load_transform_image(anchor_image_path)
        positive = self.load_transform_image(os.path.join(species_path, positive_image))
        negative = self.load_transform_image(os.path.join(negative_species_path, negative_image))

        return anchor, positive, negative



    def load_transform_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Apply default transformations if none is provided
            image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(image)
        return image


class SimCLRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.transform = transform

        for species_folder in os.listdir(root_dir):
            species_path = os.path.join(root_dir, species_folder)
            if os.path.isdir(species_path):
                for image_name in os.listdir(species_path):
                    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(species_path, image_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image_i = self.transform(image)
            image_j = self.transform(image)  # Apply transform again to get a different version
        else:
            # Apply default transformations
            default_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_i = default_transform(image)
            image_j = default_transform(image)

        return image_i, image_j

# Define your transformations for SimCLR
transform = TransformsSimCLR(size=224) #probably too small but my computer was running out of memory even on a powerful computer

def train_and_validate(encoder, train_loader, val_loader, epochs, optimizer, criterion, device, run_folder):
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        # Training
        total_train_loss = 0
        encoder.train()
        for batch in train_loader:
            anchor, positive, negative = batch
            anchor = torch.cat(anchor, dim=0).to(device)
            positive = torch.cat(positive, dim=0).to(device)
            negative = torch.cat(negative, dim=0).to(device)

            # Extract features and compute loss
            anchor_out = encoder(anchor)
            positive_out = encoder(positive)
            negative_out = encoder(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)

        # Validation
        total_val_loss = 0
        encoder.eval()
        with torch.no_grad():
            for batch in val_loader:
                anchor, positive, negative = batch
                anchor = torch.cat(anchor, dim=0).to(device)
                positive = torch.cat(positive, dim=0).to(device)
                negative = torch.cat(negative, dim=0).to(device)

                # Extract features and compute loss
                anchor_out = encoder(anchor)
                positive_out = encoder(positive)
                negative_out = encoder(negative)

                loss = criterion(anchor_out, positive_out, negative_out)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss.append(avg_val_loss)

        # Save the model state
        encoder_save_path = os.path.join(run_folder, f"encoder_epoch_{epoch}.pth")
        torch.save(encoder.state_dict(), encoder_save_path)

        print(f"Epoch {epoch + 1}/{epochs} completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(run_folder, 'loss_plot.png'))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = '/home/michael/animal/sm_train/'
    val_dir = '/home/michael/animal/sm_val/'

    # Initialize the datasets
    train_dataset = TripletDataset(root_dir=train_dir, transform=transform)
    val_dataset = TripletDataset(root_dir=val_dir, transform=transform)
    
    # Calculate weights for each image in the dataset
    class_counts = {species_folder: 0 for species_folder in os.listdir(train_dir)}
    for path in train_dataset.image_paths:
        species_folder = Path(path).parent.name
        class_counts[species_folder] += 1

    weights = [1.0 / class_counts[Path(path).parent.name] for path in train_dataset.image_paths]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Training DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4)

    
    # Initialize the model
    encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    n_features = encoder.fc.in_features
    encoder.fc = nn.Identity()
    encoder = encoder.to(device)

    # Initialize SimCLR with the encoder
    projection_dim = 64
    model = SimCLR(encoder, projection_dim, n_features).to(device)

    # Define optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = TripletMarginLoss(margin=1.0)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_folder = f"simclr_runs/run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)

    # Train and validate the model
    train_and_validate(encoder, train_loader, val_loader, epochs=100, optimizer=optimizer, criterion=criterion, device=device, run_folder=run_folder)


if __name__ == '__main__':
    main()

