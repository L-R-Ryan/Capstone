import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from torch.nn import TripletMarginLoss
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from simclr import SimCLR
from simclr.modules.transformations import TransformsSimCLR
import os
import random
import datetime

torch.cuda.empty_cache()

root_dir = '/home/michael/animal/sm_train/'

class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.species_folders = []
        self.transform = transform

        # Collect all species folders and image paths
        for species_folder in os.listdir(root_dir):
            species_path = os.path.join(root_dir, species_folder)
            if os.path.isdir(species_path):
                self.species_folders.append(species_folder)
                for image_name in os.listdir(species_path):
                    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Only add image files to the image_paths
                        self.image_paths.append((species_path, image_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        species_path, anchor_image = self.image_paths[index]

        # Get list of all image files in the same directory
        all_images = [img for img in os.listdir(species_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Choose a positive image (different from the anchor)
        positive_image = random.choice([img for img in all_images if img != anchor_image])

        # Choose a negative species and image
        negative_species = random.choice([sp for sp in self.species_folders if sp != os.path.basename(species_path)])
        negative_species_path = os.path.join(root_dir, negative_species)
        negative_images = [img for img in os.listdir(negative_species_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        negative_image = random.choice(negative_images)

        # Load and transform the images
        anchor = self.load_transform_image(os.path.join(species_path, anchor_image))
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

def train(encoder, data_loader, epochs, optimizer, criterion, device, run_folder):
    encoder.train()  # Use 'encoder.train()' if 'encoder' refers to the model
    for epoch in range(epochs):
        for batch in data_loader:
            # Extract and concatenate the batch items
            anchor = torch.cat(batch[0], dim=0).to(device)
            positive = torch.cat(batch[1], dim=0).to(device)
            negative = torch.cat(batch[2], dim=0).to(device)

            # Extract features from the images using the encoder
            anchor_out = encoder(anchor)
            positive_out = encoder(positive)
            negative_out = encoder(negative)

            # Compute the triplet loss
            loss = criterion(anchor_out, positive_out, negative_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save the model state
            encoder_save_path = os.path.join(run_folder, f"encoder_epoch_{epoch}.pth")
            torch.save(encoder.state_dict(), encoder_save_path)

    print(f"Epoch {epoch + 1}/{epochs} completed.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the dataset
    dataset = TripletDataset(root_dir=root_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

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

    # Pass this run folder to your training function
    # Pass the encoder and run folder to your training function
    train(encoder, data_loader, epochs=50, optimizer=optimizer, criterion=criterion, device=device, run_folder=run_folder)


if __name__ == '__main__':
    main()

