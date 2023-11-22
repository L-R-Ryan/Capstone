import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

class SimCLRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.transform = transform

        # Walk through each species subfolder
        for species_folder in os.listdir(root_dir):
            species_path = os.path.join(root_dir, species_folder)
            if os.path.isdir(species_path):  # Check if it is a directory
                for image_name in os.listdir(species_path):
                    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Add more extensions if needed
                        self.image_paths.append(os.path.join(species_path, image_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        return image

# Define your transformations for SimCLR
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=int(256*0.1)),
    transforms.ToTensor(),
])

def imshow(img):
    """Utility function to unnormalize and display an image"""
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

def show_batch_of_images(dataloader):
    """Function to show a batch of images"""
    images = next(iter(dataloader))
    plt.figure(figsize=(20, 10))
    for i in range(min(len(images), 4)):  # Display up to 4 images
        plt.subplot(1, 4, i + 1)
        imshow(images[i])
    plt.show()

root_dir = '/home/michael/animal/anom_copy/'  # Replace with your root directory path
dataset = SimCLRDataset(root_dir=root_dir, transform=transform)
print(f"Dataset size: {len(dataset)}")

def main():
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    show_batch_of_images(dataloader)

if __name__ == '__main__':
    main()
