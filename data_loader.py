import os
import hashlib
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch

class CustomImageMaskDataset(Dataset):
    def __init__(self, data_dir, image_size=(224, 224), transform=None):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        
        # Load image and mask file paths
        self.image_paths = []
        self.mask_paths = []
        for class_folder in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_folder)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if "_mask" not in img_file:
                        # Image file
                        img_path = os.path.join(class_path, img_file)
                        # Remove the extra .png extension by replacing .png with _mask.png
                        mask_path = os.path.join(class_path, img_file.replace(".png", "_mask.png"))
                        
                        # Debugging: print the paths of images and masks being added
                        print(f"Found image: {img_path}")
                        print(f"Expected mask: {mask_path}")
                        
                        if os.path.exists(mask_path):
                            self.image_paths.append(img_path)
                            self.mask_paths.append(mask_path)

        # Check if dataset has been populated
        print(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks.")
        # Remove duplicates
        self.image_paths, self.mask_paths = self.remove_duplicates(self.image_paths, self.mask_paths)
        print(f"After removing duplicates: {len(self.image_paths)} images and {len(self.mask_paths)} masks.")
        
        
    def __len__(self):
        return len(self.image_paths)

    # Resize both image and mask manually to ensure matching dimensions
    def __getitem__(self, idx):
        # Load image and corresponding mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # Resize both image and mask to the target size
        resize = transforms.Resize(self.image_size)
        image = resize(image)
        mask = resize(mask)

        # Apply other transformations (if any)
        if self.transform:
            image = self.transform(image)
            
        mask = transforms.ToTensor()(mask)  # Ensure the mask is also a tensor
        
        # Get the class label from the directory name (convert to integer)
        label = self.image_paths[idx].split(os.sep)[-2]
        
        # Map class labels (if needed)
        label_map = {'benign': 0, 'malignant': 1, 'normal': 2}  # Example label map
        label = label_map.get(label, -1)  # Default to -1 for unknown classes
        
        return image, mask, torch.tensor(label)  # Ensure label is a tensor

    def remove_duplicates(self, image_paths, mask_paths):
        seen_hashes = set()
        unique_image_paths = []
        unique_mask_paths = []

        for img_path, mask_path in zip(image_paths, mask_paths):
            with open(img_path, 'rb') as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
            if img_hash not in seen_hashes:
                seen_hashes.add(img_hash)
                unique_image_paths.append(img_path)
                unique_mask_paths.append(mask_path)

        return unique_image_paths, unique_mask_paths


# Step 1: Define dataset path
def get_data_loaders(data_dir, train_ratio=0.8, batch_size=32, image_size=(224, 224)):
    # Instantiate the custom dataset
    dataset = CustomImageMaskDataset(data_dir, image_size=image_size)

    # Split the dataset
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
