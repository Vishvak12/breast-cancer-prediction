import os
import shutil
from PIL import Image
import numpy as np
from hashlib import md5
from sklearn.model_selection import train_test_split

# Define dataset paths
dataset_path = "C:/Users/vishv/Documents/research paper/breast cancer/dataset"
output_path = "Cleaned_Dataset"
train_folder = os.path.join(output_path, "train")
test_folder = os.path.join(output_path, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Helper function to calculate image hash
def calculate_hash(image_path):
    with open(image_path, "rb") as f:
        return md5(f.read()).hexdigest()

# Step 1: Remove duplicates
def remove_duplicates(data_dir):
    unique_images = {}
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(root, file)
                image_hash = calculate_hash(file_path)

                if image_hash not in unique_images:
                    unique_images[image_hash] = file_path
                else:
                    print(f"Duplicate found: {file_path}, skipping.")
    
    return list(unique_images.values())

# Step 2: Process masks with OR operation
def merge_masks(mask_paths):
    merged_mask = None
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path).convert("1"))  # Binary mask
        if merged_mask is None:
            merged_mask = mask
        else:
            merged_mask = np.logical_or(merged_mask, mask)
    return Image.fromarray(merged_mask.astype(np.uint8) * 255)

# Step 3: Organize dataset
def organize_dataset(image_paths):
    train_images, test_images = train_test_split(image_paths, test_size=0.2, random_state=42)

    for i, images in enumerate([train_images, test_images]):
        target_folder = train_folder if i == 0 else test_folder
        for image_path in images:
            parent_dir = os.path.basename(os.path.dirname(image_path))
            new_image_path = os.path.join(target_folder, parent_dir)
            os.makedirs(new_image_path, exist_ok=True)
            
            # Copy image
            shutil.copy(image_path, os.path.join(new_image_path, os.path.basename(image_path)))

            # Find and process associated masks
            mask_dir = os.path.join(os.path.dirname(image_path), "mask")
            if os.path.exists(mask_dir):
                masks = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")]
                if len(masks) > 1:
                    merged_mask = merge_masks(masks)
                    merged_mask.save(os.path.join(new_image_path, f"merged_mask_{os.path.basename(image_path)}"))
                elif masks:
                    shutil.copy(masks[0], os.path.join(new_image_path, os.path.basename(masks[0])))

# Execute Steps
if __name__ == "__main__":
    unique_images = remove_duplicates(dataset_path)
    organize_dataset(unique_images)

    print(f"Dataset cleaned and organized in {output_path}")
