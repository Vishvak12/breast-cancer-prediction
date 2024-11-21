import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from dropout import ResNetWithMasksDropout
from ensemble import EnsembleModel, ResNetWithMasks, ResNetWithMasksL2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = ResNetWithMasks(num_classes=3).to(device)
model1.load_state_dict(torch.load("resnet_with_masks.pth",weights_only=True))
model1.eval()

model2 = ResNetWithMasksDropout(num_classes=3).to(device)
model2.load_state_dict(torch.load("resnetdropout.pth",weights_only=True))
model2.eval()
model3 = ResNetWithMasksL2(num_classes=3).to(device)
model3.load_state_dict(torch.load("resnet_with_masks_l2.pth",weights_only=True))
model3.eval()
ensemble_weights = [0.5, 0.3, 0.2]
ensemble_model = EnsembleModel(model1, model2, model3, ensemble_weights).to(device)
ensemble_model.eval().to(device)
# Define the image transform (same as used for training)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define a function to predict a single image
def predict_single_image(image_path, mask_path=None):
    # Load the image
    image = Image.open(image_path).convert('RGB')  # Make sure the image is in RGB
    image = image_transform(image).unsqueeze(0)  # Add batch dimension

    # If you have a mask, load and preprocess it similarly
    if mask_path:
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale
        mask = image_transform(mask).unsqueeze(0)  # Add batch dimension
    else:
        mask = torch.zeros_like(image)  # If no mask is provided, use an empty mask

    # Move image and mask to the device
    image = image.to(device)
    mask = mask.to(device)

    # Set the model to evaluation mode and perform inference
    ensemble_model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        output = ensemble_model(image, mask)  # Forward pass through the ensemble model
        _, predicted = torch.max(output, 1)  # Get the predicted class

    return predicted.item()

# Test with a single image
image_path = "split_dataset/test/benign/benign (7).png"  # Example image path
mask_path = "split_dataset/test/benign/benign (7)_mask.png"  # Replace with the path to the mask (or None if not available)

predicted_class = predict_single_image(image_path, mask_path)

print(f"Predicted Class: {predicted_class}")


