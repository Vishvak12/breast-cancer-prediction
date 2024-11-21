import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import CustomImageMaskDataset
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Step 1: Define the modified ResNet-18 model
class ResNetWithMasksDropout(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetWithMasksDropout, self).__init__()
        
        # Load pre-trained ResNet-18
        self.image_backbone = resnet18(weights = ResNet18_Weights.DEFAULT)
        self.image_backbone.fc = nn.Identity()  # Remove the final classification layer
        
        # Mask processing branch
        self.mask_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        
        # Final classification head
        image_feature_dim = 512  # ResNet-18 output features
        # Dynamically calculate mask feature dimension
        dummy_input = torch.randn(1, 1, 224, 224)  # Assuming masks are 224x224
        mask_feature_dim = self.mask_branch(dummy_input).view(-1).size(0)
        combined_dim = image_feature_dim + mask_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased Dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image, mask):
        # Extract features from the image
        image_features = self.image_backbone(image)
        
        # Process the mask
        mask_features = self.mask_branch(mask)
        
        # Concatenate features
        combined_features = torch.cat((image_features, mask_features), dim=1)
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        
        # Classification
        out = self.classifier(combined_features)
        return out
    
    

# Step 2: Define training and evaluation functions
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, masks, labels in train_loader:
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)
        print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}, min: {labels.min()}, max: {labels.max()}")
        optimizer.zero_grad()
        outputs = model(images, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return running_loss / len(train_loader), correct / total

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, masks, labels in test_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            outputs = model(images, masks)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return running_loss / len(test_loader), correct / total

