import torch
from torch import nn
from torch.utils.data import DataLoader
from data_loader import CustomImageMaskDataset, get_data_loaders  # Assuming the CustomImageMaskDataset and loader are implemented here
from torchvision.models import resnet18, ResNet18_Weights
from dropout import ResNetWithMasksDropout,evaluate_model
# Define ResNet-based models
class ResNetWithMasks(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetWithMasks, self).__init__()
        self.image_backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_backbone.fc = nn.Identity()
        self.mask_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        image_feature_dim = 512
        dummy_input = torch.randn(1, 1, 224, 224)
        mask_feature_dim = self.mask_branch(dummy_input).view(-1).size(0)
        combined_dim = image_feature_dim + mask_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, mask):
        image_features = self.image_backbone(image)
        mask_features = self.mask_branch(mask)
        combined_features = torch.cat((image_features, mask_features), dim=1)
        return self.classifier(combined_features)





class ResNetWithMasksL2(ResNetWithMasks):
    def __init__(self, num_classes=3, l2_lambda=0.001):
        super().__init__(num_classes)
        self.l2_lambda = l2_lambda

    def forward(self, image, mask):
        image_features = self.image_backbone(image)
        mask_features = self.mask_branch(mask)
        combined_features = torch.cat((image_features, mask_features), dim=1)
        output = self.classifier(combined_features)
        return output


# Ensemble model definition
class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, model3, weights):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.weights = weights  # List of weights for the models

    def forward(self, image, mask):
        output1 = self.model1(image, mask)
        output2 = self.model2(image, mask)
        output3 = self.model3(image, mask)
        weighted_output = (
            self.weights[0] * output1 +
            self.weights[1] * output2 +
            self.weights[2] * output3
        )
        return weighted_output


# Load dataset and data loaders
data_dir = "split_dataset"
batch_size = 32
image_size = (224, 224)
train_loader, test_loader = get_data_loaders(
    data_dir=data_dir,
    train_ratio=0.8,
    batch_size=batch_size,
    image_size=image_size
)
train_path = "split_dataset/train"
test_path = "split_dataset/test"
train_dataset = CustomImageMaskDataset(train_path)
test_dataset = CustomImageMaskDataset(test_path)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
# Load pre-trained models
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

# Define the ensemble model
ensemble_weights = [0.5, 0.3, 0.2]
ensemble_model = EnsembleModel(model1, model2, model3, ensemble_weights).to(device)
ensemble_model.eval().to(device)

# Evaluate the ensemble model
def evaluate_ensemble(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for images, masks, labels in data_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            outputs = model(images, masks)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f"Ensemble Model Accuracy: {accuracy:.4f}")
    return accuracy, predictions


# Run evaluation
accuracy, preds = evaluate_ensemble(ensemble_model, test_loader, device)

# Save predictions
import pandas as pd
output_df = pd.DataFrame({'Predicted': preds})
output_df.to_csv("ensemble_predictions.csv", index=False)
print("Predictions saved to ensemble_predictions.csv")
