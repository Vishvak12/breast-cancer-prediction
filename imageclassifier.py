from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import kagglehub
from dataset import get_data_loaders  # Import the data loading function from the data_split module

# Step 1: Download the dataset
path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
print("Path to dataset files:", path)

# Step 2: Define parameters for data loading
train_ratio = 0.8
batch_size = 32
image_size = (224, 224)

# Step 3: Get data loaders
# Use the path to the dataset downloaded by kagglehub
train_loader, test_loader = get_data_loaders(data_dir=path, train_ratio=train_ratio, batch_size=batch_size, image_size=image_size)

# Now you can use train_loader and test_loader for model training and evaluation


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (standard for CNNs)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pretrained models
])

# Load dataset
train_dataset = datasets.ImageFolder(root='path_to_train_data', transform=transform)
test_dataset = datasets.ImageFolder(root='path_to_test_data', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)  # Binary classification layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)  # Apply sigmoid for binary output

# Initialize models
models = [CNNModel().to(device) for _ in range(3)]  # Create an ensemble of 3 CNN models

# Train each model separately
num_epochs = 10
for model in models:
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.float().unsqueeze(1).to(device)

            # Forward pass
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Ensemble predictions
model_predictions = []

with torch.no_grad():
    for model in models:
        model.eval()  # Set model to evaluation mode
        predictions = []
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            predictions.append(outputs.cpu().round().numpy())  # Binary predictions, move to CPU
        model_predictions.append(np.concatenate(predictions, axis=0))

# Convert list of predictions to array shape (num_models, num_samples)
model_predictions = np.array(model_predictions)
ensemble_predictions = np.round(np.mean(model_predictions, axis=0))

# Extract true labels from test_dataset for final evaluation
y_test = [label for _, label in test_dataset]
y_test = np.array(y_test)

# Display results
print("Ensemble Model Accuracy:", accuracy_score(y_test, ensemble_predictions))
print(classification_report(y_test, ensemble_predictions))
