import torch.nn as nn


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
import torch.optim as optim

num_epochs = 20
models = [SimpleNN() for _ in range(3)]  # Create an ensemble of 3 models

for model in models:
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

import numpy as np

model_predictions = []

with torch.no_grad():
    for model in models:
        predictions = []
        for X_batch, _ in test_loader:
            outputs = model(X_batch).squeeze()
            predictions.append(outputs.round().numpy())  # Binary predictions
        model_predictions.append(np.concatenate(predictions, axis=0))

# Convert list of predictions to an array of shape (num_models, num_samples)
model_predictions = np.array(model_predictions)

ensemble_predictions = np.round(np.mean(model_predictions, axis=0))

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, ensemble_predictions)
report = classification_report(y_test, ensemble_predictions)
print("Ensemble Model Accuracy:", accuracy)
print(report)
