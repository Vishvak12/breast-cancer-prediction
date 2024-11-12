import kagglehub
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from dataset import get_data_loaders  # Custom loader module

# Step 1: Download the dataset
path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
print("Path to dataset files:", path)

# Step 2: Load dataset using get_data_loaders
train_loader, test_loader = get_data_loaders(data_dir=path, train_ratio=0.8, batch_size=32, image_size=(224, 224))

# Step 3: Define processor and model for ensemble
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18").to(device)
ensemble_models = [model for _ in range(3)]  # Ensemble of three models

# Step 4: Predictions with the ensemble
model_predictions = []

with torch.no_grad():
    for model in ensemble_models:
        predictions = []
        for X_batch, _ in test_loader:
            inputs = image_processor(X_batch, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            predicted = logits.argmax(-1)
            predictions.append(predicted.cpu().numpy())
        model_predictions.append(np.concatenate(predictions))

# Averaging predictions
ensemble_predictions = np.round(np.mean(np.array(model_predictions), axis=0))

# Step 5: Calculate and report metrics
y_test = [label for _, label in test_loader.dataset.samples]
y_test = np.array(y_test)

print("Ensemble Model Accuracy:", accuracy_score(y_test, ensemble_predictions))
print(classification_report(y_test, ensemble_predictions))
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

# Step 5: Calculate and report metrics
y_test = [label for _, label in test_loader.dataset.samples]
y_test = np.array(y_test)

# Accuracy
accuracy = accuracy_score(y_test, ensemble_predictions)

# Classification report includes precision, recall, F1, and support
report = classification_report(y_test, ensemble_predictions, target_names=['Class 0', 'Class 1'])
precision = precision_score(y_test, ensemble_predictions, average='binary')
recall = recall_score(y_test, ensemble_predictions, average='binary')
f1 = f1_score(y_test, ensemble_predictions, average='binary')

# Print out metrics
print("Ensemble Model Accuracy:", accuracy)
print("Ensemble Model Precision:", precision)
print("Ensemble Model Recall:", recall)
print("Ensemble Model F1 Score:", f1)
print("Detailed Classification Report:\n", report)