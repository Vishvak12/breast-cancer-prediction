# Breast Cancer Detection using Ensemble Deep Learning

A deep learning system for classifying breast ultrasound images into **benign**, **malignant**, and **normal** categories. Three ResNet-18-based models with different regularization strategies are combined into a weighted ensemble for improved generalization.

## Overview

Each model uses a **dual-branch architecture**:
- **Image branch**: Pre-trained ResNet-18 backbone extracts 512-dimensional features from the ultrasound image.
- **Mask branch**: A custom CNN processes the corresponding binary segmentation mask.
- Features from both branches are concatenated and passed to a classification head.

Three model variants are trained and then combined:

| Model | File | Regularization |
|---|---|---|
| `ResNetWithMasks` | `resnet_with_masks.pth` | Dropout (0.2) |
| `ResNetWithMasksDropout` | `resnetdropout.pth` | Dropout (0.5, two layers) |
| `ResNetWithMasksL2` | `resnet_with_masks_L2.pth` | L2 / weight decay (0.001) |

The **EnsembleModel** combines predictions with weights `[0.5, 0.3, 0.2]` for Model 1, 2, and 3 respectively.

## Dataset

Uses the [BUSI (Breast Ultrasound Images) dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) with ground-truth segmentation masks.

Expected structure after download:

```
Dataset_BUSI_with_GT/
├── benign/
│   ├── benign (1).png
│   ├── benign (1)_mask.png
│   └── ...
├── malignant/
│   └── ...
└── normal/
    └── ...
```

## Project Structure

```
.
├── dataset.py          # Splits BUSI dataset into train/test sets (80/20)
├── data_loader.py      # Custom PyTorch Dataset and DataLoader for image+mask pairs
├── dropout.py          # Defines and evaluates ResNetWithMasksDropout (Model 2)
├── imageclassifier.py  # Defines and evaluates ResNetWithMasksL2 (Model 3)
├── ensemble.py         # Defines EnsembleModel, runs ensemble evaluation
├── predict.py          # Single-image inference using the ensemble
├── split_dataset/      # Generated train/test split (created by dataset.py)
├── resnet_with_masks.pth
├── resnetdropout.pth
└── resnet_with_masks_L2.pth
```

## Setup

```bash
pip install torch torchvision scikit-learn pillow matplotlib seaborn pandas opencv-python
```

Python 3.9+ and PyTorch 2.x are recommended. GPU is optional but will significantly speed up training.

## Usage

### 1. Prepare the dataset

Place the BUSI dataset in `Dataset_BUSI_with_GT/` and run:

```bash
python dataset.py
```

This creates `split_dataset/train/` and `split_dataset/test/` with an 80/20 split per class.

### 2. Train individual models

```bash
# Train the Dropout model (Model 2)
python dropout.py

# Train the L2 model (Model 3)
python imageclassifier.py
```

Trained weights are saved as `.pth` files in the project root.

### 3. Run ensemble evaluation

```bash
python ensemble.py
```

Loads all three pre-trained models, evaluates the weighted ensemble on the test set, prints metrics, displays a confusion matrix, and saves predictions to `ensemble_predictions.csv`.

### 4. Predict on a single image

Edit the `image_path` and `mask_path` variables in `predict.py`, then run:

```bash
python predict.py
```

Outputs the predicted class (0 = benign, 1 = malignant, 2 = normal) and displays the image with the prediction overlaid.

## Evaluation Metrics

Models are evaluated using:
- Accuracy
- Precision (weighted)
- Recall / Sensitivity (weighted)
- F1 Score (weighted)
- Specificity (per-class average)
- Confusion Matrix
