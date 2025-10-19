# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:54:05 2025

@author: nater
"""

"""
This script trains, validates, and tests a deep learning model for image classification 
using transfer learning with a pre-trained ResNet-50 architecture. 

The workflow includes:
1. Loading and augmenting grayscale images converted to 3-channel RGB format.
2. Splitting data into training, validation, and testing sets.
3. Training the model over multiple epochs (50) while tracking loss and accuracy.
4. Selecting and saving the best-performing model based on validation accuracy.
5. Evaluating final performance on the test dataset.

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import copy

# Define dataset directories
data_dir = ""  # Path to main image dataset
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

# Hyperparameters
batch_size = 16
num_epochs = 50
learning_rate = 0.001
num_classes = 6  # Total number of classes

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations and augmentation (adapted for grayscale images)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAutocontrast(),
        transforms.RandomInvert(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalise for 3 channels
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, transform=data_transforms['val']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}

# Print detected class labels
print("Classes found in test dataset:", image_datasets['test'].class_to_idx)

# Create dataloaders
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False),
    'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False)
}

# Load pre-trained ResNet-50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Replace final fully connected layer for custom number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Send model to GPU (if available)
model = model.to(device)

# Define loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to train the model
def train_model(model, dataloaders, criterion, optimizer, num_epochs=50):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Enable training mode
            else:
                model.eval()   # Enable evaluation mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Save best model weights based on validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Best validation accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# Train model
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs)

# Save trained model
torch.save(model.state_dict(), "model_name.pth")
print("Model saved successfully!")

# Function to evaluate model performance on the test set
def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

    accuracy = correct.double() / total
    print(f"Test set accuracy: {accuracy:.4f}")

# Test the model
test_model(model, dataloaders["test"])
