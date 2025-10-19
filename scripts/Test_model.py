# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 20:40:56 2025

@author: nater
"""


import torch
import torchvision.models as models
import torch.nn as nn
import pandas as pd
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Define the model with the same architecture used in training
model = models.resnet50(pretrained=False)  # Make sure the same architecture (ResNet50) was used

# Modify the final layer to match the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # Change '4' to match the number of classes in the dataset

# Load the trained model weights
model_path = ""  
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()
print(model)
print("Model successfully loaded and ready for inference!")

# Define the class names
classes = ["Beach", "Dune", "Fluvial", "Nearshore"]  # Update this list according to dataset classes

# Define test data directory
test_dir = ""

# Define image transformations (must match those used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Create a DataLoader for the test dataset
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Prepare lists to store inference results
image_paths = []
true_labels = []
predicted_labels = []

# Perform inference
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs = inputs  # Use inputs.to("cuda") if GPU is available
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Extract labels from tensors
        true_label = labels.item()
        predicted_label = preds.item()

        # Ensure label indices are valid
        if true_label < len(classes) and predicted_label < len(classes):
            # Save image path and prediction results
            img_idx = batch_idx
            image_paths.append(test_dataset.imgs[img_idx][0])
            true_labels.append(classes[true_label])
            predicted_labels.append(classes[predicted_label])
        else:
            print(f"Index out of range! True: {true_label}, Predicted: {predicted_label}")

# Create a DataFrame with inference results
df = pd.DataFrame({
    "Image": image_paths,
    "True Label": true_labels,
    "Predicted Label": predicted_labels
})

# Save results to CSV file
df.to_csv("test_results.csv", index=False)
print("Results saved to 'test_results.csv'")

# Calculate model accuracy on the test dataset
accuracy = (df["True Label"] == df["Predicted Label"]).mean() * 100
print(f"Model accuracy on test dataset: {accuracy:.2f}%")