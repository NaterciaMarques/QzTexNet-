# QzTexNet - AI-based quartz grain microtextural classification models.

This document describes the scripts and models developed within the QzTexNet workflow, designed to classify quartz grains according to their microtextural characteristics using deep learning. The workflow includes dataset preparation, model training, and inference procedures.

## Image Preparation

For optimal model performance, all images must have their backgrounds removed prior to training and testing.
Background removal ensures that the neural network focuses solely on the quartz grains, preventing the model from learning irrelevant features related to background colour, texture, or lighting.

All images must be organised into folders according to their sedimentary environment or class before running any of the scripts.

## Scripts Overview:

### 1. Dataset_split.py

This script divides the original dataset into three subsets: training, validation, and testing.
Each class folder within the source directory is randomly split according to predefined ratios (typically 70% for training, 20% for validation, and 10% for testing).
The script automatically creates the folder structure and copies the images into their respective directories.

### 2. Train_model.py

This script trains a ResNet50-based convolutional neural network, referred to as QzTexNet. 
It loads the organised dataset, applies pre-defined data transformations, and optimises the model.
During training, both loss and accuracy metrics are displayed for each epoch, for the training and validation sets.
The best-performing model is saved as a .pth file for later use in testing or inference.

### 3. Test_model.py

This script performs inference using a previously trained QzTexNet model.
It loads the trained weights, predicts the class of unseen test images, and compares the predicted and true labels.
The results are saved in a .csv file that includes each image path, its true label, the predicted label, and the overall model accuracy.


## QzTexNet Models

Five versions of the QzTexNet model were trained and evaluated to assess the impact of the number of epochs and the number of classes on classification performance. All models are based on the ResNet50 architecture, fine-tuned with different datasets and training configurations:

| Model       | Classes | Epochs | Description |
|--------------|----------|---------|-------------|
| QzTexNet1    | 7        | 30      | Initial model trained for 7 classes during 30 epochs. |
| QzTexNet2    | 7        | 50      | Improved model trained for 7 classes during 50 epochs. |
| QzTexNet3    | 7        | 100     | Long-run training with 100 epochs for 7 classes. |
| QzTexNet4    | 4        | 50      | Model retrained for 4 classes during 50 epochs. |
| QzTexNet5    | 4        | 100     | Final refined model trained for 4 classes during 100 epochs. |

## Requirements

The following Python libraries are required to execute the scripts:

- torch 
- torchvision 
- pandas 
- matplotlib
- Pillow 
