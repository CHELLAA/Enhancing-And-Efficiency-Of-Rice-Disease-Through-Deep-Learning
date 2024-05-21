# Rice Leaf Disease Classification

This repository contains the implementation of a project for classifying rice leaf diseases using multiple machine learning and deep learning techniques. The project includes preprocessing steps, feature extraction, and model training and evaluation using Decision Tree, ResNet50, and a custom CNN model.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Models](#models)
  - [Decision Tree](#decision-tree)
  - [ResNet50](#resnet50)
  - [Custom CNN](#custom-cnn)
- [Results](#results)
- [Conclusion](#conclusion)
- [How to Use](#how-to-use)
- [References](#references)

## Introduction

This project aims to classify different types of diseases in rice leaves using machine learning and deep learning techniques. Identifying and classifying these diseases can help in taking timely actions to improve crop yield and health.

## Dataset

The dataset consists of images of rice leaves categorized into different classes representing various diseases and healthy leaves. The dataset is organized in a directory structure where each subdirectory contains images of a specific class.

## Preprocessing

Each image is preprocessed by resizing it to 224x224 pixels to match the input size required by CNN models. Gaussian blur is applied to reduce noise in the images.

## Feature Extraction

Histogram of Oriented Gradients (HOG) features are extracted from each image to be used as input for the Decision Tree classifier.

## Models

### Decision Tree

A Decision Tree classifier is trained using HOG features extracted from the images. Decision Trees are chosen for their simplicity and interpretability.

### ResNet50

The ResNet50 model with pre-trained weights is used, with additional dense layers for fine-tuning to our specific dataset. ResNet50 is chosen for its strong performance in image classification tasks due to its deep architecture and residual connections.

### Custom CNN

A custom Convolutional Neural Network (CNN) is defined and trained from scratch to classify the rice leaf images. CNNs are well-suited for image classification tasks because they can automatically learn and extract relevant features from the input images.

## Results

The performance of each model is evaluated using accuracy, precision, recall, and F1 score. The results show the effectiveness of each approach in classifying rice leaf diseases.

## Conclusion

This project demonstrates the application of different machine learning techniques for rice leaf disease classification. The use of pre-trained models like ResNet50 significantly improves the accuracy compared to traditional classifiers like Decision Tree.

## How to Use

1. Clone the repository.
2. Navigate to the project directory.
3. Install the required packages.
4. Run the Jupyter notebook or Python scripts to preprocess data, train models, and evaluate performance.

## References

- TensorFlow
- scikit-learn
- Keras Applications
- Histogram of Oriented Gradients (HOG)

---

Feel free to customize this description further based on the specifics of your project and any additional details you'd like to include.
