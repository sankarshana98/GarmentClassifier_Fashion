# Garment Classifier
![image](https://github.com/user-attachments/assets/bd795e56-62c3-436e-95c9-aef5c5a641e0)

## Project Overview

Fashion Forward is an innovative AI-based e-commerce clothing retailer that aims to revolutionize how customers find and purchase clothing items online. The goal of this project is to develop an image classification system that automatically categorizes new product listings into distinct garment types such as shirts, trousers, shoes, etc. This automated tagging system will enhance the shopping experience by making it easier for customers to find what they're looking for and will assist in inventory management by quickly sorting items.

## Purpose

The primary objective of this project is to create a machine learning model capable of accurately categorizing images of clothing items. By leveraging Convolutional Neural Networks (CNNs), we aim to achieve high accuracy in classifying garment types, thereby streamlining the product tagging process and improving overall efficiency.

## How It Works

### Data Preparation

We use the FashionMNIST dataset, which consists of 70,000 grayscale images of 10 different types of clothing items. The dataset is divided into a training set of 60,000 images and a test set of 10,000 images. Each image is a 28x28 pixel square.

### Model Architecture

We define a simple yet effective Convolutional Neural Network (CNN) with the following layers:
- **Convolutional Layers**: Two convolutional layers with ReLU activation and max-pooling to extract features from the images.
- **Fully Connected Layers**: Two fully connected layers to classify the extracted features into one of the 10 garment categories.
- **Dropout Layer**: A dropout layer to prevent overfitting.

### Training the Model

The model is trained using the training dataset for a few epochs (to keep the runtime manageable). During training, the model learns to minimize the cross-entropy loss function using the Adam optimizer.

### Evaluation

After training, the model's performance is evaluated on the test dataset. We compute predictions and calculate accuracy, precision, and recall for each garment category to assess the model's effectiveness.


