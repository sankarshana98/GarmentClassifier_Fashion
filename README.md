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


## Code Explanation

The code is structured as follows:

### Data Loading
We load the FashionMNIST dataset using `torchvision.datasets` and apply transformations to convert the images to tensors.

### Model Definition
A CNN model is defined with convolutional, pooling, and fully connected layers to process and classify the images.

### Training Loop
The model is trained for a few epochs using the training data, computing the loss and updating the model weights using backpropagation.

### Evaluation Loop
The model is evaluated on the test data, and predictions are collected.

### Metrics Calculation
We calculate accuracy, precision, and recall using `torchmetrics` to measure the model's performance.

## Metrics

The model's performance is summarized by the following metrics:

- **Accuracy**: The overall accuracy of the model on the test dataset is 0.8945.
- **Precision**: The precision for each garment category is as follows:
  - T-shirt/top: 0.7952
  - Trouser: 0.9918
  - Pullover: 0.8573
  - Dress: 0.8936
  - Coat: 0.7806
  - Sandal: 0.9848
  - Shirt: 0.7591
  - Sneaker: 0.9504
  - Bag: 0.9829
  - Ankle boot: 0.9527
- **Recall**: The recall for each garment category is as follows:
  - T-shirt/top: 0.8890
  - Trouser: 0.9730
  - Pullover: 0.8050
  - Dress: 0.8990
  - Coat: 0.8860
  - Sandal: 0.9700
  - Shirt: 0.6240
  - Sneaker: 0.9580
  - Bag: 0.9750
  - Ankle boot: 0.9660

## Conclusion

This project demonstrates a practical application of deep learning for image classification in the fashion industry. By automating the product tagging process, Fashion Forward can enhance the shopping experience for customers and streamline inventory management, ultimately contributing to more efficient operations and better customer satisfaction.
