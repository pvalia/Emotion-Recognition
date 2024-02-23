# Emotion Detection Using Neural Networks
This project focuses on the emotion recognition of individuals from images utilizing two different Convolutional Neural Networks (CNN) architectures. By leveraging deep learning, the project aims to accurately classify the emotion of a person based on their facial expressions. The models are inspired and adapted from state-of-the-art architectures discussed in prominent Kaggle notebooks.

## Project Overview

The core of this project lies in its use of two CNN architectures for emotion classification. These models were developed with reference to the following sources:

- Facial Emotion Recognition - Image Classification: [Kaggle Notebook by Myr9988](https://www.kaggle.com/code/myr9988/facial-emotion-recognition-image-classification)
- Emotion Detector: [Kaggle Notebook by Aayushmishra1512](https://www.kaggle.com/code/aayushmishra1512/emotion-detector/notebook)

The objective is to compare these models in terms of accuracy, efficiency, and performance on a standardized emotion dataset, providing insights into the applicability of each model for real-world applications.

# Demo
Below are the accuracy and loss graphs for the 2 models that were trained:

## Model 1
![image](https://github.com/pvalia/Emotion-Recognition/assets/77172929/631abcb1-7894-410b-8b4d-84e48121e18d)
![image](https://github.com/pvalia/Emotion-Recognition/assets/77172929/6b7bfb88-c602-44be-9d29-f5383ebae230)

## Model 2
![image](https://github.com/pvalia/Emotion-Recognition/assets/77172929/1eb342c7-3290-4bf1-b15b-ff33fe83d51e)
![image](https://github.com/pvalia/Emotion-Recognition/assets/77172929/b7f79a92-0973-477a-b0b3-9c98fae129fd)

To run the code you can either run each cell separately or the whole script. A sample output of the trained data:

![image](https://github.com/pvalia/Emotion-Recognition/assets/77172929/706db422-863c-4983-aa1b-824cbe74a31c)
![image](https://github.com/pvalia/Emotion-Recognition/assets/77172929/6aea1944-397a-4ff7-8ac9-f578a4f85f42)
![image](https://github.com/pvalia/Emotion-Recognition/assets/77172929/2d549f07-c216-4170-9f6e-ff8249568527)

## Getting Started

These instructions will guide you through the setup process to run the emotion recognition models on your local machine for development and testing purposes.

### Requirements
The dataset that the models were trained on can be downloaded from: https://www.kaggle.com/datasets/msambare/fer2013

- pandas
- numpy
- keras
- tensorflow
- scikit-learn
