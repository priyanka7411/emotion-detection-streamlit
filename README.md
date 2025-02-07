# Emotion Detection from Uploaded Images

This project aims to develop a **Streamlit-based web application** that enables users to upload images and accurately detect and classify the emotion in the image using **Convolutional Neural Networks (CNNs)**. The system utilizes **facial detection** and **facial landmark extraction** to improve the accuracy of emotion classification. The application is designed to be user-friendly and responsive, providing real-time results for emotion detection.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Performance Evaluation](#performance-evaluation)
- [Ethical Considerations](#ethical-considerations)
- [Future Work](#future-work)
- [License](#license)

## Project Overview
This project is built to provide emotion detection from facial expressions in images using **Deep Learning**. The goal is to integrate machine learning with computer vision to create a system that can detect seven primary emotions: **happy, sad, angry, surprised, neutral, disgusted, and fearful**. The project uses **FER-2013**, a dataset containing labeled facial expressions, and implements the solution using **Streamlit** for a user-friendly interface.

The project consists of:
1. **User Interface Development** with Streamlit.
2. **Facial Detection** and **Landmark Extraction** using Dlib and OpenCV.
3. **Emotion Classification** through a CNN model trained on the FER-2013 dataset.
4. **Performance Optimization** for real-time emotion classification.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: 
  - **Streamlit**: For building the web application interface.
  - **PyTorch**: For training and evaluating the CNN model.
  - **OpenCV**: For image processing and facial detection.
  - **Dlib**: For facial landmark extraction.
  - **Torchvision**: For data handling and pre-trained models.
- **Dataset**: **FER-2013** from Kaggle (Facial Expression Recognition dataset).

## System Architecture
The system is divided into several components:
1. **User Interface**: The Streamlit application allows users to upload an image and get emotion classification results.
2. **Facial Detection**: The image is processed to detect faces using OpenCV, and landmarks are extracted using Dlib.
3. **Emotion Classification**: The processed image is passed through a **CNN model**, trained on the FER-2013 dataset, to predict the emotion.
4. **Real-time Performance**: The system has been optimized for real-time processing using techniques like model pruning and quantization.

## Features
- **Image Upload**: Users can upload images in .jpg or .png formats.
- **Emotion Detection**: Displays the predicted emotion along with the confidence score.
- **Real-time Feedback**: The system provides near-instantaneous emotion classification results.
- **Performance Metrics**: Displays the accuracy, precision, recall, and F1 score of the emotion detection model.

## Installation

Follow these steps to set up the project on your local machine:

###  Clone the Repository
Set Up Virtual Environment
Create and activate a Python virtual environment:

python -m venv venv
venv\Scripts\activate

###  Install Dependencies
Install the necessary Python libraries by running the following command:


pip install -r requirements.txt

###  Run the Streamlit Application
Launch the app using Streamlit:


streamlit run app.py

### Usage
1. Open the Streamlit app in your browser.
2. Upload an image with a clear face.
3. The system will display the uploaded image and the predicted emotion, along with the confidence score.

### Model Training
The emotion detection model is trained using the **FER-2013** dataset, which consists of over 35,000 labeled facial images. The CNN model has the following architecture:

- **Convolutional Layers**: Extract relevant features from the images.
- **Fully Connected Layers**: Classify the extracted features into one of the seven emotions.
- **Output Layer**: The final output provides the probability distribution across the seven emotions.

The training is done using the **Adam optimizer** and **cross-entropy loss** for multi-class classification.

### Performance Evaluation
The model was evaluated using the following metrics:

- **Accuracy**: The percentage of correctly classified images.
- **Precision**: The proportion of correct positive predictions among all positive predictions.
- **Recall**: The proportion of correct positive predictions among all actual positive instances.
- **F1 Score**: The harmonic mean of precision and recall.

## Project Report
For a detailed explanation of the system design, implementation, performance analysis, and ethical considerations, please refer to the [Project Report](https://docs.google.com/document/d/1bO8CpsYnHfIW1_s-EJugxfAmVTPcn8IW0P9nQBPAvMw/edit?usp=sharing).


## Ethical Considerations
Emotion detection systems must be used responsibly, considering the following:

- **Privacy Concerns**: Ensure that user-uploaded images are not stored or shared without user consent. Images should only be processed temporarily and discarded after use.
- **Bias in Emotion Detection**: The FER-2013 dataset may not be fully representative of all demographics. The system may exhibit bias in emotion classification for certain age groups, genders, or ethnicities. It is crucial to mitigate such biases by using more diverse datasets and fair training practices.
- **Misuse Potential**: Emotion detection can be misused in surveillance or for manipulating users emotionally. Ethical guidelines should be followed to ensure the technology is used responsibly.

## Future Work
- **Real-Time Video Emotion Detection**: Extend the system to analyze live video feeds for emotion detection.
- **Transfer Learning**: Use pre-trained models like ResNet or VGG for better feature extraction and accuracy.
- **Bias Mitigation**: Train the model with more diverse datasets to minimize biases across different demographics.
- **Improved UI/UX**: Enhance the Streamlit interface with more interactive features and better error handling.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


