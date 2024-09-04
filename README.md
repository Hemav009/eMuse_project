# Introduction

Music profoundly impacts humans, serving as a powerful tool for relaxation, mood regulation, stress relief, and even enhancing mental and physical performance. Throughout history, across cultures, and in various contexts, music has been a constant companion, providing solace, inspiration, and connection to individuals worldwide.

Facial expression plays a vital role in predicting human emotions and mood. The project aims to create a soothing environment for users by extracting their facial features and detecting their emotions to provide
Personalized curated playlists from Spotify accordingly.

The primary objective of this project is to create a soothing environment for users by dynamically adjusting the music playlist based on their detected emotional states.

# Project Description

## Overview

The emotion recognition model is trained on the FER 2013 dataset. It can detect 7 emotions. The project works by getting a live video feed from a webcam and passing it through the model to get a prediction of emotion. Then according to the emotion predicted, the app will fetch a playlist of songs from Spotify through Spotify wrapper and recommend the songs by displaying them on the screen.

## Features

Real-time expression detection and song recommendations.
Playlists fetched from Spotify using API.
UI for website.

# Running the App

Run pip install -r requirements.txt to install all dependencies.
In Spotipy.py enter your credentials generated by your Spotify Developer account in 'auth_manager'. Note: - This is only required if you want to update recommendation playlists.
Run python app.py and give camera permission if asked.

# Tech Stack

1. Keras
2. Tensorflow
3. Spotify API
4. Flask
5. Front End

# Dataset

The dataset used for this project is the famous FER2013 dataset. Models trained on this dataset can classify 7 emotions.

# Model Architecture

The model architecture is a sequential model consisting of Conv2d, Maxpool2d, Dropout, and Dense layers:

1. Conv2D layers throughout the model have different filter sizes from 32 to 128, all with activation 'relu'
2. Pooling layers have pool size (2,2)
3. Dropout is set to 0.25 as anything above results in poor performance
4. The Final Dense layer has 'softmax' activation for classifying 7 emotions

# Image Processing and Training

The images were normalized, resized to (48,48) and converted to grayscale in batches of 64 with the help of 'ImageDataGenerator' in Keras API.
Training took around 6 hours locally for 75 epochs with an accuracy of ~70 %

# Project Components

Spotipy is a module for establishing a connection to and getting tracks from Spotify using the Spotipy wrapper.
haar cascade is for face detection.
app.py is the main flask application file.
index.html in the 'templates' directory is the web page for the application. Basic HTML and CSS.
train.py is the script for image processing and training the model.

# Further Work

Instead of CSVs, create a database and connect it to the application. The DB will fetch songs for recommendations and new songs can be updated directly onto the database
Add a feature that will update specified playlists for better and more recent recommendations, a specific day over a fixed duration say every Sunday, and append it to the database
Directly play the song.
"# eMuse"

# OUTPUT

![Screenshot 2024-05-22 174826](https://github.com/user-attachments/assets/db884569-17ff-4a63-a4f3-9904d9bba4c5)


![Screenshot 2024-05-22 174848](https://github.com/user-attachments/assets/f1a126e3-2cc8-4095-a841-a0a0c706e157)


![Screenshot 2024-05-22 174856](https://github.com/user-attachments/assets/d1f79ab7-5060-4b9a-ac17-f952dade7e8f)


![Screenshot 2024-05-22 174917](https://github.com/user-attachments/assets/0693f8e5-df53-4f7b-81e9-da047fd70107)



![Screenshot 2024-05-22 174933](https://github.com/user-attachments/assets/afa51715-7526-4854-a804-30a6d3b85bc8)








