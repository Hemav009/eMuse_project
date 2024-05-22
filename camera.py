
from flask import jsonify
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import pandas as pd
from threading import Thread
import datetime

import cv2

# Initialize face cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")



emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model_weights.weights.h5')

cv2.ocl.setUseOpenCL(False)

# Emotion and music dictionaries
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
music_dist = {0: "songs/angry_with_links.csv", 1: "songs/disgusted_with_links.csv", 2: "songs/fearful_with_links.csv", 
              3: "songs/happy_with_links.csv", 4: "songs/neutral_with_links.csv", 5: "songs/sad_with_links.csv", 
              6: "songs/surprised_with_links.csv"}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
show_text = [0]
stop_emotion_detection_flag = False
global df1
df1 = pd.DataFrame()

# Class for calculating FPS while streaming
class FPS:
    def __init__(self):
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._numFrames += 1

    def elapsed(self):
        return (self._end - self._start).total_seconds()

    def fps(self):
        return self._numFrames / self.elapsed()

# Class for using another thread for video streaming
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Class for reading video stream, generating prediction and recommendations
class VideoCamera:
    def get_frame(self):
        global cap1, df1, stop_emotion_detection_flag
        cap1 = WebcamVideoStream(src=0).start()
        image = cap1.read()
        image = cv2.resize(image, (1180, 800))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        if not stop_emotion_detection_flag:
            for (x, y, w, h) in face_rects:
                cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
                roi_gray_frame = gray[y:y+h, x:x+w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                show_text[0] = maxindex
                df1 = pd.read_csv(music_dist[show_text[0]])
                df1 = df1[['Name', 'Album', 'Artist', 'Spotify Link']].head(15)
                cv2.putText(image, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        global last_frame1
        last_frame1 = image.copy()
        img = Image.fromarray(last_frame1)
        img = np.array(img)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes(), df1

def music_rec():
    df = pd.read_csv(music_dist[show_text[0]])
    df = df[['Name', 'Album', 'Artist', 'Spotify Link']]
    return df.head(15)

def stop_emotion_detection():
    global stop_emotion_detection_flag
    stop_emotion_detection_flag = True
    return jsonify({'message': 'Emotion detection stopped successfully.'})
