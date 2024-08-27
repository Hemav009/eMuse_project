import os
import glob
import numpy as np
import librosa
import soundfile
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib  # Import joblib to save and load the model

# Function to extract features from audio file
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        
        if chroma:
            stft = np.abs(librosa.stft(X))
        
        result = np.array([])
        
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        
        if mel:
            mel_spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            mel = np.mean(mel_spectrogram.T, axis=0)
            result = np.hstack((result, mel))
        
        return result

# Dictionary mapping emotion labels to text
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# List of emotions to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Function to load data and extract features
def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob(r"C:\Users\hemav\OneDrive\Desktop\projects\eMuse\ravdess\Actor_*\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Load data and split into training and testing sets
x_train, x_test, y_train, y_test = load_data(test_size=0.25)

# Print the shape of the training and testing datasets
print(f'Training samples: {x_train.shape[0]}, Testing samples: {x_test.shape[0]}')

# Print the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

# Initialize the Multi-Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# Train the model
model.fit(x_train, y_train)

# Save the model to disk
joblib.dump(model, "emotion_recognition_model.joblib")
print("Model saved as 'emotion_recognition_model.joblib'")

# Predict for the test set
y_pred = model.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")

# To load the model later
loaded_model = joblib.load("emotion_recognition_model.joblib")
print("Model loaded successfully")

# Test the model with some data
sample_features = extract_feature(r"ravdess\Actor_01\03-01-01-01-01-01-01.wav")
predicted_emotion = loaded_model.predict([sample_features])
print(f"Predicted Emotion: {predicted_emotion[0]}")

