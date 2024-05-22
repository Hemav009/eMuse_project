import os
import shutil

fer_test_folder = 'data/test'
ckplus_train_folder = 'ck_data/train'  
merged_test_folder = 'merged_test'

# Create the merged training dataset folder if it doesn't exist
os.makedirs(merged_test_folder, exist_ok=True)

# Define the list of emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Merge training data from FER 2013 dataset
for emotion in emotion_labels:
    fer_emotion_folder = os.path.join(fer_test_folder, emotion)  # Use os.path.join for consistent path handling
    merged_emotion_folder = os.path.join(merged_test_folder, emotion)
    os.makedirs(merged_emotion_folder, exist_ok=True)
    for filename in os.listdir(fer_emotion_folder):
        shutil.copy(os.path.join(fer_emotion_folder, filename), merged_emotion_folder)

# Merge training data from CK+ dataset
for emotion in emotion_labels:
    ckplus_emotion_folder = os.path.join(ckplus_train_folder, emotion)
    merged_emotion_folder = os.path.join(merged_test_folder, emotion)
    os.makedirs(merged_emotion_folder, exist_ok=True)
    for filename in os.listdir(ckplus_emotion_folder):
        shutil.copy(os.path.join(ckplus_emotion_folder, filename), merged_emotion_folder)

# print("Folders created successfully for data merging (assuming no file operations).")
