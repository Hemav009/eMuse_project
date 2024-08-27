import os
from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera, stop_emotion_detection_flag, process_audio, save_audio_file, replay_audio_file
import pandas as pd
import joblib
from speech import extract_feature

app = Flask(__name__)

# Load the trained emotion recognition model
model = joblib.load('emotion_recognition_model.joblib')

# Global variables
headings = ("Name", "Album", "Artist", "Spotify Link")
df1 = pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html', headings=headings, data=df1.to_dict(orient='records'))

def gen(camera):
    global df1
    while True:
        frame, df1 = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/t')
def get_table():
    global df1
    return df1.to_json(orient='records')

@app.route('/stop_emotion_detection', methods=['GET'])
def stop_emotion_detection_route():
    global stop_emotion_detection_flag
    stop_emotion_detection_flag = True
    return jsonify({'message': 'Emotion detection stopped successfully.'})

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    try:
        audio_file = request.files['audio']
        file_path = "recorded_audio.wav"
        audio_file.save(file_path)
        features = extract_feature(file_path)
        emotion = model.predict([features])
        print("Voice emotion: ", emotion[0])
    
        # Clean up temporary file
        os.remove(file_path)
    
        return jsonify({'emotion': emotion[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/replay_audio', methods=['GET'])
def replay_audio():
    audio_data = replay_audio_file()
    return Response(audio_data, mimetype='audio/wav')

if __name__ == '__main__':
    app.debug = True
    app.run()
