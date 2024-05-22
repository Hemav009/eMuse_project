from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera, stop_emotion_detection, stop_emotion_detection_flag 
import pandas as pd

app = Flask(__name__)

headings = ("Name", "Album", "Artist", "Spotify Link")
df1 = pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html', headings=headings, data=df1)

def gen(camera):
    while True:
        global df1
        frame, df1 = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/t')
def gen_table():
    return df1.to_json(orient='records')

@app.route('/stop_emotion_detection')
def stop_emotion_detection_route():
    global stop_emotion_detection_flag
    stop_emotion_detection_flag = True
    return jsonify({'message': 'Emotion detection stopped successfully.'})

if __name__ == '__main__':
    app.debug = True
    app.run()
