import cv2
import numpy as np
import os
import requests
import json
from flask import Flask, Response, render_template, request
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained models
model_emotion = load_model(os.path.join(os.path.dirname(__file__), 'static/emotion_model.h5'))
model_gender = load_model(os.path.join(os.path.dirname(__file__), 'static/gender_model.h5'))
model_age = load_model(os.path.join(os.path.dirname(__file__), 'static/age_model.h5'))

# Define the video capture
cap = cv2.VideoCapture(0)

# Define the YouTube API key
YOUTUBE_API_KEY = 'API KEY'

def detect_emotion(frame):
    # Preprocess the image for emotion detection
    img = cv2.resize(frame, (48, 48))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Perform emotion detection
    emotion_predictions = model_emotion.predict(img)
    emotion_label = np.argmax(emotion_predictions)

    return emotion_label

def detect_gender(frame):
    # Preprocess the image for gender detection
    img = cv2.resize(frame, (96, 96))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Perform gender detection
    gender_predictions = model_gender.predict(img)
    gender_label = np.argmax(gender_predictions)

    return gender_label

def detect_age(frame):
    # Preprocess the image for age detection
    img = cv2.resize(frame, (96, 96))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Perform age detection
    age_predictions = model_age.predict(img)
    age_label = np.argmax(age_predictions)

    return age_label

def recommend_music(emotion):
    # Define the YouTube API URL
    url = f'https://www.googleapis.com/youtube/v3/search?part=snippet&q={emotion}%20music&type=video&maxResults=1&key={YOUTUBE_API_KEY}'

    # Perform the YouTube API request
    response = requests.get(url)
    data = json.loads(response.text)

    # Extract the video title
    video_title = data['items'][0]['snippet']['title']

    # Extract the video URL
    video_url = f'https://www.youtube.com/watch?v={data["items"][0]["id"]["videoId"]}'

    return video_title, video_url

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        emotion = detect_emotion(frame)
        gender = detect_gender(frame)
        age = detect_age(frame)

        # Recommend music based on the detected emotion
        video_title, video_url = recommend_music(emotion)

        # Overlay the information on the frame
        cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Gender: {gender}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Age: {age}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Recommended Music: {video_title}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the frame to JPEG and return it
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
