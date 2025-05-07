import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
import os
import threading
from playsound import playsound
from keras.utils import register_keras_serializable
from tensorflow.keras.models import model_from_json, Sequential


@register_keras_serializable()
class CustomSequential(Sequential):
    pass

with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()


model = model_from_json(model_json, custom_objects={'Sequential': CustomSequential})


model.load_weights("facialemotionmodel.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1).astype('float32')
    return feature / 255.0


def play_music(emotion):
    music_file = f"music/{emotion}.mp3"
    if os.path.exists(music_file):
        try:
            playsound(music_file)
        except Exception as e:
            print(f"Error playing {music_file}: {e}")


webcam = cv2.VideoCapture(0)


labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}


if not os.path.exists("emotion_logs"):
    os.mkdir("emotion_logs")
csv_filename = f"emotion_logs/emotion_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
emotion_log = pd.DataFrame(columns=["Time", "Emotion", "Confidence"])


emotion_counts = {label: 0 for label in labels.values()}

current_emotion = None
music_thread = None

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    detected_emotion = None

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face_features = extract_features(face)

        prediction = model.predict(face_features, verbose=0)
        emotion_index = int(np.argmax(prediction))
        label = labels[emotion_index]
        confidence = np.max(prediction)


        emotion_counts[label] += 1


        new_entry = pd.DataFrame({
            "Time": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Emotion": [label],
            "Confidence": [round(confidence * 100, 2)]
        })
        emotion_log = pd.concat([emotion_log, new_entry], ignore_index=True)

        text = f"{label} ({confidence*100:.1f}%)"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        detected_emotion = label

    if detected_emotion and detected_emotion != current_emotion:
        current_emotion = detected_emotion
        if music_thread and music_thread.is_alive():
            pass
        else:
            music_thread = threading.Thread(target=play_music, args=(current_emotion,))
            music_thread.start()

    y0, dy = 20, 25
    for i, (emotion, count) in enumerate(emotion_counts.items()):
        text = f"{emotion}: {count}"
        cv2.putText(frame, text, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection with Music", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

emotion_log.to_csv(csv_filename, index=False)
print(f"Emotion log saved to {csv_filename}")

webcam.release()
cv2.destroyAllWindows()
