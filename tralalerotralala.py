import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained emotion model (downloaded model.h5 must be in same folder)
# You can get a pretrained FER2013 model from:
# https://github.com/oarriaga/face_classification or https://github.com/justinshenk/fer
model = load_model("emotion_model.h5")

# Class labels (adjust based on model you download)
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Preprocess face ROI for model
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)
        face_input = np.expand_dims(face_input, axis=-1)  # (1, 48, 48, 1)

        # Predict emotion
        predictions = model.predict(face_input)
        emotion_index = np.argmax(predictions)
        emotion = emotion_labels[emotion_index]
        confidence = predictions[0][emotion_index]

        cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face + Emotion Detection (TF) - Press q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
