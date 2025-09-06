import cv2
from fer import FER  # Install first: pip install fer

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the FER emotion detector
emotion_detector = FER(mtcnn=True)

# Start video capture from the default webcam (0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret will be True
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert frame to grayscale (Face detection works better on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles and detect emotions
    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


        # Detect emotions
        result = emotion_detector.detect_emotions(face_roi)
        if result:
            # Get the emotion with highest score
            emotions = result[0]["emotions"]
            top_emotion = max(emotions, key=emotions.get)
            text = f"{top_emotion} ({emotions[top_emotion]*100:.1f}%)"

            # Put text above face rectangle
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face + Emotion Detection - Press q to Quit', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
