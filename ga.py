import cv2
import numpy as np
from tensorflow.keras.models import load_model

class FaceEmotionDetector:
    def __init__(self, model_path):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.img_size = 48
        self.model = load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_emotion(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None, []
        
        highest_confidence = 0.0
        best_emotion = "Neutral"
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (self.img_size, self.img_size))
            roi = roi_gray.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            
            prediction = self.model.predict(roi, verbose=0)[0]
            emotion_idx = np.argmax(prediction)
            confidence = float(prediction[emotion_idx])
            
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_emotion = self.emotions[emotion_idx]
        
        return best_emotion, faces

model_path = "lightweight_emotion_model_best.h5"
detector = FaceEmotionDetector(model_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    best_emotion, faces = detector.detect_emotion(frame)
    
    if best_emotion and faces is not None:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, best_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Face Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()