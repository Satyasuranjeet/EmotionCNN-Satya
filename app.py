from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

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
            return None
        
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
        
        return best_emotion

model_path = "lightweight_emotion_model_best.h5"
detector = FaceEmotionDetector(model_path)

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    best_emotion = detector.detect_emotion(image)
    
    if best_emotion is None:
        return jsonify({"error": "No face detected"}), 400
    
    return jsonify({"emotion": best_emotion})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
