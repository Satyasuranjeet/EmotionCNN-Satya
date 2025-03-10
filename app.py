import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

app = Flask(__name__)

class FaceEmotionDetector:
    def __init__(self, model_path):
        # Define emotion labels
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.img_size = 48  # FER dataset uses 48x48 grayscale images
        self.model = load_model(model_path)  # Load the pre-trained model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_emotion(self, image):
        """
        Detect faces in the image and predict their emotions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        result = []
        for (x, y, w, h) in faces:
            # Extract the face ROI
            roi_gray = gray[y:y+h, x:x+w]
            
            # Resize to expected size
            roi_gray = cv2.resize(roi_gray, (self.img_size, self.img_size))
            
            # Normalize and reshape for model input
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            
            # Make prediction
            prediction = self.model.predict(roi, verbose=0)[0]
            
            # Get max confidence emotion
            emotion_idx = np.argmax(prediction)
            emotion = self.emotions[emotion_idx]
            confidence = prediction[emotion_idx]
            
            result.append((emotion, confidence))
        
        return result
    
    def visualize_results(self, image, results):
        """
        Draw the results on the image
        """
        output = image.copy()
        
        for (x, y, w, h), emotion, confidence in results:
            # Draw rectangle around face
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add text with emotion and confidence
            text = f"{emotion}: {confidence:.2f}"
            y_offset = y - 10 if y - 10 > 10 else y + h + 10
            cv2.putText(output, text, (x, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return output

# Initialize the detector with the model path
model_path = "lightweight_emotion_model_best.h5"
detector = FaceEmotionDetector(model_path)

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    results = detector.detect_emotion(image)
    
    if not results:
        return jsonify({"error": "No face detected"}), 400
    
    max_confidence_emotion = max(results, key=lambda x: x[1])
    
    return jsonify({"emotion": max_confidence_emotion[0]})

@app.route('/webcam_demo', methods=['GET'])
def webcam_demo():
    """Run a demo using webcam feed"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return jsonify({"error": "Could not open webcam"}), 500
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        results = detector.detect_emotion(frame)
        output_frame = detector.visualize_results(frame, results)
        
        cv2.imshow('Emotion Detection', output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Webcam demo ended"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)