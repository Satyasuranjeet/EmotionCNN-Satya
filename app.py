import cv2
import numpy as np
from tensorflow.keras.models import load_model

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
        
        Args:
            image: numpy array of the image (BGR format from OpenCV)
            
        Returns:
            List of tuples with (face_box, emotion, confidence)
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
            
            result.append(((x, y, w, h), emotion, confidence))
        
        return result
    
    def visualize_results(self, image, results):
        """
        Draw the results on the image
        
        Args:
            image: Original image
            results: List of (face_box, emotion, confidence) tuples
            
        Returns:
            Image with annotated faces and emotions
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


def demo_with_webcam(model_path):
    """Run a demo using webcam feed"""
    detector = FaceEmotionDetector(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Detect emotions
        results = detector.detect_emotion(frame)

        # Visualize results
        output_frame = detector.visualize_results(frame, results)
        
        # Display result
        cv2.imshow('Emotion Detection', output_frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Path to your trained model
    model_path = "lightweight_emotion_model_best.h5"
    
    # Run the webcam demo
    demo_with_webcam(model_path)