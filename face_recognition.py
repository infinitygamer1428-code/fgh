import cv2
import numpy as np
import base64
from PIL import Image
import io
import os

class FaceRecognition:
    def __init__(self):
        # Load OpenCV's pre-trained face detector
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
    def get_face_encoding_from_file(self, image_path):
        """
        Extract face encoding from image file
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Get the largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size
            face_resized = cv2.resize(face_roi, (128, 128))
            
            # Normalize
            face_normalized = face_resized / 255.0
            
            # Flatten to create encoding
            encoding = face_normalized.flatten()
            
            return encoding
            
        except Exception as e:
            print(f"Error in face encoding: {e}")
            return None
    
    def get_face_encoding(self, image_data):
        """
        Extract face encoding from base64 image data
        """
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
            
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (128, 128))
            face_normalized = face_resized / 255.0
            encoding = face_normalized.flatten()
            
            return encoding
            
        except Exception as e:
            print(f"Error in face encoding: {e}")
            return None
    
    def compare_faces(self, encoding1, encoding2):
        """
        Compare two face encodings using correlation
        """
        if encoding1 is None or encoding2 is None:
            return 1.0
        
        # Ensure same length
        min_len = min(len(encoding1), len(encoding2))
        encoding1 = encoding1[:min_len]
        encoding2 = encoding2[:min_len]
        
        try:
            # Calculate correlation
            correlation = np.corrcoef(encoding1, encoding2)[0, 1]
            # Convert to distance (0 = perfect match, 1 = no match)
            distance = 1 - abs(correlation)
            return distance
        except:
            return 1.0
    
    def detect_faces_in_image(self, image_path):
        """
        Detect faces in image and return count and locations
        """
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            return {
                'count': len(faces),
                'faces': [{'x': x, 'y': y, 'w': w, 'h': h} for (x, y, w, h) in faces]
            }
        except Exception as e:
            print(f"Error in face detection: {e}")
            return {'count': 0, 'faces': []}