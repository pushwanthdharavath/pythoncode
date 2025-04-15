import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model

# Load the trained model and pre-processing tools
model = load_model('final_conv1d_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Capture from webcam
cap = cv2.VideoCapture(0)
print("Press 'c' to capture")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Press 'c' to capture", frame)
    
    key = cv2.waitKey(1)
    if key == ord('c'):
        img = frame.copy()
        break

cap.release()
cv2.destroyAllWindows()

# Convert image to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = face_mesh.process(img_rgb)

if results.multi_face_landmarks:
    landmarks = results.multi_face_landmarks[0]
    points = []
    for lm in landmarks.landmark:
        points.extend([lm.x, lm.y])

    points_np = np.array(points).reshape(1, -1)
    points_scaled = scaler.transform(points_np)
    points_input = points_scaled.reshape(-1, 936, 1)
    
    pred = model.predict(points_input)[0]
    class_index = np.argmax(pred)
    expression = label_encoder.inverse_transform([class_index])[0]

    # Print all class probabilities
    print("Predicted Probabilities:")
    for i, prob in enumerate(pred):
        label = label_encoder.inverse_transform([i])[0]
        print(f"{label}: {prob:.4f}")

    print("Final Prediction:", expression)
