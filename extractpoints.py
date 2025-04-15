import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Path to your dataset
dataset_path = r"C:\Users\DELL\OneDrive\Desktop\face_dataset"
expressions = ['angry', 'happy', 'neutral', 'sad', 'surprise']

data = []
columns = [f'x{i}' for i in range(468)] + [f'y{i}' for i in range(468)] + ['label']

for expression in expressions:
    folder_path = os.path.join(dataset_path, expression)
    extracted_count = 0  # count how many images were processed successfully

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]
            row = x_coords + y_coords + [expression]
            data.append(row)
            extracted_count += 1

    print(f"✅ {expression.upper()}: {extracted_count} images processed successfully.")

# Save to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv('facemesh_expression_data.csv', index=False)
print("\n✅ All expressions processed. CSV file created successfully.")
