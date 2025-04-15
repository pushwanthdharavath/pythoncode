## 📦 Facial Expression Dataset
The dataset containing FaceMesh coordinates for training facial expression models is too large for GitHub.  
You can download it from Google Drive:
https://drive.google.com/file/d/10hXbXPVBWSGvAzCtjLor_u77BCaTTQf3/view?usp=sharing


### 📁 Project File Descriptions

- **`train.py`** – Python script used to train the model using the CSV file.
- 
- **`test.py`** – Python script that tests the model in real-time using your webcam.
- 
- **`scaler.pkl`** & **`label.pkl`** – Used to standardize input features and encode facial expression labels.
- 
- **`best_conv1d_model.h5`** – Intermediate model saved before applying model checkpoints.
- 
- **`final_conv1d_model.h5`** – Final model trained after loading `best_conv1d_model.h5` and applying checkpoints.
- 
- **`description.py`** – Displays statistics about the dataset (e.g., number of images per class).
- 
- **`extractpoints.py`** – Script used to extract FaceMesh coordinates from images and save them to a CSV file.
