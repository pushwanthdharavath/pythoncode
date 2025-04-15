import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib

# 1. Load dataset
df = pd.read_csv("facemesh_expression_data.csv")  # Make sure this file exists

# 2. Separate features and label
X = df.drop('label', axis=1).values
y = df['label'].values

# 3. Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and label encoder
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

# 5. Reshape for Conv1D (936, 1)
X_cnn = X_scaled.reshape(-1, 936, 1)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

# 7. Build Conv1D model
model = Sequential([
    Input(shape=(936, 1)),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

# 8. Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 9. Callbacks (checkpoint)
checkpoint = ModelCheckpoint(
    'best_conv1d_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# 10. Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint]
)

# 11. Save final model
model.save('final_conv1d_model.h5')
