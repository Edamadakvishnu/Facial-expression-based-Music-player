import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import h5py
from pygame import mixer

# Ensure compatibility with TensorFlow's Keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Initialize pygame mixer
mixer.init()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define emotion categories
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Define songs for each emotion (Update paths)
emotion_songs = {
    "Happy": r"C:\Users\kotta\music\Buttabomma - SenSongsMp3.Co.mp3",
    "Sad": r"C:\Users\kotta\music\Nee Selavdigi  - SenSongsMp3.Co.mp3",
    "Neutral": r"C:\Users\kotta\music\New-York-Nagaram.mp3",
    "Angry": r"C:\Users\kotta\music\Sultan - Sultan 320 Kbps.mp3",
    "Fear": r"C:\Users\kotta\music\Fear.mp3",
    "Disgust": r"C:\Users\kotta\music\[iSongs.info] 05 - Jagadame.mp3",
    "Surprise": r"C:\Users\kotta\music\Rowdy Baby Maari 2 320 Kbps.mp3",
}

# 🔥 Step 1: Build CNN Model
def build_cnn_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotion classes
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 🔥 Step 2: Train Model (Use FER-2013 Dataset)
def train_model():
    train_dir = r"C:\Users\kotta\Downloads\dataset\train"
    test_dir = r"C:\Users\kotta\Downloads\dataset\test"

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(48, 48), batch_size=32, color_mode="grayscale", class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(48, 48), batch_size=32, color_mode="grayscale", class_mode='categorical')

    model = build_cnn_model()
    history = model.fit(train_generator, validation_data=test_generator, epochs=30, batch_size=32)

    model.save("emotion_model.hdf5")  # Save trained model
    print("Model trained and saved successfully!")

    # 🔥 Print Training, Validation, and Testing Accuracy
    train_acc = history.history['accuracy'][-1] * 100
    val_acc = history.history['val_accuracy'][-1] * 100

    # Evaluate on test dataset
    test_loss, test_acc = model.evaluate(test_generator)
    test_acc *= 100  # Convert to percentage

    print(f"\n🔥 Training Accuracy: {train_acc:.2f}%")
    print(f"🔥 Validation Accuracy: {val_acc:.2f}%")
    print(f"🔥 Testing Accuracy: {test_acc:.2f}%")

    return model

# Check if model exists, else train
model_path = "emotion_model.hdf5"
if os.path.exists(model_path):
    print("Loading existing model...")
    emotion_model = load_model(model_path)
else:
    print("Training new model...")
    emotion_model = train_model()

# 🔥 Step 3: Real-Time Emotion Detection & Music Player
def play_song(emotion):
    """Play a song based on detected emotion."""
    if emotion in emotion_songs:
        mixer.music.load(emotion_songs[emotion])
        mixer.music.play()

# Start Webcam
cap = cv2.VideoCapture(0)
last_detected_emotion = None  # Store last detected emotion

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = face_roi / 255.0  # Normalize

        # Predict Emotion
        emotion_prediction = emotion_model.predict(face_roi)
        emotion_index = np.argmax(emotion_prediction)
        detected_emotion = emotion_labels[emotion_index]

        # Display Emotion on Frame
        cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Play Music if Emotion Changes
        if detected_emotion != last_detected_emotion:
            mixer.music.stop()
            play_song(detected_emotion)
            last_detected_emotion = detected_emotion

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam & close windows
cap.release()
cv2.destroyAllWindows()
