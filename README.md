Facial Expression-Based Music Player

An AI-powered music player that detects facial expressions in real time using OpenCV and TensorFlow, then plays songs that match the detected mood. A CNN classifies emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral), and pygame handles audio playback.

✨ Features

🎥 Real-time face & emotion detection (webcam + Haar Cascade)

🧠 CNN model (48×48 grayscale) trained on FER-style datasets

🎼 Auto-play tracks mapped to each emotion

💾 Saves/loads model as emotion_model.hdf5

⚙️ Built with Python, OpenCV, TensorFlow/Keras, pygame

train_dir = r"C:\Users\kotta\Downloads\dataset\train"
test_dir  = r"C:\Users\kotta\Downloads\dataset\test"

Prerequisites
Python 3.9–3.11 recommended
Webcam
OS audio support (pygame/SDL)

(Windows) Build Tools may be needed for h5py in some environments

