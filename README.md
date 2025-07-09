🦴 Bone X-ray Fracture Classification Using Deep Learning
A deep learning-based web application to classify bone X-ray images as Fractured or Non-Fractured using TensorFlow, Keras, and a ResNet50 model.
This project demonstrates the potential of AI in medical image analysis and fracture detection.

🔍 Project Overview
🔎 Input: Grayscale X-ray image of a bone

🤖 Model: ResNet50 (Transfer Learning)

🧠 Output: Classification — Fractured or Non-Fractured with confidence score

🌐 Interface: Web app built with Flask, clean and modern UI

📁 Folder Structure
php
Copy
Edit
BoneFractureApp/
│
├── app.py                  # Flask application
├── bone_fracture_model.h5  # Trained deep learning model
├── static/
│   └── style.css           # Frontend styling
├── templates/
│   └── index.html          # Web interface template
├── uploads/                # Stores uploaded images
└── README.md               # This file
🚀 Features
✅ Upload bone X-ray images through a web interface

✅ Classify image as Fractured or Non-Fractured

✅ Display prediction with confidence score

✅ Image preview on screen

✅ Modern UI with Flask + HTML + CSS

🧠 Deep Learning Model
Architecture: ResNet50 with custom classification head

Input shape: 224x224 grayscale images

Framework: TensorFlow/Keras

Loss: Binary Crossentropy

Optimizer: Adam

Accuracy Achieved: ~64% (baseline model on sample dataset)
