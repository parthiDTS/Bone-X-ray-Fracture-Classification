from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model("bone_fracture_model.h5")
class_names = ['Fractured', 'Non-Fractured']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            img_array = preprocess_image(filepath)
            prediction_prob = model.predict(img_array)[0][0]
            prediction = class_names[int(prediction_prob > 0.5)]
            confidence = round(float(prediction_prob if prediction == "Fractured" else 1 - prediction_prob) * 100, 2)
            # Set image_url for UI preview (must be relative to static for browser)
            image_url = f'/static/uploads/{file.filename}'

    return render_template("index.html", prediction=prediction, confidence=confidence, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
