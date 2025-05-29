from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your pre-trained Keras model (update path as needed)
model = load_model('/Users/katyayanisingh/Desktop/cognira/model.h5')

# Target image size expected by your model
target_size = (150, 150)

# Class labels in the order of model output
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

@app.route('/')
def index():
    # Initially no prediction
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file part in the request.")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No file selected.")

    if file:
        # Save uploaded image to uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load and preprocess image
        img = image.load_img(filepath, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # normalize pixel values

        # Predict
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_class_index]

        # Pass predicted class to template
        return render_template('index.html', prediction=predicted_class)

    return render_template('index.html', prediction="Something went wrong during prediction.")

if __name__ == '__main__':
    app.run(debug=True)
