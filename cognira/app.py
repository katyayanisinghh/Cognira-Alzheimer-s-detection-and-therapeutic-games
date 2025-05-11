from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model (ensure model.h5 is in the root directory)
model = load_model('/Users/katyayanisingh/Desktop/cognira/model.h5')




# Define the target image size as required by the model
target_size = (150, 150)

# Define class names according to the model's output
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No file selected")

    if file:
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict using the model
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_class_index]

        return render_template('index.html', prediction=predicted_class)

    return render_template('index.html', prediction="Something went wrong.")

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
