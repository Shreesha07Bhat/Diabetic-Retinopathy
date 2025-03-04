# app.py
from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the class labels as per the training script's diagnosis_mapping
class_labels = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

# Load the trained model (ensure the correct path)
try:
    model = tf.keras.models.load_model('diabetic\model_final.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    """
    Load and preprocess the image, then make a prediction using the trained model.
    
    Args:
        img_path (str): Path to the image file.
    
    Returns:
        tuple: Predicted class label and confidence score.
    """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = x / 255.0  # Normalize to [0,1]
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        predictions = model.predict(x)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions, axis=1)[0]

        return class_labels[predicted_class], confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Handle file uploads and make predictions.
    
    GET: Render the upload form.
    POST: Process the uploaded file, make a prediction, and render the result.
    """
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            # Since we're removing flash, provide feedback through the rendered template
            return render_template('index.html', message='No file part in the request.', message_type='danger')
        file = request.files['file']
        
        # If user does not select a file, browser may submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', message='No file selected for uploading.', message_type='warning')
        print(allowed_file(file.filename))
        if not model:
            return render_template('index.html', message='Model is Not loaded contact the provider', message_type='danger')
        if file and allowed_file(file.filename):
            print(request.files)    
            # Secure the filename to prevent directory traversal attacks
            filename = secure_filename(file.filename)
            
            # Ensure the 'uploads' directory exists
            upload_dir = os.path.join(app.root_path, 'uploads')
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            
            # Save the uploaded file to the 'uploads' directory
            filepath = os.path.join(upload_dir, filename)
            try:
                file.save(filepath)
                print(f"File saved to {filepath}.")
            except Exception as e:
                return render_template('index.html', message='Failed to save the uploaded file.', message_type='danger')
            
            # Make prediction
            prediction, confidence = predict_image(filepath)
            
            if prediction:
                # Optionally, remove the file after prediction to save space
                try:
                    os.remove(filepath)
                    print(f"File {filepath} removed after prediction.")
                except Exception as e:
                    print(f"Error removing file: {e}")
                
                # Render the result template with prediction and confidence
                return render_template('result.html', result=prediction, confidence=round(confidence * 100, 2) )
            else:
                return render_template('index.html', message='An error occurred during prediction.', message_type='danger')
        else:
            return render_template('index.html', message='Allowed file types are png, jpg, jpeg.', message_type='warning')
    
    return render_template('index.html')

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)