import logging
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import sqlite3
import os
import tensorflow as tf
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Set up a directory for saving uploaded files
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the directory if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained machine learning model
model_path = "chop_detection_model.h5" #replace with path to detection model
classification_model_path = "chop_classification_model.h5" #replace with path to classification model

# Define custom objects. Needed for the model to be loaded
custom_objects = {
    'adam': Adam,
    'mse': tf.keras.losses.MeanSquaredError()
}

# Load the model with custom objects
model = load_model(model_path, custom_objects=custom_objects)

if os.path.exists(classification_model_path):
    classification_model = load_model(classification_model_path)
else:
    raise FileNotFoundError(f"Classification model not found at {classification_model_path}")

# Database setup
def setup_database():
    conn = sqlite3.connect('chops.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            features BLOB  -- To store image features for clustering
        )
    ''')
    conn.commit()
    conn.close()

setup_database()

def insert_chop(label, features):
    conn = sqlite3.connect('chops.db')
    c = conn.cursor()
    # Insert features as a binary blob
    c.execute("INSERT INTO chops (label, features) VALUES (?, ?)", (label, features))
    conn.commit()
    conn.close()

# Image processing
def process_image(file_path):
    try:
        IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 224, 224, 3
        ROI_WIDTH, ROI_HEIGHT = 64, 64

        image = cv2.imread(file_path)
        if image is None:
            logging.error(f"Failed to read image from {file_path}")
            return None

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # Log the shape of the preprocessed image
        logging.debug(f"Preprocessed image shape: {img.shape}")

        # Predict bounding box
        bbox = model.predict(img)[0]
        logging.debug(f"Raw bounding box prediction: {bbox}")

        xmin, ymin, xmax, ymax = bbox
        xmin, ymin, xmax, ymax = int(xmin * IMG_WIDTH), int(ymin * IMG_HEIGHT), int(xmax * IMG_WIDTH), int(ymax * IMG_HEIGHT)

        # Ensure bounding box coordinates are within image boundaries
        xmin = max(0, min(xmin, IMG_WIDTH - 1))
        ymin = max(0, min(ymin, IMG_HEIGHT - 1))
        xmax = max(0, min(xmax, IMG_WIDTH - 1))
        ymax = max(0, min(ymax, IMG_HEIGHT - 1))

        # Log the adjusted bounding box coordinates
        logging.debug(f"Adjusted bounding box coordinates: ({xmin}, {ymin}, {xmax}, {ymax})")

        # Check for valid ROI size
        if xmin >= xmax or ymin >= ymax:
            logging.error(f"Invalid bounding box coordinates after adjustment: ({xmin}, {ymin}, {xmax}, {ymax})")
            return None

        # Extract the region of interest
        roi = image[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            logging.error("ROI size is zero after extraction")
            return None

        roi = cv2.resize(roi, (ROI_WIDTH, ROI_HEIGHT))
        roi = roi.astype(np.float32) / 255.0
        roi = np.expand_dims(roi, axis=0)

        # Predict label
        pred_class = classification_model.predict(roi)
        label_idx = np.argmax(pred_class, axis=1)[0]

        # Decode the label
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load('label_classes.npy', allow_pickle=True)
        label = label_encoder.inverse_transform([label_idx])[0]

        # Insert chop into database
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            insert_chop(label, descriptors.tobytes())  # Store features as bytes

        return label
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)  # Save the file to the uploads directory
        label = process_image(file_path)
        if label is not None:
            return jsonify({'message': 'File uploaded and processed', 'label': label})
        return jsonify({'message': 'Processing failed'}), 400
    return jsonify({'message': 'No file provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
