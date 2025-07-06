Chop Detection and Classification System
This repository contains a full pipeline for detecting and classifying stamps/chops on document images. It includes preprocessing, annotation, training, and a Flask-based front-end interface for inference.

📁 Directory Structure
graphql
Copy
Edit
├── app.py                       # Flask app: serves UI, handles uploads, runs predictions
├── main.py                      # Core pipeline: preprocessing, detection, classification
├── chop_detection_model.py      # Model script for chop localization (bounding box regression)
├── chop_classification_model.py # Model script for chop type classification
├── pair_images_annotations.py   # Script to pair images and label files, outputs annotations.csv
├── preprocess_test.py           # Script to test preprocessing from Sample to Labelled
├── Annotations.csv              # Generated file: image labels + bounding box coordinates
├── chops.db                     # SQLite database to store and retrieve chop data
├── templates/
│   └── index.html               # Front-end HTML template for uploading and visualizing results
├── Sample/                      # Folder of raw document images with chops
├── Processed/                   # Folder of processed images (e.g., YCbCr color space)
├── Labelled/                    # Folder containing `.xml` labels generated with labelImg
🚀 Workflow Overview
Sample → Labelled
Use labelImg to annotate chops on document images from the Sample/ folder. Save annotations in Labelled/.

Labelled + Processed → Annotations.csv
Run pair_images_annotations.py to pair processed images with their labels and generate Annotations.csv.

Train Detection Model
Use chop_detection_model.py to train a model to predict chop bounding boxes.

Train Classification Model
Use chop_classification_model.py to train a model to classify detected chops (e.g., round, square, etc).

Run Flask App
Launch app.py to start the web interface. Upload images, view predictions, and interact with the model.

❗ Model Files Not Included
This repository does not include the following trained model files due to GitHub's size limit:

chop_detection_model.h5

chop_classification_model.h5

To use the full pipeline, you will need to train these models yourself using the provided scripts.

⚙️ Requirements
Install required Python packages using:

bash
Copy
Edit
pip install -r requirements.txt
(You can generate the requirements.txt via pip freeze > requirements.txt after installing dependencies like tensorflow, flask, opencv-python, pandas, etc.)

📬 Running the App
bash
Copy
Edit
python app.py
Navigate to http://localhost:5000 to access the web interface.

📝 Notes
Preprocessing uses YCbCr color space to enhance chop visibility.

Detection is treated as a regression problem (bounding box coordinates).

Classification is used to determine the shape/type of chop.

All annotation data is centralized in Annotations.csv.

