# Chop Detection System

## Overview

The Chop Detection System is designed to detect and classify chops in document images. The system includes tools for preprocessing images, creating a training dataset, training detection and classification models, and a Flask API for uploading images and obtaining chop detection results.

## Directory Structure


## Resources

- **CSV File**: `annotations.csv` containing preprocessed images paired with their corresponding annotation information.
- **Template HTML File**: `templates/index.html` for the web interface.
- **Preprocessing Script**: `preprocess_test.py` for processing images into YCbCr binarized form.
- **Pairing Script**: `pair_images_annotation.py` for creating the training dataset from preprocessed images and their labeled annotations.
- **Flask API**: `app.py` for uploading image files and obtaining detection results.
- **Detection Model**:
  - Code: `detection/chop_detection_model.py`
  - Model: `detection/chop_detection_model.h5`
- **Classification Model**:
  - Code: `classification/training_model.py`
  - Model: `classification/my_trained_model.h5`

## Setup and Installation

### Prerequisites

- Python 3.7+
- Pip (Python package installer)
- Virtual environment (optional but recommended)

### Installation Steps

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd chop-detection-system
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download or ensure availability of the following files in the respective directories:**
    - `annotations.csv` in the root directory.
    - `chop_detection_model.h5` in the `detection/` directory.
    - `my_trained_model.h5` in the `classification/` directory.
    - `index.html` in the `templates/` directory.

## Usage

### Preprocessing Images

1. **Preprocess images to YCbCr binarized form:**

    ```bash
    python preprocess_test.py
    ```

2. **Create the training dataset from preprocessed images and their labeled annotations:**

    ```bash
    python pair_images_annotation.py
    ```

### Training Models

1. **Train the chop detection model:**

    ```bash
    python detection/chop_detection_model.py
    ```

2. **Train the classification model:**

    ```bash
    python classification/training_model.py
    ```

### Running the Flask API

1. **Start the Flask server:**

    ```bash
    python app.py
    ```

2. **Open a web browser and navigate to:**

    ```
    http://127.0.0.1:5000/
    ```

3. **Upload an image file through the web interface and get the chop detection and classification results.**

## File Descriptions

- **`preprocess_test.py`**: Script to preprocess images into YCbCr binarized form.
- **`pair_images_annotation.py`**: Script to pair preprocessed images with their labeled annotations and create a dataset.
- **`app.py`**: Flask API for uploading image files and obtaining detection results.
- **`detection/chop_detection_model.py`**: Script to train the chop detection model.
- **`detection/chop_detection_model.h5`**: Pre-trained chop detection model.
- **`classification/training_model.py`**: Script to train the classification model.
- **`classification/my_trained_model.h5`**: Pre-trained classification model.


## Permission

This project is shared to KY and Co. for purpose of assessing the internship application of Jose Joaquin Marquez ONLY. 

