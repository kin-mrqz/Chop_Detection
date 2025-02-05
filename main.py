import os
import math
import cv2 as cv
import numpy as np
from PIL import Image
from skimage.feature import hog

# Helper function to convert PDF to images using Pillow
def convert_pdf_to_images(pdf_path):
    images = []
    with Image.open(pdf_path) as img:
        for i in range(img.n_frames):
            img.seek(i)
            img_converted = img.convert('RGB')  # Ensure it's in RGB
            img_array = np.array(img_converted)
            images.append(img_array)
    return images

# Image Processing as before
def preprocess_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    return adaptive_thresh

# Chop Detection as before
def detect_chops(image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    chops = [cv.boundingRect(contour) for contour in contours if is_chop_shape(contour)]
    return chops

def is_chop_shape(contour):
    perimeter = cv.arcLength(contour, True)
    area = cv.contourArea(contour)
    if perimeter == 0:
        return False
    circularity = 4 * math.pi * (area / (perimeter * perimeter))
    return 0.2 < circularity < 1.4

# Feature Extraction and Padding as before
def extract_and_pad_features(chops, image, target_length=256):
    features = []
    for chop in chops:
        x, y, w, h = chop
        roi = image[y:y + h, x:x + w]
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            continue
        gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        hog_features = hog(gray_roi, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2-Hys')
        if len(hog_features) < target_length:
            padded_features = np.pad(hog_features, (0, target_length - len(hog_features)), mode='constant')
        else:
            padded_features = hog_features[:target_length]
        features.append(padded_features)
    return np.stack(features) if features else np.array([])

# Model Training and Testing as before
from sklearn.ensemble import RandomForestClassifier

def train_and_label(features):
    if features.size == 0:
        raise ValueError("Insufficient data for training.")
    labels = np.random.randint(0, 2, size=len(features))
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    return model.predict(features)

# Main Processing Function as before
def process_inputs(input_directory):
    for filename in os.listdir(input_directory):
        filepath = os.path.join(input_directory, filename)
        if filepath.lower().endswith('.pdf'):
            images = convert_pdf_to_images(filepath)
        else:
            images = [cv.imread(filepath)]
        images = [img for img in images if img is not None]

        for img in images:
            preprocessed_img = preprocess_image(img)
            chops = detect_chops(preprocessed_img)
            features = extract_and_pad_features(chops, img)
            if features.size == 0:
                print("No valid features extracted.")
                continue
            labels = train_and_label(features)
            display_labeled_chops(img, chops, labels)

# Display function as before
def display_labeled_chops(img, chops, labels):
    for chop, label in zip(chops, labels):
        cv.rectangle(img, (chop[0], chop[1]), (chop[0] + chop[2], chop[1] + chop[3]), (0, 255, 0), 2)
        cv.putText(img, f'Label: {label}', (chop[0], chop[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv.imshow('Detected Chops', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    input_directory = 'Sample'
    process_inputs(input_directory)
