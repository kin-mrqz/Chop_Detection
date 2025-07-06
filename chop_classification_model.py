import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Constants
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3

# Load image and preprocess
def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img.astype(np.float32) / 255.0

# Create dataset from CSV file
def create_dataset_from_csv(csv_path):
    data = pd.read_csv(csv_path)
    images = []
    bboxes = []
    labels = []

    for index, row in data.iterrows():
        img = load_image(row['ImagePath'])
        bbox = [row['XMin'], row['YMin'], row['XMax'], row['YMax']]
        label = row['Label']

        images.append(img)
        bboxes.append(bbox)
        labels.append(label)

    return np.array(images), np.array(bboxes), np.array(labels)

csv_path = r"C:\Users\User\Desktop\Uni\Career\KyAssignment\annotations.csv" # replace with path to csv file contianing pairs of img and annotations

# Create dataset
images, bboxes, labels = create_dataset_from_csv(csv_path)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Split the data
train_images, test_images, train_bboxes, test_bboxes, train_labels, test_labels = train_test_split(
    images, bboxes, labels_encoded, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Transfer learning using VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
base_model.trainable = False  # Freeze the base model

# Model definition for bounding box regression
input_layer = base_model.input
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_bboxes = Dense(4, activation='sigmoid')(x)

bbox_model = Model(inputs=input_layer, outputs=output_bboxes)
bbox_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Training the bounding box regression model
bbox_model.fit(
    datagen.flow(train_images, train_bboxes, batch_size=4),
    validation_data=(test_images, test_bboxes),
    epochs=50,
    callbacks=[early_stopping, reduce_lr]
)

bbox_model.save('bbox_model.keras') # not as effective as the chop detection model, disregard

# Load the better classification model
classification_model = load_model('my_trained_model.h5')
