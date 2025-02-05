import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout
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
    print(data.columns)  # Check the column names
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

# Paths
csv_path = r"C:\Users\User\Desktop\Uni\Career\KyAssignment\annotations.csv"

# Create dataset
images, bboxes, labels = create_dataset_from_csv(csv_path)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Split the data
train_images, test_images, train_bboxes, test_bboxes, train_labels, test_labels = train_test_split(
    images, bboxes, labels_encoded, test_size=0.2, random_state=42)

# Model definition
input_layer = Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_bboxes = Dense(4, activation='sigmoid')(x)
output_classes = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=[output_bboxes, output_classes])
model.compile(optimizer='adam', loss=['mse', 'sparse_categorical_crossentropy'], metrics=[['accuracy'], ['accuracy']])

# Training
model.fit(train_images, [train_bboxes, train_labels],
          validation_data=(test_images, [test_bboxes, test_labels]),
          epochs=10, batch_size=2)

# Save the model
model.save('chop_detection_model.h5')
