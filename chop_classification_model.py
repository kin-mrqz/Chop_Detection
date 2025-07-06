import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np

# Load the data from the CSV file
data = pd.read_csv("annotations.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data[['XMin', 'YMin', 'XMax', 'YMax', 'ImagePath']],
    data['Label'],
    test_size=0.2,
    random_state=42
)

# Preprocess the data for the CNN model
X_train_images = []
X_test_images = []

target_size = (64, 64)

for image_path in X_train['ImagePath']:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    if h > w:
        new_h = target_size[0]
        new_w = int(w * (target_size[0] / h))
    else:
        new_w = target_size[1]
        new_h = int(h * (target_size[1] / w))
    img = cv2.resize(img, (new_w, new_h))
    pad_h = target_size[0] - new_h
    pad_w = target_size[1] - new_w
    img = cv2.copyMakeBorder(img, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2, cv2.BORDER_CONSTANT, value=0)
    X_train_images.append(img)

for image_path in X_test['ImagePath']:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    if h > w:
        new_h = target_size[0]
        new_w = int(w * (target_size[0] / h))
    else:
        new_w = target_size[1]
        new_h = int(h * (target_size[1] / w))
    img = cv2.resize(img, (new_w, new_h))
    pad_h = target_size[0] - new_h
    pad_w = target_size[1] - new_w
    img = cv2.copyMakeBorder(img, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2, cv2.BORDER_CONSTANT, value=0)
    X_test_images.append(img)

X_train = np.array(X_train_images).reshape(-1, 64, 64, 1)
X_test = np.array(X_test_images).reshape(-1, 64, 64, 1)

# Encode the target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert the encoded labels to categorical format
print(np.unique(y_train_encoded))
print(np.unique(y_test_encoded))
y_train_categorical = to_categorical(y_train_encoded, num_classes=6)
y_test_categorical = to_categorical(y_test_encoded, num_classes=6)

print("X_train shape:", X_train.shape)
print("y_train_categorical shape:", y_train_categorical.shape)
print("X_test shape:", X_test.shape)
print("y_test_categorical shape:", y_test_categorical.shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))  # Change this line to have 5 output units

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, validation_data=(X_test, y_test_categorical))

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test_categorical)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')

# Save the model to HDF5 file
model.save('chop_classification_model.h5')  # This will save the model in the current directory

model.summary()
