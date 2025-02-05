import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout
from tensorflow.keras.optimizers import Adam

# Load the saved model
bbox_model_path = r"C:\Users\User\Desktop\Uni\Career\KyAssignment\chop-detection-system\uploads\bbox_model.h5"

# Define custom objects if necessary (e.g., custom loss function)
custom_objects = {
    'adam': Adam,
    'mse': tf.keras.losses.MeanSquaredError()
}

# Load the model with custom objects
bbox_model = load_model(bbox_model_path, custom_objects=custom_objects)

# Compile the model if needed
bbox_model.compile(optimizer='adam', loss='mse')  # Adjust optimizer and loss as needed

# Now bbox_model is ready to be used for inference or further processing

