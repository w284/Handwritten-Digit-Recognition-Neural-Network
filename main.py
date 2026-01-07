import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Correct normalization and reshape
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build CNN model
model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[early_stop])

model.save('improved_handwritten.keras')

# Load THE CORRECT MODEL (ensure you're loading the improved CNN model)
model = tf.keras.models.load_model('improved_handwritten.keras')  # NOT 'handwritten.keras'

image_number = 0
while os.path.isfile(f"digits/{image_number}.png"):
    try:
        # 1. Load image PROPERLY
        img = cv2.imread(f"digits/{image_number}.png", cv2.IMREAD_GRAYSCALE)  # Force grayscale

        # 2. Resize if needed (MNIST requires 28x28)
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28))

        # 3. Invert AND Normalize (critical!)
        img = np.invert(img)  # MNIST-style white-on-black
        img = img.astype('float32') / 255.0  # Scale to 0-1

        # 4. Reshape for CNN input (add batch + channel dimensions)
        img = img.reshape(1, 28, 28, 1)  # Shape = (1,28,28,1)

        # 5. Make prediction
        prediction = model.predict(img)
        print(f"Digit probabilities: {prediction}")
        print(f"Predicted digit: {np.argmax(prediction)}")

        # Visualize EXACTLY what the model sees
        plt.imshow(img[0, :, :, 0], cmap=plt.cm.binary)  # Remove channel for display
        plt.show()

    except Exception as e:
        print(f"Error processing image {image_number}: {str(e)}")
    finally:
        image_number += 1