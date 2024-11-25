import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageOps

# Step 1: Load and preprocess MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Step 2: Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)

# Step 3: Predict custom MS Paint image
image_path = input("Enter the path to your custom image: ").strip()

# Step 4: Validate the file path
if not os.path.exists(image_path):
    print("Error: Invalid file path. Please check and try again.")
else:
    try:
        # Load the image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = ImageOps.invert(img)  # Invert colors to match MNIST format
        
        # Remove extra white background (optional cleanup)
        img_array = np.array(img)
        img_array = img_array - np.min(img_array)  # Normalize to 0-255
        img_array = (img_array / np.max(img_array)) * 255
        
        img = Image.fromarray(np.uint8(img_array))  # Convert back to image
        
        # Resize to 28x28 (replace ANTIALIAS with Resampling.LANCZOS)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Normalize and reshape the image
        img_array = np.array(img) / 255.0  # Normalize to 0-1
        img_array = img_array.reshape(1, 28, 28)
        
        # Predict the digit
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        
        # Display the prediction
        print(f"Predicted digit: {predicted_digit}")
        plt.imshow(img, cmap='gray')
        plt.title(f"Predicted Digit: {predicted_digit}")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")



