# MNIST Digit Recognition with Custom Image Prediction

This project demonstrates how to build a neural network using TensorFlow and Keras to recognize handwritten digits from the MNIST dataset. Additionally, the model can predict digits from custom images created with tools like MS Paint.

---

## Features
- Train a neural network on the MNIST dataset.
- Predict handwritten digits from user-provided images.
- Visualize predictions alongside processed input images.

---

## Requirements
 Software
- **Python 3.7 or higher**
- **Internet connection** (for downloading the MNIST dataset)

### Libraries
Ensure the following Python libraries are installed:
- **`numpy`**
- **`matplotlib`**
- **`tensorflow`**
- **`pillow`**

---

## Installation

1. **Install Python**  
   Download and install Python from the [official website](https://www.python.org/).

2. **Install Dependencies**  
   Open a terminal or command prompt and run:
   **`pip install numpy matplotlib tensorflow pillow`**

## Verify Installation
  ** `python -c "import numpy, matplotlib, tensorflow, PIL; print('All libraries are installed successfully!')"` **

### How to Run the Program in VS Code

## 1. Download the Script
Save the script as **`mnist_digit_recognition.py`** in your preferred directory.

## 2. Open in VS Code
Open VS Code and navigate to the directory containing the script.

## 3. Run the Script
Open the integrated terminal in VS Code and execute:

** `python mnist_digit_recognition.py `**

## 4. Provide a Custom Image
When prompted, enter the full path to your custom digit image (e.g., C:\Users\YourName\digit.png).

### Creating a Custom Image

## 1.Design the Image

Use MS Paint or any image editor.
Draw a digit using black on a white background.
Save the file in .png format.

## 2. File Specificatios
Ensure the image is in grayscale format.
Keep clear boundaries around the digit for optimal recognition.

### Program Output
## Training Output
The model trains for 5 epochs using the MNIST dataset and outputs accuracy and loss for each epoch.

## Prediction Output
Upon providing a valid custom image path, the program:

1. Processes the image to match the model's input requirements.
2. Predicts the digit and displays it alongside the processed image.