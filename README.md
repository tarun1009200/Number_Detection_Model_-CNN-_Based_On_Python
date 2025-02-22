#Number Detection Using Deep Learning

#Introduction
This project is a deep-learning-based number detection system that uses a Convolutional Neural Network (CNN) trained on the MNIST dataset. It provides a graphical user interface (GUI) that allows users to draw digits, which the trained model then recognizes in real-time.

#Features
- Uses a CNN model for digit recognition
- Trains on the MNIST dataset
- GUI for user-drawn digit recognition
- Saves and loads trained models to avoid retraining
- Preprocessing of drawn images for accurate predictions

#Requirements
To run this project, you need to install the following dependencies:
- Python 3.x
- TensorFlow
- NumPy
- Pillow
- Tkinter

Install dependencies using:
```sh
pip install tensorflow numpy pillow
```

# How It Works
# Model Training
1. Loading the MNIST Dataset:
   - Uses `tf.keras.datasets.mnist` to load handwritten digit data.
   - Splits into training and testing sets.

2. **Data Preprocessing:**
   - Normalizes pixel values between 0 and 1.
   - Reshapes images to (28, 28, 1) for CNN input.

3. **CNN Model Architecture:**
   - **Conv2D Layer 1:** 32 filters, 3x3 kernel, ReLU activation
   - **MaxPooling2D Layer 1:** 2x2 pooling to reduce dimensions
   - **Conv2D Layer 2:** 64 filters, 3x3 kernel, ReLU activation
   - **MaxPooling2D Layer 2:** 2x2 pooling
   - **Flatten Layer:** Converts 2D data into a 1D vector
   - **Dense Layer:** 128 neurons with ReLU activation
   - **Dropout Layer:** Prevents overfitting by randomly disabling 50% of neurons
   - **Output Layer:** 10 neurons (one for each digit 0-9) with softmax activation

4. **Training Process:**
   - Uses Adam optimizer
   - Categorical cross-entropy loss function
   - Trains for 5 epochs with validation on test data
   - Saves the trained model as `digit_recognizer_model.h5`

### Digit Recognition GUI
1. **User Interface:**
   - Built with Tkinter
   - Canvas for drawing digits
   - Buttons for clearing canvas and predicting digits

2. **Drawing Mechanism:**
   - Users draw a digit on a black background
   - White strokes (ellipses) are drawn as the user moves the mouse

3. **Image Preprocessing:**
   - Converts the drawn image into grayscale (L mode)
   - Resizes it to 28x28 pixels
   - Inverts colors so digits are black on white
   - Normalizes pixel values and reshapes to (28, 28, 1)

4. **Prediction Mechanism:**
   - Passes the processed image to the trained CNN model
   - Uses `np.argmax(predictions)` to determine the most probable digit
   - Displays predicted digit and confidence score on the canvas

## Code Structure
- `get_mnist_data()`: Loads and returns the MNIST dataset
- `train_model()`: Defines, compiles, and trains the CNN model
- `preprocess_image()`: Prepares drawn images for model input
- `predict()`: Predicts the digit using the trained model
- `DrawingApp`: GUI class for digit drawing and prediction
- `main()`: Loads or trains the model and starts the GUI

## How to Run
1. **Run the script:**
   ```sh
   python number_detection.py
   ```
2. **Use the GUI:**
   - Draw a digit on the canvas
   - Click **Predict** to see the result
   - Click **Clear** to reset the canvas

## Future Improvements
- **Enhancing Accuracy:** Train on an augmented dataset to handle variations in handwriting.
- **Adding More Features:** Include additional preprocessing techniques like noise reduction and edge detection.
- **Expanding Model Usage:** Deploy as a web app using Flask or FastAPI.
- **Improving GUI:** Provide a smoother drawing experience with better rendering.

## Conclusion
This project provides a basic yet functional digit recognition system using deep learning. It combines a robust CNN model with an interactive GUI for easy testing and visualization of digit classification results. 

