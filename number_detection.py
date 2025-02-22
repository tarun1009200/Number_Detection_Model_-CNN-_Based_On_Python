import tensorflow as tf
import numpy as np
from tkinter import Tk, Canvas, Button
from PIL import Image, ImageOps, ImageDraw

def get_mnist_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train, x_test, y_test)

def train_model(x_train, y_train, x_test, y_test):
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    return model

def preprocess_image(image):
    img = image.resize((28, 28), Image.Resampling.LANCZOS)
    img = ImageOps.invert(img)

    img_data = np.array(img)
    coords = np.column_stack(np.where(img_data > 0)) 
    if coords.any():  
        top_left = coords.min(axis=0)
        bottom_right = coords.max(axis=0)
        cropped = img_data[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
        img = Image.fromarray(cropped).resize((28, 28), Image.Resampling.LANCZOS)

    img_data = np.array(img) / 255.0
    return img_data.reshape(28, 28, 1)

def predict(model, img):
    img = np.expand_dims(img, axis=0)  
    predictions = model.predict(img)
    return np.argmax(predictions), np.max(predictions)

class DrawingApp:
    def __init__(self, model):
        self.model = model
        self.window = Tk()
        self.window.title("Digit Recognizer")
        self.canvas = Canvas(self.window, width=280, height=280, bg='black')
        self.canvas.pack()

        self.predict_button = Button(self.window, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        self.clear_button = Button(self.window, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.canvas.bind('<B1-Motion>', self.draw)
        self.image = Image.new("L", (280, 280), color=0)
        self.draw_context = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8  
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        self.draw_context.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new("L", (280, 280), color=0)
        self.draw_context = ImageDraw.Draw(self.image)

    def predict_digit(self):
        img_data = preprocess_image(self.image)
        predicted_digit, confidence = predict(self.model, img_data)
        print(f"Predicted Digit: {predicted_digit} (Confidence: {confidence:.2f})")
        self.canvas.create_text(140, 140, text=str(predicted_digit), fill='red', font=('Helvetica', 40))

    def run(self):
        self.window.mainloop()

def main():
    try:
        model = tf.keras.models.load_model('digit_recognizer_model.h5')
        print("Loaded saved model.")
    except:
        print("Training a new model...")
        x_train, y_train, x_test, y_test = get_mnist_data()
        model = train_model(x_train, y_train, x_test, y_test)
        model.save('digit_recognizer_model.h5')
        print("Model saved.")

    app = DrawingApp(model)
    app.run()

if __name__ == "__main__":
    main()