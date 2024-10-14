import tkinter as tk
from tkinter import Canvas, Button, messagebox
from keras.models import load_model # type: ignore
import numpy as np
from PIL import Image, ImageDraw

# Load your trained model
model = load_model('handwritten_digit_model.h5')  # Change to your model's path

class DigitRecognizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Handwritten Digit Recognition")
        self.canvas = Canvas(self.master, width=280, height=280, bg='white')
        self.canvas.pack()

        self.button_predict = Button(self.master, text="Predict", command=self.predict)
        self.button_predict.pack()

        self.button_clear = Button(self.master, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.image = Image.new("L", (280, 280), 255)  # L mode for grayscale
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='black', outline='black')
        self.draw.ellipse((x-5, y-5, x+5, y+5), fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        # Resize and preprocess the image
        img = self.image.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        # Show the prediction result
        messagebox.showinfo("Prediction", f"The predicted digit is: {predicted_digit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()
