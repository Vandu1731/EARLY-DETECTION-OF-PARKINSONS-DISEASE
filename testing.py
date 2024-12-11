
import numpy as np
import joblib
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Load the trained model
dt_model = joblib.load('decision_tree_model.pkl')

# Create the main window
root = tk.Tk()
root.title("Parkinson's Disease Prediction")
root.geometry("800x500")  # Adjust window size as needed

# Load and display the background image
try:
    bg_image = Image.open("Banner.jpg")  # Replace with your image file path
    bg_image = bg_image.resize((1200, 900), Image.Resampling.LANCZOS)  # Resize to fit the window
    bg_photo = ImageTk.PhotoImage(bg_image)

    # Place the background image
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
except FileNotFoundError:
    messagebox.showerror("Error", "Background image file 'Banner.jpg' not found.")

# Labels and input fields directly over the image
labels = ["MDVP:Fo(Hz):", "MDVP:Fhi(Hz):", "MDVP:Flo(Hz):",
          "MDVP:Jitter(%):", "MDVP:RAP:", "RPDE:"]
entries = []

# Place labels and entries with absolute positioning
x_start, y_start = 50, 300  
x_gap, y_gap = 150, 40     

for i, text in enumerate(labels):
    label = tk.Label(root, text=text, anchor="w", bg="lightblue", font=("Arial", 10, "bold"))
    label.place(x=x_start, y=y_start + i * y_gap)

    entry = tk.Entry(root)
    entry.place(x=x_start + x_gap, y=y_start + i * y_gap, width=150)
    entries.append(entry)

# Function to make predictions
def make_prediction():
    try:
        # Get and validate input
        inputs = [entry.get() for entry in entries]
        if any(not x.strip() for x in inputs):
            raise ValueError("All fields are required.")
        
        # Convert inputs to floats
        new_data = np.array([[float(value) for value in inputs]])

        # Predict using the model
        new_prediction = dt_model.predict(new_data)
        predicted_status = "Parkinson's Positive" if new_prediction[0] == 1 else "Parkinson's Negative"

        # Update the result label
        label_result.config(text=f"Prediction: {predicted_status}", fg="green")
    except ValueError as ve:
        messagebox.showerror("Invalid Input", str(ve))
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# Prediction button
predict_button = tk.Button(root, text="Predict", command=make_prediction, bg="blue", fg="white", font=("Arial", 12))
predict_button.place(x=50, y=y_start + len(labels) * y_gap + 20, width=100, height=30)

# Label to display the result
label_result = tk.Label(root, text="Prediction: ", fg="black", font=("Arial", 12, "bold"), bg="lightblue")
label_result.place(x=200, y=y_start + len(labels) * y_gap + 20)

# Run the application
root.mainloop()
