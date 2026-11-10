
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense, Dropout
import tkinter as tk


# Load the vocabulary
with open("vocabulary.txt", "r", encoding="utf-8") as file:
    vocabulary = [line.strip() for line in file]

# Create the text vectorizer from vocabulary
vectorizer = TextVectorization(
      max_tokens=10000,
      output_mode="int",
      output_sequence_length=100,
)

vectorizer.set_vocabulary(vocabulary[2:])

# Use the neural network architecture use during training model
model = Sequential([
      vectorizer,
      Embedding(input_dim=10000, output_dim=16),
      GlobalAveragePooling1D(),
      Dense(16, activation="relu"),
      Dropout(0.3),
      Dense(1, activation="sigmoid"),
  ])

# Build the model and load the trained model
dummy_email = tf.convert_to_tensor(["test email"], dtype=tf.string)
model(dummy_email)

model.load_weights("phishing_model.weights.h5")

# Clean as we did in training
def clean_email(text):
    lines = text.split("\n")
    clean_lines = []

    for line in lines:
        if line.startswith("Keywords:"):
            continue

        if line.startswith("Subject:"):
            line = line.replace("Subject:", "", 1)

        clean_lines.append(line)

    return "\n".join(clean_lines)

#Get the email put in the interface
def check_email():
    email = email_box.get("1.0", tk.END).strip()

    if email == "":
        result_label.config(text="Please paste an email first")
        score_label.config(text="Prediction score:")
        return

    email = clean_email(email)

    email_tensor = tf.convert_to_tensor([email], dtype=tf.string)
    prediction = model.predict(email_tensor, verbose=0)

    score = prediction[0][0]

    # Classification threshold
    # Score >= 0.5 = Phishing, score < 0.5 = Legitimate
    if score >= 0.5:
        result_label.config(text="jajaja, hackers think we are dumb, Phishing Detected!!! ")
    else:
        result_label.config(text="You are safe brother, Not Phishing Detected")

    score_label.config(text=f"Prediction score: {score:.4f}")

#User interface for DEMO
window = tk.Tk()
window.title("Phishing Email Detector")
window.geometry("600x500")

title_label = tk.Label(
    window,
    text="Phishing Email Detector",
    font=("Georgia", 18)
)
title_label.pack(pady=20)

instruction_label = tk.Label(
    window,
    text="Don't let hackers trick you, paste your email: ",
    font=("Courier New", 10)
)
instruction_label.pack(anchor="w", padx=40)

email_box = tk.Text(
    window,
    height=15,
    width=65
)
email_box.pack(pady=(10,0))

check_button = tk.Button(
    window,
    text="Check Email",
    command=check_email,
    font=("Courier New", 10)
)
check_button.pack(anchor="e", padx=50, pady=(5,0))

result_label = tk.Label(
    window,
    text="Result:",
    font=("Georgia", 8, "bold")
)
result_label.pack(anchor="w", padx=50, pady=(0, 5))

score_label = tk.Label(
    window,
    text="Prediction score:"
)
score_label.pack(anchor="w", padx=50)

window.mainloop()