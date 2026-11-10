import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout

#Load and clean the dataset (Features and labels)
data = pd.read_csv("kaggle/phishing_legit_dataset_KD_10000.csv")

def clean_emails(text):
    lines = text.split("\n")
    clean_lines = []

    for line in lines:
        if line.startswith("Keywords:"):
            continue

        if line.startswith("Subject:"):
            line = line.replace("Subject:", "", 1)

        clean_lines.append(line)

    return "\n".join(clean_lines)

#Clean the email
data["text"] = data["text"].apply(clean_emails)


data = data[["text", "label"]]

#Split the data to training and confirming sets (Training and Evaluation)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

split = int(len(data) * 0.8)

train_data = data[:split]
val_data = data[split:]

train_texts = train_data["text"]
train_labels = train_data["label"]

val_texts = val_data["text"]
val_labels = val_data["label"]

#Convert the data into TensorFlow tensors
train_texts = tf.convert_to_tensor(train_texts.to_numpy(), dtype=tf.string)
train_labels = tf.convert_to_tensor(train_labels.to_numpy(), dtype=tf.float32)

val_texts = tf.convert_to_tensor(val_texts.to_numpy(), dtype=tf.string)
val_labels = tf.convert_to_tensor(val_labels.to_numpy(), dtype=tf.float32)

print("Training emails: ", len(train_data))
print("Validation emails: ", len(val_data))


#Convert email to numerical tokens (Numerical Data and text Representation)
vectorizer = TextVectorization(
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=100
)

vectorizer.adapt(train_texts)

vocabulary = vectorizer.get_vocabulary()

with open("vocabulary.txt", "w", encoding="utf-8") as file:
    for word in vocabulary:
        file.write(word + "\n")

#Build the neural network with Embeddings
#Here we use embeddings to convert words into numerical representations to make the neural network
#Learn from it.
model = Sequential([
    vectorizer,
    Embedding(input_dim=10000, output_dim=16),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
    Dropout(0.3),                                    #Reduce overfitting
    Dense(1, activation="sigmoid")  #Output probability between 0 and 1
])

#Compile the classification model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

#Train the neural network

model.fit(
    train_texts,
    train_labels,
    validation_data=(val_texts, val_labels),
    epochs=3
)

test_emails = [
    "Your bank account has been suspended. Click the link below immediately to verify your password.",
    "Hi Castia, the meeting has been moved to 2 PM tomorrow. Let me know if that works for you. Thanks, Lenin."
]

test_emails_tensor = tf.convert_to_tensor(test_emails, dtype=tf.string)
predictions = model.predict(test_emails_tensor, verbose=0)

for email, prediction in zip(test_emails, predictions):
    print("\nEmail:")
    print(email)

    print("Prediction score:", prediction[0])

    # 0.5 classification threshold
    # The sigmoid gives score between 0 and.
    # Then the threshold o 0.5 to identify phishing or legitimate.
    if prediction[0] >= 0.5:
        print("Result: Phishing")
    else:
        print("Result: Legitimate")


model.save_weights("phishing_model.weights.h5")
