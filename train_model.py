import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

data = pd.read_csv("kaggle/phishing_legit_dataset_KD_10000.csv")

print(data.head())
print(data["label"].value_counts())

print(data.shape)
print(data.isnull().sum())

data = data[["text", "label"]]

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

split = int(len(data) * 0.8)

train_data = data[:split]
val_data = data[split:]

train_texts = train_data["text"]
train_labels = train_data["label"]

val_texts = val_data["text"]
val_labels = val_data["label"]

print("Training emails: ", len(train_data))
print("Validation emails: ", len(val_data))

print(data.head())


vectorizer = TextVectorization(
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=100
)

vectorizer.adapt(train_texts)

sample_text = "Your account has been locked"

vectorizer_text = vectorizer([sample_text])

print(vectorizer_text)