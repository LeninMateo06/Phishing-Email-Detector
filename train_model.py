import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

data = pd.read_csv("kaggle/phishing_legit_dataset_KD_10000.csv")

print(data.head())
print(data["label"].value_counts())

print(data.shape)
print(data.isnull().sum())

data = data[["text", "label"]]

print(data.head())

texts = data["text"]
labels = data["label"]

vectorizer = TextVectorization(
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=100
)

vectorizer.adapt(texts)

sample_text = "Your account has been locked"

vectorizer_text = vectorizer([sample_text])

print(vectorizer_text)