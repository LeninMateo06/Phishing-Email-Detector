import pandas as pd

data = pd.read_csv("kaggle/phishing_legit_dataset_KD_10000.csv")

print(data.head())
print(data["label"].value_counts())

print(data.shape)
print(data.isnull().sum())

data = data[["text", "label"]]

print(data.head())