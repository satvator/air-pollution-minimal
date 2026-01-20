import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split(path):
    df = pd.read_csv(path)

    X = df[["temperature", "humidity"]]
    y = df["pm25"]

    return train_test_split(X, y, test_size=0.25, random_state=42)
