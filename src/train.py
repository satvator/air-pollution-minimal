from sklearn.linear_model import LinearRegression
import joblib
import os

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def save_model(model, path="models/model.pkl"):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)
