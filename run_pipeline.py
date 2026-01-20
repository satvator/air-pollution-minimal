from src.preprocess import load_and_split
from src.train import train_model, save_model
from src.evaluate import evaluate

X_train, X_test, y_train, y_test = load_and_split("data/sample.csv")

model = train_model(X_train, y_train)
mse = evaluate(model, X_test, y_test)

save_model(model)

print("PM2.5 prediction MSE:", mse)
