import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import os

def load_data(processed_train_path):
    df = pd.read_csv(processed_train_path)

    features = ["GrLivArea", "BedroomAbvGr", "FullBath" ]
    X =df[features]
    y=df["SalePrice"]

    return X,y
def train_model(X,Y):
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42)

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"RMSE value: {rmse:.2f}")
    return model

def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    processed_train_path = "processed/train_preprocessed.csv"
    model_path = "models/linear_regression_model.pkl"
    X, y = load_data(processed_train_path)
    model = train_model(X, y)
    save_model(model, model_path)