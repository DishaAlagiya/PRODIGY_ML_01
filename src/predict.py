import pandas as pd
import joblib
import os

def load_test_data(test_path):
    df = pd.read_csv(test_path)
    ids = df["Id"]
    features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
    X_test = df[features]
    return X_test, ids

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def make_predictions(model, X_test):
    preds = model.predict(X_test)
    preds = [max(0, p) for p in preds]
    return preds

def save_predictions(ids, preds, output_path):
    result_df = pd.DataFrame({
        "Id": ids,
        "SalePrice": preds
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    test_path = "processed/test_preprocessed.csv"
    model_path = "models/linear_regression_model.pkl"
    output_path = "predictions/predictions.csv"

    X_test, ids = load_test_data(test_path)
    model = load_model(model_path)
    preds = make_predictions(model, X_test)
    save_predictions(ids, preds, output_path)
