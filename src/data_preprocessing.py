# src/data_preprocessing.py

import pandas as pd
import os

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df):
    features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
    train_df = train_df[features + ["SalePrice"]]
    test_df = test_df[features + ["Id"]]

    return train_df, test_df

def save_processed(train_df, test_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train_preprocessed.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_preprocessed.csv'), index=False)
    print(f"Preprocessed files saved in {output_dir}")

if __name__ == "__main__":
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    output_dir = "processed"

    train_df, test_df = load_data(train_path, test_path)
    train_df, test_df = preprocess_data(train_df, test_df)
    save_processed(train_df, test_df, output_dir)
