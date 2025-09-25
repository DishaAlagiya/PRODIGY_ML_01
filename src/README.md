# PRODIGY_ML_01

## House Price Prediction

A machine learning project to predict house prices using the “House Prices: Advanced Regression Techniques” dataset from Kaggle.

## Project Structure

PRODIGY_ML_01/
│
├── data/ # Dataset (download from Kaggle)
│ ├── train.csv
│ ├── test.csv
│ └── data_description.txt (or other metadata)
├── src/ # Python scripts
│ ├── data_preprocessing.py # Script to clean and preprocess data
│ ├── train_model.py # Script to train ML model
│ └── predict.py # Script to make predictions on test/new data
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## Workflow
1. Run preprocessing → creates cleaned datasets  
2. Train Linear Regression model → calculates RMSE & saves model  
3. Predict on test data → outputs predictions in CSV  

## Evaluation
- Metric used: **RMSE (Root Mean Squared Error)**
- Current RMSE: ~52,975.72 (on validation split)