import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

predictions_path = "predictions/predictions.csv"
test_path = "processed/test_preprocessed.csv"

pred_df = pd.read_csv(predictions_path)
test_df = pd.read_csv(test_path)

merged_df = pd.merge(test_df, pred_df, on="Id")

plt.figure(figsize=(8, 6))
sns.regplot(x=merged_df["GrLivArea"], y=merged_df["SalePrice"], scatter_kws={'alpha':0.5})
plt.title("GrLivArea vs Predicted Sale Price")
plt.xlabel("Above Ground Living Area (sq ft)")
plt.ylabel("Predicted Sale Price")
plt.tight_layout()
plt.show()
