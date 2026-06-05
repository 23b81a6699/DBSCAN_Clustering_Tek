import pandas as pd
import joblib

from sklearn.metrics import silhouette_score

df = pd.read_csv("data/CreditCardDataset.csv")

df.drop("CUST_ID", axis=1, inplace=True)

df.fillna(df.median(numeric_only=True), inplace=True)

X = df

scaler = joblib.load("models/scaler.pkl")

X_scaled = scaler.transform(X)

labels = joblib.load("models/labels.pkl")

valid_clusters = len(set(labels))

if valid_clusters > 1:

    score = silhouette_score(
        X_scaled,
        labels
    )

    print(f"Silhouette Score: {score:.4f}")

else:

    print("Only one cluster found.")