import os
import joblib
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/CreditCardDataset.csv")

df.drop("CUST_ID", axis=1, inplace=True)

df.fillna(df.median(numeric_only=True), inplace=True)

X = df

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(
    eps=1.5,
    min_samples=10
)

labels = dbscan.fit_predict(X_scaled)

joblib.dump(dbscan, "models/dbscan.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(labels, "models/labels.pkl")

print("✅ DBSCAN Model Saved")
print("✅ Scaler Saved")
print("✅ Labels Saved")