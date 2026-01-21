# train_model.py
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# --- CONFIG ---
INPUT_CSV = "train.csv"   # change if your CSV has another name
MODEL_OUT = "house_price_model.pkl"
NEIGHBOR_JSON = "neighborhood_categories.json"

# Selected features (6 numeric + 1 categorical)
FEATURES = [
    "OverallQual", "GrLivArea", "TotalBsmtSF",
    "GarageCars", "FullBath", "YearBuilt", "Neighborhood"
]
TARGET = "SalePrice"

# --- Load ---
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Couldn't find dataset '{INPUT_CSV}'. Put train.csv in the same folder and run again.")

df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=FEATURES + [TARGET])  # simple cleanup; tune as needed

X = df[FEATURES]
y = df[TARGET]

numeric_features = [c for c in FEATURES if c != "Neighborhood"]
categorical_features = ["Neighborhood"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

print("Training model...")
pipeline.fit(X, y)
print("Training complete. Saving model...")

joblib.dump(pipeline, MODEL_OUT)
print(f"Saved model to {MODEL_OUT}")

# save neighborhood categories so the app can show dropdowns (optional)
ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
categories = ohe.categories_[0].tolist()  # Neighborhood categories
with open(NEIGHBOR_JSON, "w") as f:
    json.dump(categories, f, indent=2)
print(f"Saved neighborhood categories to {NEIGHBOR_JSON}")
