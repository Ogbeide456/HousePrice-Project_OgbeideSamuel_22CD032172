# streamlit_app.py
import streamlit as st
from pathlib import Path
import joblib
import tempfile
import pandas as pd
import json

# Compute model path relative to this file
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "house_price_model.pkl"
NEIGHBORHOOD_JSON = MODEL_DIR / "neighborhood_categories.json"

st.title("House Price Prediction")

@st.cache_resource
def load_model_safe(path: Path):
    """Load model with joblib and return it."""
    return joblib.load(path)

@st.cache_data
def load_neighborhoods(path: Path):
    """Load neighborhood mapping from JSON."""
    if path.exists():
        return json.load(open(path, "r"))
    return None

model = None

# Try to load from repo path
if MODEL_PATH.exists():
    try:
        model = load_model_safe(MODEL_PATH)
        st.success(f"Loaded model: {MODEL_PATH}")
    except Exception as e:
        st.error(f"Model file found but failed to load: {e}")
else:
    st.warning(f"Model not found at {MODEL_PATH}")
    st.info("You can upload a trained `house_price_model.pkl` file here to run the app.")
    uploaded = st.file_uploader("Upload house_price_model.pkl", type=["pkl", "joblib"])
    if uploaded is not None:
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
            tmp.write(uploaded.getbuffer())
            tmp.flush()
            tmp.close()
            model = load_model_safe(tmp.name)

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            st.success("Model uploaded and saved to model/house_price_model.pkl")
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")

if model is None:
    st.stop()

# --------------------
# Helper: detect required columns
# --------------------
def get_required_columns(model):
    """
    Tries to detect required columns from model (sklearn pipeline).
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # try to inspect pipeline
    try:
        # if model is pipeline
        if hasattr(model, "named_steps"):
            if "preprocessor" in model.named_steps:
                pre = model.named_steps["preprocessor"]
                # try to read from preprocessor
                if hasattr(pre, "transformers_"):
                    cols = []
                    for name, transformer, col_list in pre.transformers_:
                        if isinstance(col_list, (list, tuple)):
                            cols.extend(list(col_list))
                    if cols:
                        return cols
    except Exception:
        pass

    # fallback (manual list)
    return ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars",
            "YearBuilt", "Neighborhood", "FullBath"]

def ensure_input_columns(X_df, required_cols):
    """
    Add missing columns with default values.
    """
    for col in required_cols:
        if col not in X_df.columns:
            X_df[col] = 0
    return X_df[required_cols]

# --------------------
# Load neighborhood list (if exists)
# --------------------
neigh_map = load_neighborhoods(NEIGHBORHOOD_JSON)
if neigh_map:
    neighborhood_list = list(neigh_map.keys())
else:
    neighborhood_list = ["NAmes", "CollgCr", "OldTown", "Edwards"]  # fallback

# --------------------
# Input form
# --------------------
st.sidebar.header("Input features")

OverallQual = st.sidebar.number_input("OverallQual (1-10)", min_value=1, max_value=10, value=7, step=1)
GrLivArea = st.sidebar.number_input("GrLivArea (sqft)", min_value=100, max_value=10000, value=1500)
TotalBsmtSF = st.sidebar.number_input("TotalBsmtSF (sqft)", min_value=0, max_value=10000, value=800)
GarageCars = st.sidebar.number_input("GarageCars", min_value=0, max_value=10, value=2)
YearBuilt = st.sidebar.number_input("YearBuilt", min_value=1800, max_value=2050, value=2005)
FullBath = st.sidebar.number_input("FullBath", min_value=0, max_value=10, value=1, step=1)

Neighborhood = st.sidebar.selectbox("Neighborhood", neighborhood_list)

# Build DataFrame
X_input = pd.DataFrame([{
    "OverallQual": OverallQual,
    "GrLivArea": GrLivArea,
    "TotalBsmtSF": TotalBsmtSF,
    "GarageCars": GarageCars,
    "YearBuilt": YearBuilt,
    "Neighborhood": Neighborhood,
    "FullBath": FullBath
}])

required_cols = get_required_columns(model)
X_input = ensure_input_columns(X_input, required_cols)

# Predict
if st.button("Predict"):
    try:
        pred = model.predict(X_input)
        price = float(pred[0])
        st.metric("Predicted sale price", f"${price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
