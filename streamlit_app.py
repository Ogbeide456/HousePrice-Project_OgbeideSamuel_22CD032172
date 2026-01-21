# streamlit_app.py
import streamlit as st
from pathlib import Path
import joblib
import tempfile
import pandas as pd

# Compute model path relative to this file
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "house_price_model.pkl"

st.title("House Price Prediction")

@st.cache_resource
def load_model_safe(path: Path):
    """Load model with joblib and return it or raise."""
    return joblib.load(path)

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
            # Save uploaded bytes to a temporary file and load with joblib
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
            tmp.write(uploaded.getbuffer())
            tmp.flush()
            tmp.close()
            # load model
            model = load_model_safe(tmp.name)
            # Ensure model directory exists and persist model so future runs find it
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            st.success("Model uploaded and saved to model/house_price_model.pkl")
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")

# If model still None, stop further UI (we cannot predict)
if model is None:
    st.stop()

# --- Example simple input form (adapt to your real features) ---
st.sidebar.header("Input features")
OverallQual = st.sidebar.number_input("OverallQual (1-10)", min_value=1, max_value=10, value=7, step=1)
GrLivArea = st.sidebar.number_input("GrLivArea (sqft)", min_value=100, max_value=10000, value=1500)
TotalBsmtSF = st.sidebar.number_input("TotalBsmtSF (sqft)", min_value=0, max_value=10000, value=800)
GarageCars = st.sidebar.number_input("GarageCars", min_value=0, max_value=10, value=2)
YearBuilt = st.sidebar.number_input("YearBuilt", min_value=1800, max_value=2050, value=2005)
Neighborhood = st.sidebar.text_input("Neighborhood", value="NAmes")

# Build DataFrame with same column order as training
X_input = pd.DataFrame([{
    "OverallQual": OverallQual,
    "GrLivArea": GrLivArea,
    "TotalBsmtSF": TotalBsmtSF,
    "GarageCars": GarageCars,
    "YearBuilt": YearBuilt,
    "Neighborhood": Neighborhood
}])

if st.button("Predict"):
    try:
        pred = model.predict(X_input)
        price = float(pred[0])
        st.metric("Predicted sale price", f"${price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
