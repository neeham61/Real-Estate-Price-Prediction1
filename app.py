"""Real Estate Price Prediction - Streamlit web app.
Loads the saved pipeline (best_model.joblib) created by the Colab notebook.
The model is NOT trained here; it is only loaded and used for prediction.
"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = "best_model.joblib"
INFO_FILE = "best_model_info.json"

# Same feature names and ORDER as in the training notebook
FEATURES = [
    "X1 transaction date",
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude",
]


@st.cache_resource
def load_model():
    """Load the fitted pipeline once (not on every click)."""
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)


def load_info():
    """Optional info file written by the notebook (model name, training ranges)."""
    if os.path.exists(INFO_FILE):
        with open(INFO_FILE) as f:
            return json.load(f)
    return {}


def build_input(values):
    """Create a one-row table with the exact feature order used in training."""
    return pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)


def check_inputs(values, ranges):
    """Return (errors, warnings). Errors block prediction; warnings do not."""
    errors, warnings = [], []
    if values["X2 house age"] < 0:
        errors.append("House age cannot be negative.")
    if values["X3 distance to the nearest MRT station"] < 0:
        errors.append("Distance cannot be negative.")
    if values["X4 number of convenience stores"] < 0:
        errors.append("Number of stores cannot be negative.")
    if not -90 <= values["X5 latitude"] <= 90:
        errors.append("Latitude must be between -90 and 90.")
    if not -180 <= values["X6 longitude"] <= 180:
        errors.append("Longitude must be between -180 and 180.")
    for f in FEATURES:
        if f in ranges:
            lo, hi = ranges[f]
            if not lo <= values[f] <= hi:
                warnings.append(f"{f} = {values[f]} is outside the range seen in training data "
                                f"({lo:g} to {hi:g}); the estimate may be unreliable.")
    return errors, warnings


st.set_page_config(page_title="Real Estate Price Prediction", page_icon="🏠")
st.title("Real Estate Price Prediction")
st.write(
    "This app uses a machine learning model trained on the Kaggle *Real Estate Price Prediction* dataset "
    "to estimate the **house price of unit area** from six property features."
)

model = load_model()
info = load_info()
if model is None:
    st.error(f"Model file '{MODEL_FILE}' was not found. Run the Colab notebook, then put "
             f"'{MODEL_FILE}' in the same folder as app.py (and in the GitHub repository).")
    st.stop()

if info:
    st.caption(f"Model: {info.get('best_model', 'unknown')} | 10-fold CV RMSE: {info.get('cv_mean_rmse', float('nan')):.3f}")
ranges = info.get("feature_ranges", {})


def default(feature, fallback):
    if feature in ranges:
        lo, hi = ranges[feature]
        return float((lo + hi) / 2)
    return fallback


st.subheader("Enter property details")
values = {
    "X1 transaction date": st.number_input("X1 Transaction date (e.g. 2013.5 = mid 2013)", value=default("X1 transaction date", 2013.0), format="%.3f"),
    "X2 house age": st.number_input("X2 House age (years)", min_value=0.0, value=default("X2 house age", 15.0)),
    "X3 distance to the nearest MRT station": st.number_input("X3 Distance to nearest MRT station (metres)", min_value=0.0, value=default("X3 distance to the nearest MRT station", 500.0)),
    "X4 number of convenience stores": st.number_input("X4 Number of convenience stores nearby", min_value=0, step=1, value=int(round(default("X4 number of convenience stores", 4)))),
    "X5 latitude": st.number_input("X5 Latitude", value=default("X5 latitude", 24.97), format="%.5f"),
    "X6 longitude": st.number_input("X6 Longitude", value=default("X6 longitude", 121.54), format="%.5f"),
}

if st.button("Predict"):
    errors, warns = check_inputs(values, ranges)
    for e in errors:
        st.error(e)
    if not errors:
        for w in warns:
            st.warning(w)
        try:
            prediction = float(model.predict(build_input(values))[0])
            st.success(f"Estimated house price of unit area: **{prediction:.2f}**")
        except Exception as exc:  # show a readable message instead of a crash
            st.error(f"Prediction failed: {exc}")

st.info("This is only a statistical estimate from a small dataset (one region). "
        "It is not a guaranteed market price.")
