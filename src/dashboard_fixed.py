import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Prediction on PM2.5 Levels",
    page_icon=":bar_chart:",
    layout="centered"
)

st.title("PM2.5 Level Prediction")
st.write("Enter the relevant features to predict PM2.5 levels.")


@st.cache_resource
def load_model():
    model_path = Path(__file__).with_name("linear_regression_model_pm2_5.pkl")

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    return joblib.load(model_path)


model = load_model()

# These names must match the feature names stored in the pickle.
# The model expects:
#   1. Rainfall (mm)
#   2. Pm10 （ug/m³）

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=1800.0,
    value=20.0,
    step=0.1
)

pm10 = st.number_input(
    "PM10 (ug/m³)",
    min_value=0.0,
    max_value=100.0,
    value=3.0,
    step=0.1
)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "Rainfall (mm)": [rainfall],
        "Pm10 （ug/m³）": [pm10]
    })

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.metric(
            "Predicted PM2.5 Level",
            f"{prediction:.2f} μg/m³"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
