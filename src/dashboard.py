import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Prediction on PM2.5 Levels",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 PM2.5 Level Prediction")
st.write("Enter the relevant features to predict PM2.5 levels.")

@st.cache_resource
def load_model():
    return joblib.load("linear_regression_model.pkl")

model = load_model()

attendance = st.number_input(
    "",
    min_value=0.0,
    max_value=100.0,
    value=85.0,
    step=0.1
)

study_hours = st.number_input(
    "",
    min_value=0.0,
    value=10.0,
    step=0.1
)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "": [attendance],
        "": [study_hours]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.metric("Predicted PM2.5 Level", f"{prediction:.2f}")
