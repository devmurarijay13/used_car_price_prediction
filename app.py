import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import traceback

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
st.markdown("# 🚗 Used Car Price Prediction")

st.markdown("""
- This model predicts used car prices with **93% accuracy**.
- It uses a machine learning model trained on Car Dekho’s dataset (15,400+ rows).
""")

# --- Paths ---
MODEL_PATH = 'rf_model.joblib'
PREPROCESSOR_PATH = 'preprocessor.joblib'
DATASET_PATH = 'cardekho_dataset.csv'

# --- File Existence Check ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(DATASET_PATH):
    st.error("Required file(s) not found. Make sure model, preprocessor, and dataset files are in the app directory.")
    st.stop()

# --- Load Files with Caching ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_PATH)
    df['model'] = df['model'].str.strip()
    df['fuel_type'] = df['fuel_type'].astype(str).str.strip()
    df['transmission_type'] = df['transmission_type'].astype(str).str.strip()
    df['seller_type'] = df['seller_type'].astype(str).str.strip()
    return df

model = load_model()
preprocessor = load_preprocessor()
df = load_data()

# --- UI Form ---
with st.container():
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            model_name = st.selectbox("Car Model", sorted(df['model'].unique()))
            current_year = datetime.now().year
            manufacturing_year = st.number_input(
                "Manufacturing Year", min_value=1980, max_value=current_year,
                value=current_year - 5, step=1
            )
            vehicle_age = current_year - manufacturing_year
            km_driven = st.number_input("Kilometers Driven", 0, 500000, 50000, 1000)
            seller_type = st.selectbox("Seller Type", sorted(df['seller_type'].unique()))

        with col2:
            fuel_type = st.selectbox("Fuel Type", sorted(df['fuel_type'].unique()))
            transmission_type = st.selectbox("Transmission Type", sorted(df['transmission_type'].unique()))
            mileage = st.number_input("Mileage (kmpl)", 0.0, 50.0, 15.0, 0.1, format="%.1f")
            engine = st.number_input("Engine Capacity (CC)", 500, 8000, 1200, 100)
            seats = st.number_input("Number of Seats", 2, 10, 5, 1)

        submit_button = st.form_submit_button("Predict Price")

# --- Prediction ---
if submit_button:
    try:
        input_data = pd.DataFrame([{
            'model': model_name,
            'vehicle_age': vehicle_age,
            'km_driven': km_driven,
            'seller_type': seller_type,
            'fuel_type': fuel_type,
            'transmission_type': transmission_type,
            'mileage': mileage,
            'engine': engine,
            'seats': seats
        }])

        transformed_data = preprocessor.transform(input_data)
        predicted_price = model.predict(transformed_data)[0]

        # Format price
        price_text = f"₹{predicted_price/100000:.2f} Lakhs" if predicted_price >= 100000 else f"₹{predicted_price:,.0f}"
        st.success(f"**Predicted Price:** {price_text}")

        with st.expander("View Input Summary"):
            st.dataframe(input_data, hide_index=True)

    except Exception as e:
        st.error("Prediction failed. See error below:")
        st.text(traceback.format_exc())
