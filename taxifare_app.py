import streamlit as st
import numpy as np
import joblib
model = joblib.load("fare_model.pkl")
scaler = joblib.load("fare_scaler.pkl")
service_encoder = joblib.load("service_encoder.pkl")
payment_encoder = joblib.load("payment_encoder.pkl")
st.set_page_config(
    page_title="FareSense - Smart Taxi Fare Predictor",
    page_icon="🚕",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.markdown(""" 
    <style> 
        .title { 
            text-align: center; 
            font-size: 38px; 
            color: #2A7AE2; 
            font-weight: 800; 
            margin-bottom: 5px; 
        } 
        .subtitle { 
            text-align: center; 
            font-size: 18px; 
            color: #4B4B4B; 
            margin-bottom: 30px; 
        } 
        .stButton>button { 
            width: 100%; 
            background-color: #2A7AE2; 
            color: white; 
            font-size: 18px; 
            padding: 12px; 
            border-radius: 10px; 
        } 
        .stButton>button:hover { 
            background-color: #1F5BB5; 
            color: white; 
        } 
        .card { 
            padding: 20px; 
            border-radius: 15px; 
            background-color: #F0F6FF; 
            box-shadow: 1px 1px 10px #d1d1d1; 
        } 
        .footer { 
            text-align: center; 
            color: #888888; 
            font-size: 14px; 
            margin-top: 30px; 
        } 
    </style> 
""", unsafe_allow_html=True) 


st.markdown('<div class="title">🚕 FareSense</div>', unsafe_allow_html=True) 
st.markdown('<div class="subtitle">Smart Taxi Fare Predictor</div>', unsafe_allow_html=True) 

st.markdown('<div class="card">', unsafe_allow_html=True) 

col1, col2 = st.columns(2) 
with col1: 
    distance = st.number_input("Distance (km)", 0.0, 100.0, step=0.5) 
    service = st.selectbox("Service Type", list(service_encoder.classes_)) 
with col2: 
    duration = st.number_input("Duration (minutes)", 0, 200, step=1) 
    payment = st.selectbox("Payment Method", list(payment_encoder.classes_)) 

st.markdown('</div>', unsafe_allow_html=True) 
st.write("") 

predict_btn = st.button("Predict Fare") 

if predict_btn: 
    service_enc = service_encoder.transform([service])[0] 
    payment_enc = payment_encoder.transform([payment])[0] 

    input_data = np.array([[distance, duration, service_enc, payment_enc]]) 
    input_scaled = scaler.transform(input_data) 
    predicted_fare = model.predict(input_scaled)[0] 
    predicted_fare = max(predicted_fare, 0)  # fares can't be negative 

    st.success(f"Estimated fare: **₹{predicted_fare:.2f}**") 

st.write("---") 
st.markdown( 
    "<p class='footer'>© 2026 FareSense - All Rights Reserved.</p>", 
    unsafe_allow_html=True, 
)
