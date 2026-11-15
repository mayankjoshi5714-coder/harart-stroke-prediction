import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and column order
model = joblib.load("LOGISreg_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("HEART STROKE PREDICTION MODEL ❤️")
st.markdown("Provide the following details--")

# Inputs
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("SEX", ['M', 'F'])
chestPain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood sugar > 120 mg/dl", [0, 1])
resting_ECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_heartrate = st.slider("MAX heart RATE", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST depression )", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope:", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        "Age": age,
        "RestingBP": RestingBP,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_heartrate,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chestPain: 1,
        "RestingECG_" + resting_ECG: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    # Scale numeric features
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low risk of Heart Disease")
