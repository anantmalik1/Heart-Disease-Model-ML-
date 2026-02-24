import streamlit as st
import pandas as pd
import joblib

# -------------------- LOAD FILES --------------------

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")

# Load columns correctly and ensure it's a list
expected_columns = joblib.load("columns.pkl")

# Safety conversion (handles numpy / pandas index / method issues)
if callable(expected_columns):
    expected_columns = expected_columns()

expected_columns = list(expected_columns)

# -------------------- UI --------------------

st.title("❤️ Heart Disease Model ML")
st.markdown("### Provide the following details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# -------------------- PREDICTION --------------------

if st.button("Predict"):

    # Base numeric features
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
    }

    # One-hot encoded features
    raw_input["Sex_" + sex] = 1
    raw_input["ChestPainType_" + chest_pain] = 1
    raw_input["RestingECG_" + resting_ecg] = 1
    raw_input["ExerciseAngina_" + exercise_angina] = 1
    raw_input["ST_Slope_" + st_slope] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure correct column order
    input_df = input_df[expected_columns]

    # Scale
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # Output
    if prediction == 1:
        st.error("⚠ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
