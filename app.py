import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Heart Disease Model ML", layout="wide")

# ------------------ BACKGROUND ------------------
def add_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1588776814546-b0d1f9a9f0b7");
            background-size: cover;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg()

# ------------------ LOAD ANIMATION ------------------
def load_lottie(url):
    r = requests.get(url)
    return r.json()

lottie_heart = load_lottie(
    "https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"
)

# ------------------ LOAD MODEL FILES ------------------
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

if callable(expected_columns):
    expected_columns = expected_columns()

expected_columns = list(expected_columns)

# ------------------ HEADER ------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.title("❤️ Heart Disease Model ML")
    st.markdown("### Advanced AI Risk Prediction Dashboard")

with col2:
    st_lottie(lottie_heart, height=200)

st.markdown("---")

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("Patient Details")

age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.sidebar.number_input("Resting BP", 80, 200, 120)
cholesterol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1])
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ------------------ PREDICTION ------------------
if st.sidebar.button("Predict Risk"):

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
    }

    raw_input["Sex_" + sex] = 1
    raw_input["ChestPainType_" + chest_pain] = 1
    raw_input["RestingECG_" + resting_ecg] = 1
    raw_input["ExerciseAngina_" + exercise_angina] = 1
    raw_input["ST_Slope_" + st_slope] = 1

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    risk_score = model.predict_proba(scaled_input)[0][1] * 100

    st.markdown("## 🔍 Prediction Result")

    colA, colB = st.columns(2)

    # ---------- SPEEDOMETER ----------
    with colA:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Heart Disease Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    # ---------- FEATURE CHART ----------
    with colB:
        df_display = input_df.T
        df_display.columns = ["Value"]

        fig2 = px.bar(df_display, title="Input Feature Overview")
        st.plotly_chart(fig2, use_container_width=True)

    # ---------- FINAL RESULT ----------
    if prediction == 1:
        st.error("⚠ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
