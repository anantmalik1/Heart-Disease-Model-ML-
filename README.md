<img src="https://img.freepik.com/free-vector/heart-health-concept-illustration_114360-892.jpg" width="100%" />

# ❤️ Heart Disease Prediction System  
### AI Powered Cardiovascular Risk Assessment Dashboard  

---

## 🚀 Live Deployment

🔗 **Try the Live App Here:**  
https://heart-disease-prediction-01.streamlit.app/

---

## 📌 Abstract

Heart disease remains one of the leading causes of mortality worldwide. Early detection and prediction of cardiovascular risk can significantly improve treatment outcomes and reduce healthcare costs.

This project presents a **Heart Disease Prediction System** built using **Machine Learning (KNN Algorithm)** and deployed using **Streamlit**. The system predicts the likelihood of heart disease based on key medical attributes such as age, blood pressure, cholesterol levels, fasting blood sugar, heart rate, and more.

The application provides:

- 📊 Real-time Risk Percentage (Speedometer Gauge)
- 📈 3D Interactive Feature Visualization
- 🌗 Dark Mode Dashboard
- ⚡ Instant Prediction System
- 🧠 Intelligent Risk Assessment

The model performance is evaluated using classification metrics such as accuracy, precision, recall, and confusion matrix.

---

## 📖 Introduction

Healthcare systems generate vast amounts of medical data daily. Extracting meaningful insights from this data using Machine Learning can assist doctors in making early and effective medical decisions.

This system leverages a supervised machine learning algorithm (K-Nearest Neighbors) to classify patients into high-risk or low-risk categories based on 11+ clinical features.

The goal is to build a reliable and interactive medical dashboard that demonstrates the practical implementation of ML in real-world healthcare scenarios.

---

## 🎯 Aim

To predict heart disease risk based on user-provided medical parameters using a trained machine learning model.

---

## 🎯 Objectives

- Develop an ML-based heart disease classification model.
- Create a real-time interactive web application.
- Visualize patient risk using dynamic dashboards.
- Deploy the model for public access using Streamlit Cloud.

---

## 🌍 Project Scope

This project is a generic predictive healthcare system that can be extended for:

- Clinical support systems
- Preventive health monitoring
- Medical research tools
- AI-based hospital dashboards

The system can be expanded to include advanced algorithms and explainable AI tools.

---

## ⚙️ System Architecture

### Modules:

- **User Input Module** – Collects patient medical parameters.
- **Prediction Module** – Uses trained KNN model for classification.
- **Visualization Module** – Displays risk score and 3D feature charts.
- **Deployment Module** – Hosted on Streamlit Cloud.

---

## 🛠 Technology Stack

### 💻 Programming Language
- Python

### 📊 Data Processing
- Pandas
- NumPy

### 🤖 Machine Learning
- Scikit-Learn
- K-Nearest Neighbors (KNN)
- StandardScaler

### 📈 Visualization
- Plotly
- Streamlit

### ☁️ Deployment
- Streamlit Cloud

### 🧰 Tools Used
- VS Code
- GitHub

---

## 🧬 Machine Learning Model

- Algorithm: **K-Nearest Neighbors (KNN)**
- Data Preprocessing: StandardScaler
- Output: Binary Classification (High Risk / Low Risk)
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix

---

## 📂 Project Structure

Heart-Disease-Model-ML/
│
├── app.py                  # Main Streamlit application
├── KNN_heart.pkl           # Trained KNN model file
├── scaler.pkl              # StandardScaler object
├── columns.pkl             # Feature column order file
├── requirements.txt        # Required dependencies
└── README.md               # Project documentation
