import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open('income_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load encoders
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'gender', 'native-country']
encoders = {}
for col in categorical_columns:
    with open(f'{col}_encoder.pkl', 'rb') as f:
        encoders[col] = pickle.load(f)

# Load model feature columns
with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.set_page_config(page_title="Income Predictor", layout="wide")

# Sidebar input for single prediction
st.sidebar.header("🎯 Input Employee Details")

age = st.sidebar.slider("Age", 18, 90, 30)
education = st.sidebar.selectbox("Education Level", encoders['education'].classes_)
occupation = st.sidebar.selectbox("Job Role", encoders['occupation'].classes_)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
education_num = st.sidebar.slider("Years of Education", 1, 16, 8)

# Additional optional sidebar fields
marital_status = st.sidebar.selectbox("Marital Status", encoders['marital-status'].classes_)
workclass = st.sidebar.selectbox("Workclass", encoders['workclass'].classes_)
relationship = st.sidebar.selectbox("Relationship", encoders['relationship'].classes_)
race = st.sidebar.selectbox("Race", encoders['race'].classes_)
gender = st.sidebar.selectbox("Gender", encoders['gender'].classes_)
native_country = st.sidebar.selectbox("Native Country", encoders['native-country'].classes_)
capital_gain = st.sidebar.number_input("Capital Gain", 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0)
fnlwgt = st.sidebar.number_input("Fnlwgt", 1)

# Title and file uploader
st.title("📊 Income Classification Predictor")
st.markdown("Upload CSV or use the sidebar to predict if income >50K")

uploaded_file = st.file_uploader("📂 Upload your CSV file here (limit 200MB)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded data preview:")
    st.dataframe(data.head())

    # Encode columns
    for col in categorical_columns:
        if col in data.columns:
            data[col] = encoders[col].transform(data[col])

    # Ensure required columns
    missing_cols = [col for col in model_columns if col not in data.columns]
    for col in missing_cols:
        data[col] = 0

    data = data[model_columns]

    # Prediction
    predictions = model.predict(data)
    data["Predicted Income"] = np.where(predictions == 1, ">50K", "<=50K")

    st.subheader("✅ Predictions:")
    st.dataframe(data.head())

    # Download button
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")

# Single prediction from sidebar input
st.markdown("---")
st.subheader("🎯 Predict for Entered Details")

input_dict = {
    'age': age,
    'workclass': encoders['workclass'].transform([workclass])[0],
    'fnlwgt': fnlwgt,
    'education': encoders['education'].transform([education])[0],
    'educational-num': education_num,
    'marital-status': encoders['marital-status'].transform([marital_status])[0],
    'occupation': encoders['occupation'].transform([occupation])[0],
    'relationship': encoders['relationship'].transform([relationship])[0],
    'race': encoders['race'].transform([race])[0],
    'gender': encoders['gender'].transform([gender])[0],
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': encoders['native-country'].transform([native_country])[0]
}

input_df = pd.DataFrame([input_dict])
input_df = input_df[model_columns]

if st.button("📌 Predict Now"):
    pred = model.predict(input_df)[0]
    result = ">50K" if pred == 1 else "<=50K"
    st.success(f"🎉 Predicted Income: {result}")
