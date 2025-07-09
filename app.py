# ⚠️ st.set_page_config must be FIRST Streamlit command
!pip install streamlit pandas numpy scikit-learn

import streamlit as st
st.set_page_config(page_title="Obesity Category Predictor", layout="centered")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ✅ Use st.cache_resource for ML pipelines
@st.cache_resource
def load_model():
    df = pd.read_csv("data.csv")

    # Encode categorical features
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols.remove('NObeyesdad')

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Encode target
    target_le = LabelEncoder()
    df['NObeyesdad'] = target_le.fit_transform(df['NObeyesdad'])

    # Split features and target
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)

    return model, label_encoders, target_le, scaler, X.columns.tolist()

# Load model and encoders
model, label_encoders, target_le, scaler, feature_names = load_model()

# ---------------------- UI ------------------------
st.title("🏥 Obesity Category Predictor")
st.markdown("Fill in your lifestyle and health details to predict your obesity level.")

with st.form("obesity_form"):
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.slider("Age", 10, 100, 25)
    height = st.number_input("Height (in meters)", 1.0, 2.5, 1.70, step=0.01)
    weight = st.number_input("Weight (in kg)", 30, 200, 70)

    fhwo = st.selectbox("Family history with overweight?", ['yes', 'no'])
    favc = st.selectbox("Do you eat high caloric food frequently?", ['yes', 'no'])
    fcvc = st.slider("How often do you eat vegetables? (1=Never, 3=Frequently)", 1, 3, 2)
    ncp = st.slider("How many main meals do you have daily?", 1.0, 5.0, 3.0, step=0.5)

    caec = st.selectbox("Do you eat between meals?", ['no', 'Sometimes', 'Frequently', 'Always'])
    smoke = st.selectbox("Do you smoke?", ['yes', 'no'])
    ch2o = st.slider("Water intake (liters/day)", 0.5, 3.0, 2.0, step=0.1)
    scc = st.selectbox("Do you monitor your calorie intake?", ['yes', 'no'])

    faf = st.slider("Physical activity (days/week)", 0.0, 3.0, 1.0, step=0.5)
    tue = st.slider("Daily tech usage time (hours/day)", 0, 3, 2)
    calc = st.selectbox("Alcohol consumption", ['no', 'Sometimes', 'Frequently', 'Always'])
    mtrans = st.selectbox("Transport mode", ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'])

    submitted = st.form_submit_button("Predict Obesity Category")

    if submitted:
        user_input = {
            'Gender': label_encoders['Gender'].transform([gender])[0],
            'Age': age,
            'Height': height,
            'Weight': weight,
            'family_history_with_overweight': label_encoders['family_history_with_overweight'].transform([fhwo])[0],
            'FAVC': label_encoders['FAVC'].transform([favc])[0],
            'FCVC': fcvc,
            'NCP': ncp,
            'CAEC': label_encoders['CAEC'].transform([caec])[0],
            'SMOKE': label_encoders['SMOKE'].transform([smoke])[0],
            'CH2O': ch2o,
            'SCC': label_encoders['SCC'].transform([scc])[0],
            'FAF': faf,
            'TUE': tue,
            'CALC': label_encoders['CALC'].transform([calc])[0],
            'MTRANS': label_encoders['MTRANS'].transform([mtrans])[0],
        }

        user_df = pd.DataFrame([user_input])
        user_scaled = scaler.transform(user_df)
        prediction = model.predict(user_scaled)[0]
        label = target_le.inverse_transform([prediction])[0]

        st.success(f"🏷️ Predicted Obesity Category: **{label}**")
