# app_streamlit.py
"""
Streamlit Admission Predictor — clean version (no white/gray boxes)
Auto-adapts to Streamlit dark/light theme.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------- CONFIG --------
FEATURES = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
TARGET = 'Chance of Admit '
MODEL_FILES = {
    'lr': 'model_linear_regression.pkl',
    'dt': 'model_decision_tree.pkl',
    'rf': 'model_random_forest.pkl',
    'scaler': 'scaler_minmax.pkl',
    'meta': 'model_meta.pkl'
}

st.set_page_config(page_title="Graduate Admission Prediction", layout="wide")

# -------- CSS (minimal clean style, no boxes) --------
st.markdown("""
<style>
.title-bar {
    background: #666666;
    color: #fff;
    padding: 10px 20px;
    border-radius: 4px;
    font-weight: 700;
    text-align: center;
}
.predict-btn .stButton>button {
    background-color: #2ecc71;
    color: white;
    width: 100%;
    height:44px;
    border-radius:8px;
    font-weight:600;
}
.result-text {
    text-align:center;
    font-weight:700;
    margin-top:10px;
    font-size:18px;
}
.divider-center {
    width: 20px;
    height: 20px;
    background: #8884;
    border-radius: 50%;
    margin:auto;
    margin-top:180px;
}
</style>
""", unsafe_allow_html=True)

# -------- Title --------
st.markdown('<div class="title-bar">Graduate Admission Prediction using Linear Regression</div>', unsafe_allow_html=True)
st.write("")

# -------- Layout --------
col1, col_mid, col2 = st.columns([3, 0.2, 3])

with col1:
    st.markdown("#### In this project, I build a linear regression model to predict the chance of admission into a particular university based on student's profile.")
    st.markdown("**Instructions for Input Features**")
    st.markdown("""
    - GRE Score (out of 340)  
    - TOEFL Score (out of 120)  
    - University Rating (out of 5)  
    - Statement of Purpose (SOP) (out of 5)  
    - Letter of Recommendation (LOR) Strength (out of 5)  
    - Undergraduate CGPA (out of 10)  
    - Research Experience (Either 0 or 1)
    """)

with col_mid:
    st.markdown('<div class="divider-center"></div>', unsafe_allow_html=True)

with col2:
    uploaded_file = st.file_uploader("Upload CSV file (optional)", type=["csv"])
    if uploaded_file:
        try:
            preview_df = pd.read_csv(uploaded_file)
            st.caption("Preview (top 3 rows):")
            st.dataframe(preview_df.head(3), height=100)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")

    with st.form(key='predict_form'):
        GRE = st.number_input("GRE Score", min_value=0, max_value=340, value=320, step=1)
        TOEFL = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110, step=1)
        Univ = st.number_input("University Rating", min_value=1, max_value=5, value=3, step=1)
        SOP = st.number_input("SOP", min_value=1.0, max_value=5.0, value=3.0, step=0.1, format="%.1f")
        LOR = st.number_input("LOR", min_value=1.0, max_value=5.0, value=3.0, step=0.1, format="%.1f")
        CGPA = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.5, step=0.01, format="%.2f")
        Research = st.selectbox("Research", options=[0, 1], index=1, format_func=lambda x: "Yes" if x==1 else "No")

        predict_button = st.form_submit_button("Predict")

# -------- Model helper functions --------
def load_or_train_models(uploaded_csv):
    # if all model files exist, load them
    if all(os.path.exists(MODEL_FILES[k]) for k in MODEL_FILES):
        try:
            lr = joblib.load(MODEL_FILES['lr'])
            dt = joblib.load(MODEL_FILES['dt'])
            rf = joblib.load(MODEL_FILES['rf'])
            scaler = joblib.load(MODEL_FILES['scaler'])
            return lr, dt, rf, scaler
        except Exception:
            pass

    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
    elif os.path.exists("Admission_Predict.csv"):
        df = pd.read_csv("Admission_Predict.csv")
    else:
        raise FileNotFoundError("No dataset found! Please upload or add Admission_Predict.csv")

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)
    if y.max() > 1.0:
        y = y / 100.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    min_vals = X_train.min()
    max_vals = X_train.max()
    scaler = {'min': min_vals.to_dict(), 'max': max_vals.to_dict()}
    joblib.dump(scaler, MODEL_FILES['scaler'])

    lr = LinearRegression(); lr.fit(X_train, y_train)
    dt = DecisionTreeRegressor(random_state=42, max_depth=6); dt.fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8); rf.fit(X_train, y_train)
    joblib.dump(lr, MODEL_FILES['lr'])
    joblib.dump(dt, MODEL_FILES['dt'])
    joblib.dump(rf, MODEL_FILES['rf'])
    joblib.dump({'features': FEATURES, 'target': TARGET}, MODEL_FILES['meta'])
    return lr, dt, rf, scaler

# -------- Prediction --------
if predict_button:
    try:
        lr_model, dt_model, rf_model, scaler_obj = load_or_train_models(uploaded_file)
    except Exception as e:
        st.error(f"Model load/train error: {e}")
        st.stop()

    input_vals = [GRE, TOEFL, Univ, SOP, LOR, CGPA, 1 if Research==1 else 0]
    Xinp = np.array(input_vals).reshape(1, -1)

    pred_lr = float(lr_model.predict(Xinp)[0])
    pred_dt = float(dt_model.predict(Xinp)[0])
    pred_rf = float(rf_model.predict(Xinp)[0])

    pred_lr = np.clip(pred_lr, 0.0, 1.0)
    pred_dt = np.clip(pred_dt, 0.0, 1.0)
    pred_rf = np.clip(pred_rf, 0.0, 1.0)

    ensemble = (pred_lr + pred_dt + pred_rf) / 3.0
    result_val = f"[{ensemble*100:.5f}]"

    st.markdown(f"<div class='result-text'>Admission chances are {result_val}</div>", unsafe_allow_html=True)



