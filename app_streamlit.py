# app_streamlit.py
"""
Streamlit Admission Predictor - corrected (no NameError)
- Drop-in replacement: trains/loads sklearn models, predicts, shows percentages.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback
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

# -------- CSS (clean; no white boxes) --------
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
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
    margin:auto;
    margin-top:180px;
}
</style>
""", unsafe_allow_html=True)

# -------- Title & layout skeleton --------
st.markdown('<div class="title-bar">Graduate Admission Prediction using Linear Regression</div>', unsafe_allow_html=True)
st.write("")

col1, col_mid, col2 = st.columns([3, 0.2, 3])

# ---------- Left: instructions ----------
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

# ---------- Right: uploader + form ----------
uploaded_file = None
with col2:
    uploaded_file = st.file_uploader("Upload CSV file (optional)", type=["csv"])
    if uploaded_file:
        try:
            preview_df = pd.read_csv(uploaded_file)
            st.caption("Preview (top 3 rows):")
            st.dataframe(preview_df.head(3), height=100)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")

    # Put the form inside col2 so predict_button is always defined before use
    with st.form(key='predict_form'):
        GRE = st.number_input("GRE Score", min_value=0, max_value=340, value=320, step=1)
        TOEFL = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110, step=1)
        Univ = st.number_input("University Rating", min_value=1, max_value=5, value=3, step=1)
        SOP = st.number_input("SOP", min_value=1.0, max_value=5.0, value=3.0, step=0.1, format="%.1f")
        LOR = st.number_input("LOR", min_value=1.0, max_value=5.0, value=3.0, step=0.1, format="%.1f")
        CGPA = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.5, step=0.01, format="%.2f")
        Research = st.selectbox("Research", options=[0, 1], index=1, format_func=lambda x: "Yes" if x==1 else "No")

        predict_button = st.form_submit_button("Predict")

# -------- Helpers: load or train models --------
def load_or_train_models(uploaded_csv):
    status_msgs = []
    # Try loading existing models
    try:
        if all(os.path.exists(MODEL_FILES[k]) for k in MODEL_FILES):
            lr = joblib.load(MODEL_FILES['lr'])
            dt = joblib.load(MODEL_FILES['dt'])
            rf = joblib.load(MODEL_FILES['rf'])
            scaler = joblib.load(MODELSCALER := MODEL_FILES['scaler'])
            status_msgs.append("Loaded models from disk.")
            return lr, dt, rf, scaler, "\n".join(status_msgs)
    except Exception:
        status_msgs.append("Failed to load models from disk; will attempt to train.")

    # Load CSV (uploaded preferred)
    try:
        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
            status_msgs.append("Using uploaded CSV for training.")
        else:
            if not os.path.exists("Admission_Predict.csv"):
                raise FileNotFoundError("No Admission_Predict.csv in repo root and no file uploaded.")
            df = pd.read_csv("Admission_Predict.csv")
            status_msgs.append("Using Admission_Predict.csv from repo for training.")
    except Exception as e:
        raise RuntimeError(f"Could not load CSV for training: {e}")

    # Validate columns
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Prepare data
    df = df.dropna(subset=FEATURES + [TARGET])
    if df.shape[0] < 5:
        raise ValueError("Not enough rows after dropping NA to train models.")
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)
    if y.max() > 1.0:
        y = y / 100.0

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save scaler (min/max)
    min_vals = X_train.min()
    max_vals = X_train.max()
    scaler = {'min': min_vals.to_dict(), 'max': max_vals.to_dict()}
    joblib.dump(scaler, MODEL_FILES['scaler'])
    status_msgs.append("Saved scaler info.")

    # Train models
    lr = LinearRegression(); lr.fit(X_train, y_train)
    dt = DecisionTreeRegressor(random_state=42, max_depth=6); dt.fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8); rf.fit(X_train, y_train)

    # Save models & meta
    joblib.dump(lr, MODEL_FILES['lr'])
    joblib.dump(dt, MODEL_FILES['dt'])
    joblib.dump(rf, MODEL_FILES['rf'])
    joblib.dump({'features': FEATURES, 'target': TARGET}, MODEL_FILES['meta'])
    status_msgs.append(f"Trained and saved models (rows used: {len(X_train)+len(X_test)}).")

    return lr, dt, rf, scaler, "\n".join(status_msgs)


# -------- Prediction logic (safe, no NameError) --------
if predict_button:
    try:
        lr_model, dt_model, rf_model, scaler_obj, status = load_or_train_models(uploaded_file)
        st.success("Models ready.")
        st.info(status)
    except Exception as e:
        st.error(f"Model load/train error: {e}")
        st.text("Traceback (debug):")
        st.text(traceback.format_exc())
        st.stop()

    # build input array
    try:
        input_vals = [float(GRE), float(TOEFL), float(Univ), float(SOP), float(LOR), float(CGPA), 1 if Research==1 else 0]
        Xinp = np.array(input_vals).reshape(1, -1)
    except Exception as e:
        st.error(f"Invalid input values: {e}")
        st.stop()

    # Predict per-model, handle errors
    preds = {}
    for name, model in [("Linear Regression", lr_model),
                        ("Decision Tree", dt_model),
                        ("Random Forest", rf_model)]:
        try:
            p = model.predict(Xinp)[0]
            if not np.isfinite(p):
                raise ValueError("Non-finite prediction")
            p = float(np.clip(p, 0.0, 1.0))
            preds[name] = p
        except Exception as e:
            preds[name] = None
            st.warning(f"{name} prediction failed: {e}")

    if all(v is None for v in preds.values()):
        st.error("All model predictions failed. See warnings above.")
        st.stop()

    valid_vals = [v for v in preds.values() if v is not None]
    ensemble_val = float(np.mean(valid_vals))

    # Display ensemble result like screenshot
    display_bracket = f"[{ensemble_val*100:.5f}]"
    st.markdown(f"<div class='result-text'>Admission chances are {display_bracket}</div>", unsafe_allow_html=True)

    # Show per-model percentages
    def to_pct(x):
        return "N/A" if x is None else f"{x*100:.5f}%"
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Linear Regression", to_pct(preds.get("Linear Regression")))
    col_b.metric("Decision Tree", to_pct(preds.get("Decision Tree")))
    col_c.metric("Random Forest", to_pct(preds.get("Random Forest")))

    # Debug table of raw outputs
    debug_df = pd.DataFrame({
        "model": list(preds.keys()),
        "pred_raw": [preds[k] if preds[k] is not None else np.nan for k in preds]
    })
    st.caption("Debug: raw model outputs (0-1 scale)")
    st.table(debug_df)
