# app_streamlit.py
"""
Streamlit Admission Predictor - UI styled like your screenshots.
- Upload CSV OR the app will try to read 'Admission_Predict.csv' from repo root.
- Trains Linear Regression, Decision Tree, Random Forest (sklearn).
- Provides a two-column layout: left instructions, right input form with green Predict button.
- Displays result text like: "Admission chances are [63.00]"
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

# ---------------- CONFIG ----------------
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

# ---------- CSS styling to match screenshots ----------
st.markdown(
    """
    <style>
    /* page background and card look */
    .title-bar {
        background: #666666;
        color: #fff;
        padding: 10px 20px;
        border-radius: 4px;
        font-weight: 700;
        text-align: center;
    }
    .left-box {
        background: #f7f7f7;
        padding: 24px;
        border-radius: 6px;
        height: 420px;
    }
    .right-box {
        background: white;
        padding: 20px 24px;
        border-radius: 6px;
        height: 420px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    /* input fields styling - make them look like rounded boxes */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 4px !important;
        padding: 12px !important;
        height: 44px;
    }
    /* green full-width predict button */
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
    }
    /* small divider circle in middle like screenshot */
    .divider-center {
        width: 20px;
        height: 20px;
        background: #efefef;
        border-radius: 50%;
        margin:auto;
        margin-top:180px;
    }
    </style>
    """, unsafe_allow_html=True
)

# ---------- Title bar ----------
st.markdown('<div class="title-bar">Graduate Admission Prediction using Linear Regression</div>', unsafe_allow_html=True)
st.write("")  # spacing

# ---------- Top layout: left instructions, center divider, right form ----------
col1, col_mid, col2 = st.columns([3, 0.2, 3])

with col1:
    st.markdown('<div class="left-box">', unsafe_allow_html=True)
    st.markdown("#### In this project, I build a linear regression model to predict the chance of admission into a particular university based on student\'s profile.")
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
    st.markdown("</div>", unsafe_allow_html=True)

with col_mid:
    st.markdown('<div class="divider-center"></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="right-box">', unsafe_allow_html=True)

    # File uploader and preview (small)
    uploaded_file = st.file_uploader("Upload CSV file (optional)", type=["csv"])
    if uploaded_file:
        try:
            preview_df = pd.read_csv(uploaded_file)
            st.caption("Preview (top 3 rows):")
            st.dataframe(preview_df.head(3), height=100)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")

    # Small instruction note
    st.markdown("##")
    # Input fields (matching order in screenshot)
    with st.form(key='predict_form'):
        GRE = st.number_input("GRE Score", min_value=0, max_value=340, value=320, step=1)
        TOEFL = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110, step=1)
        Univ = st.number_input("University Rating", min_value=1, max_value=5, value=3, step=1)
        SOP = st.number_input("SOP", min_value=1.0, max_value=5.0, value=3.0, step=0.1, format="%.1f")
        LOR = st.number_input("LOR", min_value=1.0, max_value=5.0, value=3.0, step=0.1, format="%.1f")
        CGPA = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.5, step=0.01, format="%.2f")
        Research = st.selectbox("Research", options=[0,1], index=1, format_func=lambda x: "Yes" if x==1 else "No")
        st.write("")  # spacing

        # Predict button styled green
        predict_button = st.form_submit_button("Predict")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Helper functions ----------
def load_or_train_models(uploaded_csv):
    """
    If model files exist, load them. Otherwise train on uploaded_csv or repo CSV.
    Returns models (lr, dt, rf) and scaler (min/max dict) or raises error.
    """
    # if all model files exist, load them
    if all(os.path.exists(MODEL_FILES[k]) for k in MODEL_FILES):
        try:
            lr = joblib.load(MODEL_FILES['lr'])
            dt = joblib.load(MODEL_FILES['dt'])
            rf = joblib.load(MODEL_FILES['rf'])
            scaler = joblib.load(MODEL_FILES['scaler'])
            return lr, dt, rf, scaler
        except Exception:
            # fall through to train
            pass

    # load training data (uploaded or file in repo)
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
    else:
        if not os.path.exists("Admission_Predict.csv"):
            raise FileNotFoundError("No Admission_Predict.csv in repo root and no file uploaded.")
        df = pd.read_csv("Admission_Predict.csv")

    # validate columns
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)
    if y.max() > 1.0:
        y = y / 100.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # scaler info:
    min_vals = X_train.min()
    max_vals = X_train.max()
    scaler = {'min': min_vals.to_dict(), 'max': max_vals.to_dict()}
    joblib.dump(scaler, MODEL_FILES['scaler'])

    # train models
    lr = LinearRegression(); lr.fit(X_train, y_train)
    dt = DecisionTreeRegressor(random_state=42, max_depth=6); dt.fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8); rf.fit(X_train, y_train)
    joblib.dump(lr, MODEL_FILES['lr'])
    joblib.dump(dt, MODEL_FILES['dt'])
    joblib.dump(rf, MODEL_FILES['rf'])
    # save meta
    joblib.dump({'features': FEATURES, 'target': TARGET}, MODEL_FILES['meta'])

    return lr, dt, rf, scaler

# ---------- Prediction & display ----------
if predict_button:
    # attempt to load or train models
    try:
        lr_model, dt_model, rf_model, scaler_obj = load_or_train_models(uploaded_file)
    except Exception as e:
        st.error(f"Model load/train error: {e}")
        st.stop()

    # prepare input
    input_vals = [GRE, TOEFL, Univ, SOP, LOR, CGPA, 1 if Research==1 else 0]
    Xinp = np.array(input_vals).reshape(1, -1)

    # predict
    pred_lr = float(lr_model.predict(Xinp)[0])
    pred_dt = float(dt_model.predict(Xinp)[0])
    pred_rf = float(rf_model.predict(Xinp)[0])

    # clip predictions 0-1
    pred_lr = float(np.clip(pred_lr, 0.0, 1.0))
    pred_dt = float(np.clip(pred_dt, 0.0, 1.0))
    pred_rf = float(np.clip(pred_rf, 0.0, 1.0))

    # ensemble (simple avg)
    ensemble = (pred_lr + pred_dt + pred_rf) / 3.0

    # format percentage value like the screenshot: e.g. 63.06085954 -> show 63.06 (but within brackets)
    display_val = f"{ensemble*100:.5f}"  # keep many decimals like screenshot (you can change)
    # but screenshot shows a number like [63.06085954] - we'll format to two decimals inside brackets for clarity
    display_val_bracket = f"[{ensemble*100:.5f}]"

    # center the result as in screenshot
    st.markdown(f"<div class='result-text'>Admission chances are {display_val_bracket}</div>", unsafe_allow_html=True)

    # Also show per-model values beneath (small)
    st.write("")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Linear Regression", f"{pred_lr*100:.2f}%")
    col_b.metric("Decision Tree", f"{pred_dt*100:.2f}%")
    col_c.metric("Random Forest", f"{pred_rf*100:.2f}%")

# ---------- Footer spacing ----------
st.write("") 
st.write("")


