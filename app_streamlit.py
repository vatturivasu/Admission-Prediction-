# app_streamlit.py
"""
Streamlit Admission Predictor app (no TensorFlow).
- Upload CSV OR the app will try to read 'Admission_Predict.csv' from the repo root.
- Trains Linear Regression, Decision Tree, and Random Forest (sklearn).
- Shows metrics, prediction UI, and allows downloading saved model zip.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------- CONFIG ----------
FEATURES = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
TARGET = 'Chance of Admit '
MODEL_FILENAMES = {
    'Linear Regression': 'model_linear_regression.pkl',
    'Decision Tree': 'model_decision_tree.pkl',
    'Random Forest': 'model_random_forest.pkl',
    'Scaler': 'scaler_minmax.pkl',
    'Meta': 'model_meta.pkl'
}

# ---------- HELPERS ----------
def load_csv(uploaded_file):
    try:
        if uploaded_file is None:
            # try to load default file from repo root
            df = pd.read_csv("Admission_Predict.csv")
        else:
            df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return None

def preprocess(df):
    # ensure required columns exist
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    # drop rows missing required features or target
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)
    # convert target to 0-1 if it's 0-100
    if y.max() > 1.0:
        y = y / 100.0
    return X, y

def train_models(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    dt = DecisionTreeRegressor(random_state=42, max_depth=6)
    dt.fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
    rf.fit(X_train, y_train)
    return lr, dt, rf

def save_models(lr, dt, rf, min_vals, max_vals):
    joblib.dump(lr, MODEL_FILENAMES['Linear Regression'])
    joblib.dump(dt, MODEL_FILENAMES['Decision Tree'])
    joblib.dump(rf, MODEL_FILENAMES['Random Forest'])
    joblib.dump({'min': min_vals.to_dict(), 'max': max_vals.to_dict()}, MODEL_FILENAMES['Scaler'])
    joblib.dump({'features': FEATURES, 'target': TARGET}, MODEL_FILENAMES['Meta'])

def make_models_zip(zip_name="saved_models.zip"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for fname in MODEL_FILENAMES.values():
            if os.path.exists(fname):
                zf.write(fname)
    buf.seek(0)
    return buf

def normalize_input(vals, scaler):
    min_vals = scaler['min']
    max_vals = scaler['max']
    out = []
    for i, f in enumerate(FEATURES):
        v = vals[i]
        mn = min_vals[f]
        mx = max_vals[f]
        out.append((v - mn) / (mx - mn + 1e-9))
    return np.array(out).reshape(1, -1)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Admission Predictor", layout="centered")
st.title("Admission Prediction (Streamlit)")

st.markdown("""
Upload the `Admission_Predict.csv` (or commit it to the repo root) and click **Train**.
The app trains Linear Regression, Decision Tree, and Random Forest and shows metrics.
""")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if st.button("Load & Preview"):
    df_preview = load_csv(uploaded_file)
    if df_preview is not None:
        st.success("Loaded dataset")
        st.dataframe(df_preview.head())

st.write("---")
st.header("Training")

train_clicked = st.button("Train models (sklearn only)")
if train_clicked:
    df = load_csv(uploaded_file)
    if df is None:
        st.stop()
    try:
        X, y = preprocess(df)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    st.write(f"Dataset rows after cleaning: {len(X)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    min_vals = X_train.min()
    max_vals = X_train.max()
    # Train
    with st.spinner("Training models..."):
        lr, dt, rf = train_models(X_train, y_train)
    # Predict & metrics
    y_pred_lr = lr.predict(X_test)
    y_pred_dt = dt.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    # clip predictions 0-1
    y_pred_lr = np.clip(y_pred_lr, 0.0, 1.0)
    y_pred_dt = np.clip(y_pred_dt, 0.0, 1.0)
    y_pred_rf = np.clip(y_pred_rf, 0.0, 1.0)
    # metrics (RMSE computed by sqrt of MSE for compatibility)
    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {"R2": float(r2_score(y_true, y_pred)), "RMSE": float(np.sqrt(mse))}
    m_lr = metrics(y_test, y_pred_lr)
    m_dt = metrics(y_test, y_pred_dt)
    m_rf = metrics(y_test, y_pred_rf)
    st.subheader("Metrics on test set")
    st.write("Linear Regression:", m_lr)
    st.write("Decision Tree:", m_dt)
    st.write("Random Forest:", m_rf)

    # Save models & scaler
    save_models(lr, dt, rf, min_vals, max_vals)
    st.success("Models trained and saved to local files.")
    # show download link for zip
    zbuf = make_models_zip()
    st.download_button("Download saved models (zip)", zbuf, file_name="saved_models.zip")

st.write("---")
st.header("Make a prediction")

# If models exist, load them
models_exist = all(os.path.exists(fname) for fname in MODEL_FILENAMES.values())
if not models_exist:
    st.info("No saved models found. Train models first (or upload model files).")

try:
    if models_exist:
        lr = joblib.load(MODEL_FILENAMES['Linear Regression'])
        dt = joblib.load(MODEL_FILENAMES['Decision Tree'])
        rf = joblib.load(MODEL_FILENAMES['Random Forest'])
        scaler = joblib.load(MODEL_FILENAMES['Scaler'])
    else:
        lr = dt = rf = scaler = None
except Exception as e:
    st.error(f"Error loading models: {e}")
    lr = dt = rf = scaler = None

# Input form
with st.form("input_form"):
    st.write("Enter student details:")
    GRE = st.number_input("GRE Score (0-340)", min_value=0, max_value=340, value=320, step=1)
    TOEFL = st.number_input("TOEFL Score (0-120)", min_value=0, max_value=120, value=110, step=1)
    Univ = st.number_input("University Rating (1-5)", min_value=1, max_value=5, value=3, step=1)
    SOP = st.number_input("SOP (1.0-5.0)", min_value=1.0, max_value=5.0, value=3.0, step=0.1, format="%.1f")
    LOR = st.number_input("LOR (1.0-5.0)", min_value=1.0, max_value=5.0, value=3.0, step=0.1, format="%.1f")
    CGPA = st.number_input("CGPA (0.0-10.0)", min_value=0.0, max_value=10.0, value=8.5, step=0.01, format="%.2f")
    Research = st.selectbox("Research (0 = No, 1 = Yes)", [0, 1], index=1)
    submit = st.form_submit_button("Predict")

if submit:
    input_vals = [GRE, TOEFL, Univ, SOP, LOR, CGPA, Research]
    X_in = np.array(input_vals).reshape(1, -1)
    if lr is None:
        st.error("No models available. Train models first.")
    else:
        pred_lr = float(lr.predict(X_in)[0])
        pred_dt = float(dt.predict(X_in)[0])
        pred_rf = float(rf.predict(X_in)[0])
        # ensemble average (no ANN here)
        ensemble = (pred_lr + pred_dt + pred_rf) / 3.0
        def fmt(x): return f"{max(0.0, min(1.0, x))*100:.2f}%"
        st.write("### Predictions")
        st.write("Linear Regression:", fmt(pred_lr))
        st.write("Decision Tree:", fmt(pred_dt))
        st.write("Random Forest:", fmt(pred_rf))
        st.write("Ensemble Average:", fmt(ensemble))

st.write("---")
st.markdown("**Notes:** This app trains models on the uploaded CSV (or `Admission_Predict.csv` in repo root). It does not require TensorFlow.")

