# ----- Replace or insert these helper + prediction blocks into app_streamlit.py -----

import traceback

def load_or_train_models(uploaded_csv):
    """
    Load models if present, otherwise train on uploaded_csv or repo CSV.
    Returns (lr, dt, rf, scaler, status_message)
    """
    status_msgs = []
    # Try to load existing files first
    try:
        if all(os.path.exists(MODEL_FILES[k]) for k in MODEL_FILES):
            lr = joblib.load(MODEL_FILES['lr'])
            dt = joblib.load(MODEL_FILES['dt'])
            rf = joblib.load(MODEL_FILES['rf'])
            scaler = joblib.load(MODEL_FILES['scaler'])
            status_msgs.append("Loaded models from disk.")
            return lr, dt, rf, scaler, "\n".join(status_msgs)
    except Exception as e:
        status_msgs.append(f"Failed loading models from disk: {e}. Will try training.")
    
    # If loading failed, train using uploaded or repo CSV
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

    # Clean and prepare
    df = df.dropna(subset=FEATURES + [TARGET])
    if df.shape[0] < 5:
        raise ValueError("Not enough rows after dropping NA to train models.")
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)
    if y.max() > 1.0:
        y = y / 100.0

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save scaler (min/max)
    min_vals = X_train.min()
    max_vals = X_train.max()
    scaler = {'min': min_vals.to_dict(), 'max': max_vals.to_dict()}
    joblib.dump(scaler, MODEL_FILES['scaler'])
    status_msgs.append("Saved scaler info.")

    # Train sklearn models
    lr = LinearRegression(); lr.fit(X_train, y_train)
    dt = DecisionTreeRegressor(random_state=42, max_depth=6); dt.fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8); rf.fit(X_train, y_train)

    # Save models
    joblib.dump(lr, MODEL_FILES['lr'])
    joblib.dump(dt, MODEL_FILES['dt'])
    joblib.dump(rf, MODEL_FILES['rf'])
    joblib.dump({'features': FEATURES, 'target': TARGET}, MODEL_FILES['meta'])
    status_msgs.append(f"Trained and saved models (rows used: {len(X_train)+len(X_test)}).")

    return lr, dt, rf, scaler, "\n".join(status_msgs)


# ----- Prediction block -----
if predict_button:
    # Try to load/train models and show status
    try:
        lr_model, dt_model, rf_model, scaler_obj, status = load_or_train_models(uploaded_file)
        st.success("Models ready.")
        st.info(status)
    except Exception as e:
        st.error(f"Model load/train error: {e}")
        # show traceback in logs area for debugging
        st.text("Traceback (debug):")
        st.text(traceback.format_exc())
        st.stop()

    # Prepare input array
    try:
        input_vals = [float(GRE), float(TOEFL), float(Univ), float(SOP), float(LOR), float(CGPA), 1 if Research==1 else 0]
        Xinp = np.array(input_vals).reshape(1, -1)
    except Exception as e:
        st.error(f"Invalid input values: {e}")
        st.stop()

    # Do predictions (wrap each in try/except to show per-model errors)
    preds = {}
    for name, model in [("Linear Regression", lr_model),
                        ("Decision Tree", dt_model),
                        ("Random Forest", rf_model)]:
        try:
            p = model.predict(Xinp)[0]
            # numeric sanity
            if not np.isfinite(p):
                raise ValueError("Non-finite prediction")
            p = float(np.clip(p, 0.0, 1.0))
            preds[name] = p
        except Exception as e:
            preds[name] = None
            st.warning(f"{name} prediction failed: {e}")

    # If all models failed, stop
    if all(v is None for v in preds.values()):
        st.error("All model predictions failed. See warnings above.")
        st.stop()

    # Ensemble: average over available model predictions
    valid_vals = [v for v in preds.values() if v is not None]
    ensemble_val = float(np.mean(valid_vals))

    # Format for display: percentages
    def to_pct(x):
        if x is None:
            return "N/A"
        return f"{x*100:.5f}%"  # 5 decimal places like your screenshot; adjust if needed

    # Show main centered result like screenshot (bracketed)
    display_bracket = f"[{ensemble_val*100:.5f}]"
    st.markdown(f"<div style='text-align:center; font-weight:700; margin-top:10px;'>Admission chances are {display_bracket}</div>", unsafe_allow_html=True)

    # Also show per-model metrics below
    st.write("")  # spacing
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Linear Regression", to_pct(preds.get("Linear Regression")))
    col_b.metric("Decision Tree", to_pct(preds.get("Decision Tree")))
    col_c.metric("Random Forest", to_pct(preds.get("Random Forest")))

    # Optional: show the raw numeric values in a small table for debugging
    debug_df = pd.DataFrame({
        "model": list(preds.keys()),
        "pred_raw": [preds[k] if preds[k] is not None else np.nan for k in preds]
    })
    st.caption("Debug: raw model outputs (0-1 scale)")
    st.table(debug_df)
