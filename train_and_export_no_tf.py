# train_and_export_no_tf.py
"""
Train models locally outside Streamlit (no TensorFlow).
Run: python train_and_export_no_tf.py
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "Admission_Predict.csv"
FEATURES = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
TARGET = 'Chance of Admit '

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=FEATURES + [TARGET])
X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)
if y.max() > 1.0:
    y = y / 100.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

min_vals = X_train.min()
max_vals = X_train.max()
joblib.dump({'min': min_vals.to_dict(), 'max': max_vals.to_dict()}, "scaler_minmax.pkl")

lr = LinearRegression(); lr.fit(X_train, y_train)
dt = DecisionTreeRegressor(random_state=42, max_depth=6); dt.fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8); rf.fit(X_train, y_train)

joblib.dump(lr, "model_linear_regression.pkl")
joblib.dump(dt, "model_decision_tree.pkl")
joblib.dump(rf, "model_random_forest.pkl")
joblib.dump({'features': FEATURES, 'target': TARGET}, "model_meta.pkl")

# print metrics
for name, model in [("Linear Regression", lr), ("Decision Tree", dt), ("Random Forest", rf)]:
    y_pred = model.predict(X_test)
    print(name, "R2:", r2_score(y_test, y_pred), "RMSE:", mean_squared_error(y_test, y_pred, squared=False))

print("Saved models to current directory.")
