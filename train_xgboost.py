from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

np.random.seed(42)
n_samples = 1000

age = np.random.randint(18, 70, n_samples)
balance = np.random.uniform(0, 120000, n_samples)
credit_score = np.random.randint(300, 850, n_samples)
num_products = np.random.randint(1, 4, n_samples)
is_active = np.random.randint(0, 2, n_samples)

logit = -2.0 + 0.04 * age - 0.00001 * balance - 0.001 * credit_score - 0.7 * is_active + 0.6 * num_products
prob = 1 / (1 + np.exp(-logit))
exited = (prob > 0.45).astype(int)

df = pd.DataFrame({
    "Age": age,
    "Balance": balance.round(2),
    "CreditScore": credit_score,
    "NumOfProducts": num_products,
    "IsActiveMember": is_active,
    "Exited": exited
})

x = df.drop("Exited", axis=1)
y = df["Exited"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

xgb = XGBClassifier(random_state=42, eval_metric="logloss")

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

print("=== Starting Grid Search Optimization ===")
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)
y_prob = best_model.predict_proba(x_test)[:, 1]

print("\n=== Best Hyperparameters Found ===")
print(grid_search.best_params_)

print("\n=== Model Performance ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

joblib.dump(best_model, os.path.join(OUTPUTS_DIR, "xgboost_best_model.pkl"))
print("\nBest XGBoost model saved successfully to 'outputs' folder!")