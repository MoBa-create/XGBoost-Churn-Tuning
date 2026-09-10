# Customer Churn Prediction with XGBoost & Hyperparameter Tuning

An end-to-end Machine Learning classification pipeline built with **XGBoost** and **GridSearchCV** to predict customer churn with automated hyperparameter optimization.

---

## 📌 Key Highlights
* **Advanced Ensemble Learning:** Utilized `XGBClassifier` to model complex non-linear churn risk patterns.
* **Hyperparameter Optimization:** Executed a 5-fold cross-validated `GridSearchCV` across 16 parameter combinations (80 total fits).
* **Optimal Configuration:**
  * `learning_rate`: 0.1
  * `max_depth`: 3
  * `n_estimators`: 100
  * `subsample`: 0.8
* **Top-Tier Performance:**
  * **Accuracy:** 98%
  * **ROC-AUC Score:** 0.9980
  * **F1-Score:** 0.98 (Class 1) / 0.99 (Class 0)
* **Pipeline Serialization:** Exported the best estimator as a production-ready `.pkl` artifact via `joblib`.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** XGBoost, Scikit-Learn, Pandas, NumPy, Joblib

---

## 📁 Repository Structure
```text
XGBoost-Churn-Tuning/
│── train_xgboost.py          # Main training script with GridSearchCV optimization
│── README.md                 # Project documentation
│── requirements.txt          # Python dependencies
│── .gitignore                # Git ignore rules
└── outputs/                  # Saved model artifacts
    └── xgboost_best_model.pkl # Optimized XGBoost model instance

🚀 How to Run

1 . Clone the repository:

	git clone https://github.com/MoBa-create/XGBoost-Churn-Tuning.git
cd XGBoost-Churn-Tuning

2 . Install dependencies:

	pip install -r requirements.txt

3 . Execute model training:

	python train_xgboost.py