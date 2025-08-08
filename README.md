# 🔧 Anomaly/Defect Prediction (Manufacturing Sensors)

**Consulting Engagement:** Built and delivered for an early-stage manufacturing startup, providing a reproducible pipeline and stakeholder-ready explainability to support quality and yield decisions.

Supervised models to predict **short defects** from machine/sensor data. The pipeline covers EDA, preprocessing, class-imbalance handling, model benchmarking, and explainability.

## 📂 Dataset
- **Source:** CSV (e.g., `Defect Data V3 - output.csv`)
- **Target:** `short_defect_present_prediction` (binary)
- **Features:** Sensor & machine signals; timestamp dropped during preprocessing

## 🛠 Tools & Frameworks
Python • Pandas • NumPy • Scikit-learn • XGBoost • Imbalanced-learn (SMOTE) • TensorFlow/Keras • Matplotlib • Seaborn • SHAP ( + SciKeras for permutation importance)

## 🔍 Workflow (Light Overview)
- **EDA:** shape, head, describe, info; class distribution plot  
- **Preprocessing:** drop all-NaN columns; one-hot encode categoricals; scale numerics with `StandardScaler`  
- **Split:** stratified train/test  
- **Models:** Logistic Regression, Decision Tree, Random Forest, XGBoost, and a Keras MLP (early stopping)  
- **Imbalance:** `class_weight='balanced'` and **SMOTE** variants  
- **Evaluation:** classification report (precision/recall/F1, accuracy); plots for confusion matrix & performance  
- **Explainability:** coefficients & tree importances, permutation importance, **SHAP** (summary/waterfall)

## 📊 Results (Summary)
- Robust baselines with tree/ensemble and NN models; improved minority-class recall via SMOTE  
- Feature importance & SHAP highlight top sensor signals influencing defect predictions

## 🗺 Roadmap
- Extend to additional targets and larger datasets  
- Try **1D-CNNs**, **LSTM/GRU**, and **autoencoders** for temporal/unsupervised signals  
- Sliding-window time-series features and early anomaly forecasting  
- Deeper interpretability & monitoring (PR-AUC/ROC-AUC dashboards, SHAP reports)

