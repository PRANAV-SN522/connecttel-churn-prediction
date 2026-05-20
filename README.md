# ConnectTel Customer Churn Prediction

**Author:** SN PRANAV | **Date:** May 2026

## Project Overview
Built an end-to-end machine learning system to predict customer churn
for ConnectTel, a telecommunications provider losing revenue every
quarter due to customers switching to competitors.

## Results
| Model | AUC-ROC | Recall | Precision |
|---|---|---|---|
| Logistic Regression | 0.8410 | 0.7914 | 0.4992 |
| Random Forest | 0.8442 | 0.7380 | 0.5691 |
| **XGBoost (tuned)** | **0.8470** | **0.7968** | **0.5129** |

## Key Findings
- Fiber Optic users churn at 42% vs DSL at 19% (p < 0.05)
- Month-to-month contracts are the single largest churn driver (SHAP)
- Optimal decision threshold: 0.37 (not default 0.50)
- Estimated ROI of retention campaign: 3,471%

## Files
- `connecttel_churn_prediction.ipynb` — Full notebook with EDA, models, SHAP
- `connecttel_churn_source.py` — Clean runnable source code
- `ConnectTel_Churn_Prediction_Report.pdf` — Business report
- `ConnectTel_Churn_Prediction.pptx` — Presentation slides

## Tech Stack
Python · XGBoost · SHAP · Scikit-learn · Pandas · Matplotlib · Seaborn

## Dataset
[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
7,043 customers · 21 features · 26.5% churn rate
