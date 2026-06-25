# Modeling Insights Inventory
## 1) Modeling Objective
Predict Remaining Useful Life (RUL) for aircraft engines using engineered sensor features and ML regression models.

## 2) Data & Feature Setup (Stage 1 -> Stage 2)
- Train file: `train_stage1.csv` (engineered rolling window features + clipped RUL)
- Validation file: `val_stage1.csv`
- Test file: `test_stage1.csv` (engineered features; RUL not included)
- Feature count used for modeling: **95**
- Scaling: StandardScaler fitted on TRAIN only, applied to val/test.

## 3) Model Comparison (Validation)
| Model | Run ID | Val RMSE | Val MAE | Val R2 | Model File |
|---|---:|---:|---:|---:|---|
| Linear Regression (baseline) | run_001 | 19.6373 | 15.9466 | 0.7782 | `model_v1_linear_regression.pkl` |
| Random Forest | run_002 | 18.4393 | 13.0336 | 0.8044 | `model_v2_random_forest.pkl` |
| XGBoost | run_003 | 17.9456 | 12.6944 | 0.8147 | `model_v3_xgboost.pkl` |
| XGBoost Tuned | run_004 | 17.4496 | 12.3685 | 0.8248 | `model_v4_xgboost_tuned.pkl` |

## 4) Best Model Selection
- Selected best model: **XGBoost Tuned**
- Criterion: lowest Validation RMSE
- Best validation metrics: RMSE=17.4496, MAE=12.3685, R2=0.8248
- Registered model artifact: `model_v4_xgboost_tuned.pkl`

## 5) Key Insights
1. Linear Regression is fast but underfits non-linear degradation patterns.
2. Random Forest can overfit (large train->validation gap) while still improving baseline.
3. XGBoost improves validation performance by learning non-linear relationships with built-in regularization.
4. XGBoost Tuned provides the best validation RMSE in this run.
5. Consistent feature pipeline (rolling features) and train-only scaling help reduce avoidable variance.

## 6) MLOps Artifacts Created
- Feature Store: `MLOPS_PIPELINE/feature_store/`
- Model Registry: `MLOPS_PIPELINE/model_registry/`
  - Scaler: `scaler.pkl`
  - Models: `model_v1_*` to `model_v4_*`
  - Best pointer: `best_model.pkl`
