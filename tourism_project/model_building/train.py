
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# for preprocessing and pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline

# Constants
HF_DATASET_REPO  = "bpinto16/amlops-visit-with-us"
HF_MODEL_REPO    = "bpinto16/wellness-tourism-model"
MODEL_FILENAME   = "best_wellness_model.joblib"
RANDOM_STATE     = 42

# Classification threshold
CLASSIFICATION_THRESHOLD = 0.40

# Hugging Face auth
api = HfApi(token=os.getenv("HF_TOKEN"))

CITYTIER_ORDER   = [[1, 2, 3]]
CATEGORICAL_COLS = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "ProductPitched",
    "Designation",
]
ORDINAL_CITY_COL = ["CityTier"]
NUMERIC_COLS = [
    "Age",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "PitchSatisfactionScore",
    "NumberOfChildrenVisiting",
    "MonthlyIncome_log",
]
BINARY_COLS = ["Passport", "OwnCar"]


# LOAD PREPROCESSED SPLITS
print("Load preprocessed splits from Hugging Face")

BASE = f"hf://datasets/{HF_DATASET_REPO}"

Xtrain = pd.read_csv(f"{BASE}/Xtrain.csv")
Xtest  = pd.read_csv(f"{BASE}/Xtest.csv")

# .squeeze() converts single-column DataFrame to 1D Series
ytrain = pd.read_csv(f"{BASE}/ytrain.csv").squeeze("columns")
ytest  = pd.read_csv(f"{BASE}/ytest.csv").squeeze("columns")

print(f"Xtrain : {Xtrain.shape}   ytrain : {ytrain.shape}")
print(f"Xtest  : {Xtest.shape}    ytest  : {ytest.shape}")
print(f"ytrain dtype : {ytrain.dtype}")
print(f"ytest dtype  : {ytest.dtype}")

print("\n-- Fit ColumnTransformer on Xtrain --")
preprocessor = ColumnTransformer(
    transformers=[
        (
            "ohe",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop=None,
            ),
            CATEGORICAL_COLS,
        ),
        (
            "city",
            OrdinalEncoder(
                categories=CITYTIER_ORDER,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            ORDINAL_CITY_COL,
        ),
        (
            "scaler",
            StandardScaler(),
            NUMERIC_COLS,
        ),
        (
            "binary",
            "passthrough",
            BINARY_COLS,
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=True,
)


# CLASS IMBALANCE WEIGHT
print("Compute class imbalance weight")


# scale_pos_weight tells XGBoost to penalise false negatives on the minority
scale_pos_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Class counts  : {ytrain.value_counts().to_dict()}")
print(f"scale_pos_weight : {scale_pos_weight:.4f}  "
      f"(negative / positive = {ytrain.value_counts()[0]} / {ytrain.value_counts()[1]})\n")


# MODEL & HYPERPARAMETER GRID
print("Define XGBoost model and hyperparameter grid")

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    eval_metric="logloss",              # internal eval during boosting rounds
    verbosity=0,                        # suppress per-tree logs
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# Grid covers the most impactful XGBoost knobs for this dataset size
param_grid = {
    "xgbclassifier__n_estimators"      : [50, 100, 150],
    "xgbclassifier__max_depth"         : [3, 4, 5],
    "xgbclassifier__learning_rate"     : [0.01, 0.05, 0.1],
    "xgbclassifier__colsample_bytree"  : [0.5, 0.7],
    "xgbclassifier__colsample_bylevel" : [0.5, 0.7],
    "xgbclassifier__reg_lambda"        : [0.5, 1.0],
}

# CROSS-VALIDATED GRID SEARCH
print("GridSearchCV with StratifiedKFold")

# StratifiedKFold preserves the class ratio in every fold
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring="roc_auc",   # robust to class imbalance unlike accuracy
    n_jobs=-1,           # use all available CPU cores
    refit=True,          # refit best params on full Xtrain after search
    verbose=1,
)

print("Fitting GridSearchCV — this may take several minutes...")
grid_search.fit(Xtrain, ytrain)

print(f"\nBest CV ROC-AUC : {grid_search.best_score_:.4f}")
print(f"Best params     :\n")
for k, v in grid_search.best_params_.items():
    print(f"  {k:25s}: {v}")


# BEST MODEL
print("Extract best model")

best_pipeline = grid_search.best_estimator_


# THRESHOLD TUNING
print("Threshold tuning")

proba_test = best_pipeline.predict_proba(Xtest)[:, 1]

print(f"\n{'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
print("-" * 46)
for thresh in np.arange(0.25, 0.56, 0.05):
    preds = (proba_test >= thresh).astype(int)
    p  = precision_score(ytest, preds, zero_division=0)
    r  = recall_score(ytest, preds, zero_division=0)
    f1 = f1_score(ytest, preds, zero_division=0)
    marker = "  selected" if abs(thresh - CLASSIFICATION_THRESHOLD) < 0.001 else ""
    print(f"{thresh:>10.2f}  {p:>10.3f}  {r:>10.3f}  {f1:>10.3f}{marker}")

# Final Evaluation
print("\n-- Final evaluation --")
proba_train  = best_pipeline.predict_proba(Xtrain)[:, 1]
y_pred_train = (proba_train >= CLASSIFICATION_THRESHOLD).astype(int)
y_pred_test  = (proba_test  >= CLASSIFICATION_THRESHOLD).astype(int)

print("\nTraining set:")
print(classification_report(ytrain, y_pred_train, target_names=["No Purchase", "Purchase"]))

print("Test set:")
print(classification_report(ytest, y_pred_test, target_names=["No Purchase", "Purchase"]))

roc_auc   = roc_auc_score(ytest, proba_test)
pr_auc    = average_precision_score(ytest, proba_test)
roc_train = roc_auc_score(ytrain, proba_train)
print(f"Test  ROC-AUC : {roc_auc:.4f}  (target ≥ 0.80)")
print(f"Test  PR-AUC  : {pr_auc:.4f}")
print(f"Train ROC-AUC : {roc_train:.4f}")
print(f"Overfit gap   : {roc_train - roc_auc:.4f}  "
      f"({'acceptable' if roc_train - roc_auc < 0.05 else 'WARNING: possible overfit'})")

# Save Pipeline
print("\nSave pipeline ")
joblib.dump(best_pipeline, MODEL_FILENAME)
print(f"  Saved : {MODEL_FILENAME}  (pipeline: preprocessor + XGBClassifier)")

# Upload to Hugging Face Model Repo
print("\n Upload to Hugging Face model repo")
try:
    api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
    print(f"Model repo '{HF_MODEL_REPO}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False,
                token=os.getenv("HF_TOKEN"))
    print(f"Created model repo : {HF_MODEL_REPO}")

api.upload_file(
    path_or_fileobj=MODEL_FILENAME,
    path_in_repo=MODEL_FILENAME,
    repo_id=HF_MODEL_REPO,
    repo_type="model",
)
print(f"  Uploaded : {MODEL_FILENAME} to {HF_MODEL_REPO}")

print("\ntrain.py completed successfully.")
