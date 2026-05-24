
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# for preprocessing and pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# for experiment tracking
import mlflow
import mlflow.sklearn

# Constants
HF_DATASET_REPO  = "bpinto16/amlops-visit-with-us"
HF_MODEL_REPO    = "bpinto16/wellness-tourism-mlflow-model"
MODEL_FILENAME   = "best_wellness_mlflow_model.joblib"
RANDOM_STATE     = 42
MLFLOW_EXPERIMENT_NAME = "wellness-tourism-prediction-experiment"

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

mlflow.set_tracking_uri("http://localhost:5000")

# # Configure MLflow Tracking Server
# if os.getenv("MLFLOW_TRACKING_URI"):
#     mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# try:
#     mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
# except Exception:
#     if not mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME):
#         mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
#     mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Enable automatic logging for hyperparameter tuning structures
mlflow.sklearn.autolog(max_tuning_runs=None, log_models=False)

# LOAD PREPROCESSED SPLITS
print("Load preprocessed splits from Hugging Face")
BASE = f"hf://datasets/{HF_DATASET_REPO}"

Xtrain = pd.read_csv(f"{BASE}/Xtrain.csv")
Xtest  = pd.read_csv(f"{BASE}/Xtest.csv")
ytrain = pd.read_csv(f"{BASE}/ytrain.csv").squeeze("columns")
ytest  = pd.read_csv(f"{BASE}/ytest.csv").squeeze("columns")

print(f"Xtrain : {Xtrain.shape}   ytrain : {ytrain.shape}")
print(f"Xtest  : {Xtest.shape}    ytest  : {ytest.shape}")

print("\n Fit ColumnTransformer on Xtrain")
preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None), CATEGORICAL_COLS),
        ("city", OrdinalEncoder(categories=CITYTIER_ORDER, handle_unknown="use_encoded_value", unknown_value=-1), ORDINAL_CITY_COL),
        ("scaler", StandardScaler(), NUMERIC_COLS),
        ("binary", "passthrough", BINARY_COLS),
    ],
    remainder="drop",
    verbose_feature_names_out=True,
)

# CLASS IMBALANCE WEIGHT
print("Compute class imbalance weight")
scale_pos_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# MODEL & HYPERPARAMETER GRID USING EXPLICIT PIPELINE STEPS
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    verbosity=0,
)

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

param_grid = {
    "model__n_estimators"      : [100, 150, 200],
    "model__max_depth"         : [3, 4, 5],
    "model__learning_rate"     : [0.01, 0.05, 0.1],
    "model__colsample_bytree"  : [0.6, 0.8],
    "model__colsample_bylevel" : [0.5, 0.7],
    "model__reg_lambda"        : [1.0, 1.5],
}

print("GridSearchCV with StratifiedKFold")
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

input_example = pd.DataFrame([{
    col: (Xtrain[col].median()
          if Xtrain[col].dtype in ['int64', 'float64']
          else Xtrain[col].mode()[0])
    for col in Xtrain.columns
}])

INTEGER_COLS = [
    "CityTier", "NumberOfPersonVisiting",
    "Passport", "PitchSatisfactionScore", "OwnCar",
]
input_example[INTEGER_COLS] = input_example[INTEGER_COLS].astype("float64")

print("\n-- MLflow tracking --")
with mlflow.start_run(run_name="xgb_gridsearch_parent") as parent_run:

    mlflow.set_tags({
        "model_type"              : "XGBoostClassifier",
        "pipeline_step"           : "train",
        "dataset"                 : HF_DATASET_REPO,
        "classification_threshold": str(CLASSIFICATION_THRESHOLD),
        "training_date"           : datetime.now().strftime("%Y-%m-%d"),
        "data_version"            : "v1",
        "model_version"           : "v1",
    })

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    print("Fitting GridSearchCV...")
    grid_search.fit(Xtrain, ytrain)

    # Best model extracted post-refit
    best_pipeline = grid_search.best_estimator_
    print(f"\nBest CV ROC-AUC : {grid_search.best_score_:.4f}")
    print("Best params:")
    for k, v in grid_search.best_params_.items():
        print(f"  {k:35s}: {v}")

    # Threshold tuning metrics evaluation
    print("\n-- Threshold tuning --")
    proba_test  = best_pipeline.predict_proba(Xtest)[:, 1]
    proba_train = best_pipeline.predict_proba(Xtrain)[:, 1]

    print(f"\n{'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    for thresh in np.arange(0.25, 0.56, 0.05):
        preds  = (proba_test >= thresh).astype(int)
        p  = precision_score(ytest, preds, zero_division=0)
        r  = recall_score(ytest, preds, zero_division=0)
        f1 = f1_score(ytest, preds, zero_division=0)
        marker = "selected" if abs(thresh - CLASSIFICATION_THRESHOLD) < 0.001 else ""
        print(f"{thresh:>10.2f}  {p:>10.3f}  {r:>10.3f}  {f1:>10.3f}{marker}")

    # Final predictions at chosen threshold 
    y_pred_train = (proba_train >= CLASSIFICATION_THRESHOLD).astype(int)
    y_pred_test  = (proba_test  >= CLASSIFICATION_THRESHOLD).astype(int)

    # Compute classification metrics
    train_roc_auc = roc_auc_score(ytrain, proba_train)
    test_roc_auc  = roc_auc_score(ytest,  proba_test)
    train_pr_auc  = average_precision_score(ytrain, proba_train)
    test_pr_auc   = average_precision_score(ytest,  proba_test)

    train_f1        = f1_score(ytrain,        y_pred_train, pos_label=1)
    test_f1         = f1_score(ytest,         y_pred_test,  pos_label=1)
    train_precision = precision_score(ytrain, y_pred_train, pos_label=1, zero_division=0)
    test_precision  = precision_score(ytest,  y_pred_test,  pos_label=1, zero_division=0)
    train_recall    = recall_score(ytrain,    y_pred_train, pos_label=1)
    test_recall     = recall_score(ytest,     y_pred_test,  pos_label=1)
    overfit_gap     = train_roc_auc - test_roc_auc

    # Log specialized metrics to parent run
    mlflow.log_metrics({
        "train_roc_auc"          : train_roc_auc,
        "test_roc_auc"           : test_roc_auc,
        "train_pr_auc"           : train_pr_auc,
        "test_pr_auc"            : test_pr_auc,
        "train_f1_purchase"      : train_f1,
        "test_f1_purchase"       : test_f1,
        "train_precision_purchase": train_precision,
        "test_precision_purchase" : test_precision,
        "train_recall_purchase"  : train_recall,
        "test_recall_purchase"   : test_recall,
        "overfit_gap"            : overfit_gap,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
    })

    print("\n-- Final evaluation --")
    print("Training set:")
    print(classification_report(ytrain, y_pred_train, target_names=["No Purchase", "Purchase"]))
    print("Test set:")
    print(classification_report(ytest,  y_pred_test,  target_names=["No Purchase", "Purchase"]))

    print(f"Test  ROC-AUC : {test_roc_auc:.4f}  (target ≥ 0.80)")
    print(f"Test  PR-AUC  : {test_pr_auc:.4f}")
    print(f"Train ROC-AUC : {train_roc_auc:.4f}")
    print(f"Overfit gap   : {overfit_gap:.4f}  "
          f"({'acceptable' if overfit_gap < 0.05 else 'WARNING: possible overfit'})")

    # Log model artifact to MLflow Registry
    mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        name="wellness_tourism_pipeline",
        registered_model_name="WellnessTourismPurchasePredictor",
        input_example=input_example,
    )
    print(f"\nMLflow parent run ID : {parent_run.info.run_id}")

# Upload to Hugging Face Model Hub
print("\n-- Save & upload to HF Hub --")
joblib.dump(best_pipeline, MODEL_FILENAME)
 
try:
    api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False, token=os.getenv("HF_TOKEN"))
 
api.upload_file(
    path_or_fileobj=MODEL_FILENAME,
    path_in_repo=MODEL_FILENAME,
    repo_id=HF_MODEL_REPO,
    repo_type="model",
)

print(f"  Uploaded : {MODEL_FILENAME} to {HF_MODEL_REPO}")
print("\ntrain.py completed successfully.")
