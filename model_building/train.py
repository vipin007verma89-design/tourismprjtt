# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Ensure HF_TOKEN is available in the environment for HfApi
# Check if HF_TOKEN is None, if so, the previous cell didn't set it.
# This is a fallback and good practice, though the earlier cell should handle it.
hf_token_val = os.getenv("HF_TOKEN") # Get token once
if hf_token_val is None:
    print("HF_TOKEN environment variable not set. Please set it.")
    # Optionally, raise an error or exit if token is critical

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi(token=hf_token_val) # Pass the token to HfApi


Xtrain_path = "hf://datasets/verma89/tourismprjtt/Xtrain.csv"
Xtest_path = "hf://datasets/verma89/tourismprjtt/Xtest.csv"
ytrain_path = "hf://datasets/verma89/tourismprjtt/ytrain.csv"
ytest_path = "hf://datasets/verma89/tourismprjtt/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# Define numeric and categorical features
numeric_features = [
    'Age', 'DurationOfPitch', 'NumberOfTrips', 'MonthlyIncome'
]

categorical_features = [
   'TypeofContact', 'CityTier', 'Occupation', 'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation'

]

# set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost Regressor
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)


# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75],
    'xgbclassifier__max_depth': [2, 3],
    'xgbclassifier__colsample_bytree': [0.4, 0.5],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5],
    'xgbclassifier__learning_rate': [0.01, 0.05],
    'xgbclassifier__reg_lambda': [0.4, 0.5]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_neg_mse", mean_score)
            mlflow.log_metric("std_neg_mse", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    #store an evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.5

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict = True)
    test_report = classification_report(ytest, y_pred_test, output_dict = True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report['1']["precision"],
        "train_recall": train_report['1']["recall"],
        "train_f1-score": train_report['1']["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report['1']["precision"],
        "test_recall": test_report['1']["recall"],
        "test_f1-score": test_report['1']["f1-score"]
    })

    # Save the model locally
    model_path = "best_tourism_package_taker_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    # Use a distinct repo_id for the model to avoid conflict with dataset repo
    repo_id_model = "verma89/tourismprjtt-model" # Changed repo_id here
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id_model, repo_type=repo_type)
        print(f"Model repository '{repo_id_model}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repository '{repo_id_model}' not found. Creating new model repository...")
        # Explicitly pass the token to create_repo
        create_repo(repo_id=repo_id_model, repo_type=repo_type, private=False, token=hf_token_val)
        print(f"Model repository '{repo_id_model}' created.")

    api.upload_file(
        path_or_fileobj="best_tourism_package_taker_v1.joblib",
        path_in_repo="best_tourism_package_taker_v1.joblib",
        repo_id=repo_id_model,
        repo_type=repo_type,
    )
