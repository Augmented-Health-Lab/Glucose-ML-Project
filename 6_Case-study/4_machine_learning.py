from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



'''
Runs a Logistic Regression model using the feature calculations.
'''

seed = 20
feature_data = pd.read_csv("feature_calcs.csv", dtype={"person_id": str})
non_features = ["person_id", "diabetes_type", "split_assignment", "dataset"]
features = [feat for feat in feature_data.columns if feat not in non_features]

train_data = feature_data[feature_data["split_assignment"] == "train"].copy()
test_data = feature_data[feature_data["split_assignment"] == "test"].copy()
validate_data = feature_data[feature_data["split_assignment"] == "validate"].copy() 

x_train = train_data[features].copy()
y_train = train_data["diabetes_type"].copy()

x_test = test_data[features].copy()
y_test = test_data["diabetes_type"].copy()

x_validate = validate_data[features].copy()
y_validate = validate_data["diabetes_type"].copy()



logistic_regression = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",
        random_state=seed,
    )),
])
logistic_regression.fit(x_train, y_train)
valid_pred = logistic_regression.predict(x_validate)
test_pred = logistic_regression.predict(x_test)

diabetes_groups = sorted(feature_data["diabetes_type"].unique())
test_confusion_matrix = confusion_matrix(y_test, test_pred, labels=diabetes_groups)

out_folder = Path("Logistic-regression-results") 
out_folder.mkdir(parents=True, exist_ok=True)

cm_df = pd.DataFrame(test_confusion_matrix,
    index=[f"true_{c}" for c in diabetes_groups],
    columns=[f"pred_{c}" for c in diabetes_groups],
    )
cm_df.to_csv("Logistic-regression-results/test_confusion_matrix.csv")

accuracy_test = accuracy_score(y_test, test_pred)
macro_f1_test = f1_score(y_test, test_pred, average="macro")
balanced_acc_test = balanced_accuracy_score(y_test, test_pred)

test_entry = pd.DataFrame([{
    "data_type": "test",
    "accuracy": accuracy_test,
    "macro_f1": macro_f1_test,
    "balanced_accuracy": balanced_acc_test
}])

accuracy_validate = accuracy_score(y_validate, valid_pred)
macro_f1_validate = f1_score(y_validate, valid_pred, average="macro")
balanced_acc_validate = balanced_accuracy_score(y_validate, valid_pred)

validate_entry = pd.DataFrame([{
    "data_type": "validate",
    "accuracy": accuracy_validate,
    "macro_f1": macro_f1_validate,
    "balanced_accuracy": balanced_acc_validate
}])

score_df = pd.concat([test_entry, validate_entry], ignore_index=True)

score_df.to_csv("Logistic-regression-results/test_scores.csv", index=False)

print("Done!")

