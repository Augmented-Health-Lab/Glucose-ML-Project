from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


'''
Runs Logistic Regression, Random Forest, and XGBoost models
using the feature calculations.
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

diabetes_groups = sorted(feature_data["diabetes_type"].unique())


def save_model_outputs(model_name, y_validate, valid_pred, y_test, test_pred, diabetes_groups):
    out_folder = Path(model_name)
    out_folder.mkdir(parents=True, exist_ok=True)

    test_confusion_matrix = confusion_matrix(y_test, test_pred, labels=diabetes_groups)

    cm_df = pd.DataFrame(
        test_confusion_matrix,
        index=[f"true_{c}" for c in diabetes_groups],
        columns=[f"pred_{c}" for c in diabetes_groups],
    )
    cm_df.to_csv(out_folder / "test_confusion_matrix.csv")

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
    score_df.to_csv(out_folder / "test_scores.csv", index=False)


#Log regression
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

save_model_outputs(
    "Logistic-regression-results",
    y_validate, valid_pred,
    y_test, test_pred,
    diabetes_groups
)


#Random Forest
random_forest = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )),
])

random_forest.fit(x_train, y_train)
valid_pred = random_forest.predict(x_validate)
test_pred = random_forest.predict(x_test)

save_model_outputs(
    "Random-forest-results",
    y_validate, valid_pred,
    y_test, test_pred,
    diabetes_groups
)



# XGBoost
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_validate_encoded = label_encoder.transform(y_validate)
y_test_encoded = label_encoder.transform(y_test)

xgboost_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        eval_metric="mlogloss",
        num_class=len(label_encoder.classes_),
        random_state=seed,
        n_jobs=-1,
    )),
])

xgboost_model.fit(x_train, y_train_encoded)

valid_pred_encoded = xgboost_model.predict(x_validate)
test_pred_encoded = xgboost_model.predict(x_test)

valid_pred = label_encoder.inverse_transform(valid_pred_encoded)
test_pred = label_encoder.inverse_transform(test_pred_encoded)

save_model_outputs(
    "XGBoost-results",
    y_validate, valid_pred,
    y_test, test_pred,
    diabetes_groups
)

print("Done!")