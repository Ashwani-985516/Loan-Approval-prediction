"""Train and evaluate loan-approval classifiers with leakage-safe SMOTE."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "loan_approval_dataset.csv"
DEFAULT_MODEL = ROOT / "best_model.pkl"
DEFAULT_METRICS = ROOT / "model_metrics.csv"
DEFAULT_ROC_PLOT = ROOT / "roc_curve.png"
TARGET = "loan_status"
ID_COLUMNS = ("loan_id",)
RANDOM_STATE = 42


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the Kaggle CSV and normalize its columns and categorical values."""
    data = pd.read_csv(data_path, skipinitialspace=True)
    data.columns = (
        data.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    data = data.drop_duplicates().copy()

    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain a {TARGET!r} target column.")

    if not pd.api.types.is_numeric_dtype(data[TARGET]):
        target_values = data[TARGET].astype("string").str.strip().str.casefold()
        data[TARGET] = target_values.map({"approved": 1, "rejected": 0})
    if data[TARGET].isna().any():
        raise ValueError("loan_status contains missing or unsupported target labels.")

    category_maps = {
        "education": {"graduate": 1, "not graduate": 0},
        "self_employed": {"yes": 1, "no": 0},
    }
    for column, mapping in category_maps.items():
        if column in data.columns and not pd.api.types.is_numeric_dtype(data[column]):
            values = data[column].astype("string").str.strip().str.casefold()
            encoded = values.map(mapping)
            if encoded.isna().any():
                raise ValueError(f"{column} contains unsupported category values.")
            data[column] = encoded

    data[TARGET] = pd.to_numeric(data[TARGET], errors="raise").astype(int)
    return data


def build_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def build_pipeline(classifier: Any, scale: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.extend(
        [
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", classifier),
        ]
    )
    return Pipeline(steps)


def train_and_evaluate(
    data_path: Path = DEFAULT_DATA,
    model_path: Path = DEFAULT_MODEL,
    metrics_path: Path = DEFAULT_METRICS,
    roc_plot_path: Path = DEFAULT_ROC_PLOT,
) -> tuple[pd.DataFrame, Pipeline, pd.DataFrame, pd.Series, Any, Any]:
    """Compare models on CV training folds and an untouched stratified test set."""
    data = load_dataset(data_path)
    feature_columns = [
        column
        for column in data.columns
        if column != TARGET and column not in ID_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No model features remain after excluding target and ID columns.")

    X = data[feature_columns].apply(pd.to_numeric, errors="raise")
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    records = []
    fitted_models: dict[str, Pipeline] = {}
    for name, classifier in build_models().items():
        pipeline = build_pipeline(
            classifier=classifier,
            scale=name == "Logistic Regression",
        )
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        records.append(
            {
                "model": name,
                "cv_roc_auc_mean": cv_scores.mean(),
                "cv_roc_auc_std": cv_scores.std(),
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_roc_auc": roc_auc_score(y_test, y_prob),
                "test_mcc": matthews_corrcoef(y_test, y_pred),
            }
        )
        fitted_models[name] = pipeline

    results = pd.DataFrame(records).sort_values(
        "cv_roc_auc_mean", ascending=False
    ).reset_index(drop=True)
    best_name = str(results.loc[0, "model"])
    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(X_test)
    best_probabilities = best_model.predict_proba(X_test)[:, 1]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    roc_plot_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(best_model, model_file)
    results.to_csv(metrics_path, index=False)

    plt.figure(figsize=(9, 6))
    for name, fitted_model in fitted_models.items():
        probabilities = fitted_model.predict_proba(X_test)[:, 1]
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
        score = roc_auc_score(y_test, probabilities)
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name} (AUC = {score:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Loan Approval Model ROC Curves (Holdout Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=150)
    plt.close()

    print(f"Rows used after removing exact duplicates: {len(data)}")
    print(f"Excluded identifier columns: {', '.join(ID_COLUMNS)}")
    print("SMOTE is applied only inside training folds and the training split.")
    print(f"Selected by mean cross-validation ROC-AUC: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved ROC plot: {roc_plot_path}")
    return (
        results,
        best_model,
        X_test,
        y_test,
        best_predictions,
        best_probabilities,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--roc-output", type=Path, default=DEFAULT_ROC_PLOT)
    args = parser.parse_args()

    results, _, _, _, _, _ = train_and_evaluate(
        data_path=args.data,
        model_path=args.model_output,
        metrics_path=args.metrics_output,
        roc_plot_path=args.roc_output,
    )
    print("\nModel comparison:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
