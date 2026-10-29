
from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "ai4i2020.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

RANDOM_STATE = 42
TARGET = "Machine failure"

EXCLUDED_COLUMNS = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def make_preprocessor(numeric_features, categorical_features,
                      scale_numeric=False):
    """Build preprocessing that is fitted only on training data."""

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def select_threshold(y_true, probabilities):
    """
    Select a threshold using validation data only.
    Maximize F1; break ties by choosing the higher threshold.
    """

    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_f1 = np.max(f1_values)
    candidates = np.flatnonzero(
        np.isclose(f1_values, best_f1)
    )

    best_index = candidates[
        np.argmax(thresholds[candidates])
    ]

    return float(thresholds[best_index])


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # --------------------------------------------------
    # 1. Load and validate data
    # --------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        TARGET,
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {sorted(missing_columns)}"
        )

    if df[TARGET].isnull().any():
        raise ValueError("Target column contains missing values.")

    X = df.drop(
        columns=[c for c in EXCLUDED_COLUMNS if c in df.columns]
    )
    y = df[TARGET].astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError("Target must contain both 0 and 1.")

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    print("=" * 65)
    print("EDGEOPS AI — PREDICTIVE MAINTENANCE TRAINING")
    print("=" * 65)
    print(f"Dataset shape: {df.shape}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    print("\nTarget distribution:")
    print(y.value_counts().sort_index())
    print(f"\nMissing values in raw dataset: {df.isnull().sum().sum()}")

    # Save dataset summary and class distribution.

    summary = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "model_feature_count": int(X.shape[1]),
        "failure_count": int((y == 1).sum()),
        "normal_count": int((y == 0).sum()),
        "failure_percentage": float((y == 1).mean() * 100),
        "missing_values": int(df.isnull().sum().sum()),
        "model_features": X.columns.tolist(),
    }

    with open(RESULTS_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, hue=y, legend=False)
    plt.xticks([0, 1], ["Normal", "Failure"])
    plt.title("AI4I 2020 — Machine Failure Distribution")
    plt.xlabel("Machine status")
    plt.ylabel("Number of observations")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "target_distribution.png", dpi=200
    )
    plt.close()

    # --------------------------------------------------
    # 2. Split data before preprocessing
    # --------------------------------------------------

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Validation split is taken only from the training partition.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    print("\nSplit sizes:")
    print(f"Training:   {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test:       {len(X_test)}")

    # --------------------------------------------------
    # 3. Define exactly three models
    # --------------------------------------------------

    rf_pipeline = Pipeline([
        (
            "preprocessor",
            make_preprocessor(
                numeric_features, categorical_features
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])

    xgb_pipeline = Pipeline([
        (
            "preprocessor",
            make_preprocessor(
                numeric_features, categorical_features
            ),
        ),
        (
            "classifier",
            XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=(
                    (y_train == 0).sum()
                    / max((y_train == 1).sum(), 1)
                ),
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
        ),
    ])

    lr_pipeline = Pipeline([
        (
            "preprocessor",
            make_preprocessor(
                numeric_features,
                categorical_features,
                scale_numeric=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    model_templates = {
        "Random Forest": rf_pipeline,
        "XGBoost": xgb_pipeline,
        "Logistic Regression": lr_pipeline,
    }

    # --------------------------------------------------
    # 4. Fit, select validation thresholds, evaluate test
    # --------------------------------------------------

    metrics = []
    fitted_models = {}
    test_predictions = {}
    test_probabilities = {}

    for name, template in model_templates.items():
        print(f"\n{'=' * 65}")
        print(f"Training: {name}")
        print("=" * 65)

        # Fit preprocessing and model on training data only.
        model = clone(template)
        model.fit(X_train, y_train)

        # Choose threshold using validation data only.
        val_probabilities = model.predict_proba(X_val)[:, 1]
        threshold = select_threshold(y_val, val_probabilities)

        # Evaluate once on the untouched test set.
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(y_test, predictions, zero_division=0)
        pr_auc = average_precision_score(y_test, probabilities)

        print(f"Validation-selected threshold: {threshold:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"PR-AUC:    {pr_auc:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Normal", "Failure"],
                zero_division=0,
            )
        )

        metrics.append({
            "model": name,
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "pr_auc": pr_auc,
        })

        fitted_models[name] = model
        test_predictions[name] = predictions
        test_probabilities[name] = probabilities

        # Save fitted model pipeline and threshold.
        filename = name.lower().replace(" ", "_")
        joblib.dump(
            {
                "pipeline": model,
                "threshold": threshold,
                "features": X.columns.tolist(),
            },
            MODELS_DIR / f"{filename}.joblib",
        )

        with open(
            RESULTS_DIR / f"{filename}_report.txt", "w"
        ) as f:
            f.write(
                classification_report(
                    y_test,
                    predictions,
                    labels=[0, 1],
                    target_names=["Normal", "Failure"],
                    zero_division=0,
                )
            )
            f.write(f"\nValidation-selected threshold: {threshold:.6f}\n")

        # Save confusion matrix.
        cm = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        )

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Failure"],
            yticklabels=["Normal", "Failure"],
        )
        plt.title(f"{name} — Confusion Matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(
            RESULTS_DIR / f"{filename}_confusion_matrix.png",
            dpi=200,
        )
        plt.close()

        # Save feature importance for tree models.
        if name in ("Random Forest", "XGBoost"):
            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]

            feature_names = preprocessor.get_feature_names_out()
            importance = classifier.feature_importances_

            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importance,
            }).sort_values("importance", ascending=False)

            importance_df.to_csv(
                RESULTS_DIR / f"{filename}_feature_importance.csv",
                index=False,
            )

            top_features = importance_df.head(12).sort_values(
                "importance"
            )

            plt.figure(figsize=(9, 6))
            plt.barh(
                top_features["feature"],
                top_features["importance"],
            )
            plt.title(f"{name} — Top Feature Importances")
            plt.xlabel("Feature importance")
            plt.tight_layout()
            plt.savefig(
                RESULTS_DIR / f"{filename}_feature_importance.png",
                dpi=200,
            )
            plt.close()

    # --------------------------------------------------
    # 5. Save comparison table and chart
    # --------------------------------------------------

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(
        RESULTS_DIR / "model_comparison.csv", index=False
    )

    print("\nFINAL MODEL COMPARISON")
    print(metrics_df.round(4).to_string(index=False))

    metric_columns = [
        "accuracy", "precision", "recall", "f1_score", "pr_auc"
    ]

    chart_data = metrics_df.set_index("model")[metric_columns]
    chart_data.plot(kind="bar", figsize=(11, 6))
    plt.title("EdgeOps AI — Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "model_comparison.png", dpi=200
    )
    plt.close()

    # --------------------------------------------------
    # 6. Save example sensor predictions
    # --------------------------------------------------

    # Use real held-out test observations as examples.
    examples = X_test.head(10).copy()
    examples.insert(
        0, "actual_failure", y_test.loc[examples.index].values
    )

    for name, model in fitted_models.items():
        filename = name.lower().replace(" ", "_")
        probabilities = model.predict_proba(
            X_test.loc[examples.index]
        )[:, 1]

        threshold = next(
            item["threshold"]
            for item in metrics
            if item["model"] == name
        )

        examples[f"{filename}_failure_probability"] = probabilities
        examples[f"{filename}_prediction"] = (
            probabilities >= threshold
        ).astype(int)

    examples.to_csv(
        RESULTS_DIR / "example_predictions.csv", index=False
    )

    print("\nExample predictions saved.")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Results saved in: {RESULTS_DIR}")
    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    main()
