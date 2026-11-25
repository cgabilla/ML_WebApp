#!/usr/bin/env python3
"""Train and export the Lab 02 Naive Bayes pipeline for offline web use.

Matches Gabilla_Cyrus_SupervisedML.ipynb:
  - Target: AtRisk = 1 if G3 < 10 else 0
  - Features: all columns except G1, G2, G3, AtRisk
  - Preprocess: StandardScaler (numeric) + OneHotEncoder (categorical)
  - Model: GaussianNB only (selected Lab 02 winner by F1)
  - Split: 80/20, random_state=42, stratify=y
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_performance.csv"
MODEL_PATH = ROOT / "model" / "naive_bayes_pipeline.joblib"
INFO_PATH = ROOT / "model" / "MODEL_INFO.txt"
META_PATH = ROOT / "model" / "feature_order.json"


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", GaussianNB()),
        ]
    )


def main() -> int:
    if not DATA_PATH.exists():
        print(f"Missing dataset: {DATA_PATH}", file=sys.stderr)
        return 1

    df_raw = pd.read_csv(DATA_PATH)
    df = df_raw.copy()
    df["AtRisk"] = (df["G3"] < 10).astype(int)
    feature_cols = [c for c in df.columns if c not in ["G1", "G2", "G3", "AtRisk"]]

    X = df[feature_cols].copy()
    y = df["AtRisk"].copy()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipe = build_pipeline(numeric_features, categorical_features)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_at_risk": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall_at_risk": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_at_risk": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    meta = {
        "feature_order": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": "AtRisk",
        "positive_class": 1,
        "positive_label": "At risk",
        "negative_label": "Pass",
        "model": "GaussianNB",
        "random_state": 42,
        "metrics_test": metrics,
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    INFO_PATH.write_text(
        "\n".join(
            [
                "Naive Bayes student AtRisk pipeline (Lab 02 parity)",
                f"python: {sys.version.split()[0]}",
                f"scikit-learn: {sklearn.__version__}",
                f"dataset: {DATA_PATH.name} ({len(df)} rows)",
                f"features: {len(feature_cols)} (dropped G1, G2, G3)",
                f"target: AtRisk = 1 if G3 < 10 else 0",
                f"model_file: {MODEL_PATH.name}",
                f"test_accuracy: {metrics['accuracy']}",
                f"test_precision_at_risk: {metrics['precision_at_risk']}",
                f"test_recall_at_risk: {metrics['recall_at_risk']}",
                f"test_f1_at_risk: {metrics['f1_at_risk']}",
                "Notes: Save the full Pipeline so the web app applies the same",
                "StandardScaler + OneHotEncoder fit during training.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved pipeline -> {MODEL_PATH}")
    print(f"Saved info     -> {INFO_PATH}")
    print(f"Saved meta     -> {META_PATH}")
    print("Test metrics:", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
