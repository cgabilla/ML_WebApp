#!/usr/bin/env python3
"""Flask web app: offline Naive Bayes AtRisk prediction (Lab 03)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from flask import Flask, flash, redirect, render_template, request, url_for

ROOT = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


APP_CONFIG = load_yaml(ROOT / "config" / "app_config.yaml")
FEATURES_CONFIG = load_yaml(ROOT / "config" / "features.yaml")
FEATURE_DEFS: list[dict[str, Any]] = FEATURES_CONFIG["features"]
FEATURE_BY_NAME = {f["name"]: f for f in FEATURE_DEFS}

# UI groupings for a clearer form layout (names must match features.yaml).
FEATURE_GROUPS: list[dict[str, Any]] = [
    {
        "id": "basics",
        "title": "Student basics",
        "blurb": "Who the student is and where they study.",
        "fields": ["school", "sex", "age", "address"],
    },
    {
        "id": "family",
        "title": "Family & support",
        "blurb": "Home situation and parent background.",
        "fields": [
            "famsize",
            "Pstatus",
            "Medu",
            "Fedu",
            "Mjob",
            "Fjob",
            "guardian",
            "famrel",
            "famsup",
        ],
    },
    {
        "id": "school",
        "title": "School life & habits",
        "blurb": "Study time, absences, and school-related choices.",
        "fields": [
            "reason",
            "traveltime",
            "studytime",
            "failures",
            "schoolsup",
            "paid",
            "activities",
            "nursery",
            "higher",
            "internet",
            "absences",
        ],
    },
    {
        "id": "daily",
        "title": "Daily life",
        "blurb": "Free time, friends, health, and routines.",
        "fields": ["romantic", "freetime", "goout", "Dalc", "Walc", "health"],
    },
]


def grouped_features() -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for group in FEATURE_GROUPS:
        fields = [FEATURE_BY_NAME[name] for name in group["fields"] if name in FEATURE_BY_NAME]
        groups.append({**group, "feature_defs": fields})
    return groups

MODEL_PATH = ROOT / APP_CONFIG["model"]["path"]
FEATURE_ORDER_PATH = ROOT / APP_CONFIG["model"]["feature_order_path"]

with FEATURE_ORDER_PATH.open(encoding="utf-8") as f:
    FEATURE_META = json.load(f)
FEATURE_ORDER: list[str] = FEATURE_META["feature_order"]

PIPELINE = joblib.load(MODEL_PATH)

app = Flask(__name__)
app.secret_key = "lab03-local-dev-only"


def empty_form_values() -> dict[str, Any]:
    """Blank form — no pre-filled defaults."""
    return {field["name"]: "" for field in FEATURE_DEFS}


def validate_and_parse(form) -> tuple[dict[str, Any], list[str]]:
    """Validate request form against features.yaml. Returns (values, errors)."""
    values: dict[str, Any] = {}
    errors: list[str] = []

    for field in FEATURE_DEFS:
        name = field["name"]
        raw = (form.get(name) or "").strip()
        label = field["label"]

        if field.get("required", True) and raw == "":
            errors.append(f"Missing: “{label}” still needs an answer.")
            values[name] = ""
            continue

        if field["type"] == "categorical":
            # Coerce YAML values to str (bare yes/no can become bools).
            allowed = {str(c["value"]) for c in field["choices"]}
            if raw not in allowed:
                errors.append(
                    f"“{label}” needs one of these options: {', '.join(sorted(allowed))}."
                )
            values[name] = raw
            continue

        if field["type"] == "integer":
            try:
                number = int(raw)
            except ValueError:
                errors.append(f"“{label}” should be a whole number.")
                values[name] = raw
                continue
            lo = field.get("min")
            hi = field.get("max")
            if lo is not None and number < lo:
                errors.append(f"“{label}” looks too low — use at least {lo}.")
            if hi is not None and number > hi:
                errors.append(f"“{label}” looks too high — use at most {hi}.")
            values[name] = number
            continue

        errors.append(f"Something went wrong with “{label}”.")
        values[name] = raw

    return values, errors


def values_to_dataframe(values: dict[str, Any]) -> pd.DataFrame:
    row = {name: values[name] for name in FEATURE_ORDER}
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def build_insights(values: dict[str, Any], is_at_risk: bool) -> list[str]:
    """Friendly profile highlights based on patterns from EDA / domain sense.

    These are explanations of what stood out in the answers — not a full
    mathematical breakdown of Naive Bayes internals.
    """
    insights: list[str] = []
    failures = int(values.get("failures") or 0)
    studytime = int(values.get("studytime") or 0)
    absences = int(values.get("absences") or 0)
    higher = str(values.get("higher") or "")
    medu = int(values.get("Medu") or 0)
    fedu = int(values.get("Fedu") or 0)
    goout = int(values.get("goout") or 0)
    dalc = int(values.get("Dalc") or 0)
    walc = int(values.get("Walc") or 0)

    if is_at_risk:
        if failures >= 2:
            insights.append(
                f"Past class failures are high ({failures}). In this dataset, that was the "
                "clearest warning sign linked to struggling later."
            )
        elif failures == 1:
            insights.append(
                "There is already one past failure on record, which often raises the chance "
                "of needing support."
            )
        if higher == "no":
            insights.append(
                "This profile says the student does not plan on higher education, which often "
                "shows up with weaker outcomes in the training data."
            )
        if studytime <= 1:
            insights.append(
                "Weekly study time is on the low side, so the model has less evidence of "
                "steady academic effort."
            )
        if absences >= 10:
            insights.append(
                f"Absences are fairly high ({absences}). Missing class can stack up with "
                "other risk factors."
            )
        if medu <= 1 and fedu <= 1:
            insights.append(
                "Parent education levels are low in this profile, which sometimes correlates "
                "with fewer academic supports at home."
            )
        if goout >= 4 or dalc >= 4 or walc >= 4:
            insights.append(
                "Social time or alcohol use looks high relative to many classmates, which can "
                "pull attention away from schoolwork."
            )
        if not insights:
            insights.append(
                "No single answer screams danger on its own — the model is reacting to the "
                "overall mix of background and school habits."
            )
        insights.append(
            "Takeaway: this looks closer to a student who may benefit from a check-in or "
            "extra support before finals."
        )
    else:
        if failures == 0:
            insights.append(
                "No past class failures — that is usually a reassuring signal in this dataset."
            )
        if higher == "yes":
            insights.append(
                "Wanting higher education is a positive sign the model often associates with "
                "students who keep going."
            )
        if studytime >= 3:
            insights.append(
                "Study time looks solid, which supports a steadier academic routine."
            )
        if absences <= 5:
            insights.append(
                "Absences are low, so the profile suggests more consistent class presence."
            )
        if medu >= 3 or fedu >= 3:
            insights.append(
                "At least one parent has stronger formal education, which can go with more "
                "academic support at home."
            )
        if not insights:
            insights.append(
                "The overall mix of answers looks closer to students who passed in the "
                "training data."
            )
        insights.append(
            "Takeaway: nothing here strongly points to early failure risk, though real life "
            "can still surprise us."
        )

    return insights


def predict_row(frame: pd.DataFrame, values: dict[str, Any] | None = None) -> dict[str, Any]:
    pred = int(PIPELINE.predict(frame)[0])
    labels = APP_CONFIG["labels"]
    label = labels["positive"] if pred == labels["positive_class"] else labels["negative"]
    is_at_risk = pred == labels["positive_class"]

    result: dict[str, Any] = {
        "class": pred,
        "label": label,
        "technical_label": "At risk" if is_at_risk else "Pass",
        "is_at_risk": is_at_risk,
    }

    if APP_CONFIG["model"].get("show_probability", True) and hasattr(PIPELINE, "predict_proba"):
        proba = PIPELINE.predict_proba(frame)[0]
        # classes_ order from the fitted classifier
        classes = list(PIPELINE.named_steps["model"].classes_)
        proba_map = {int(c): float(p) for c, p in zip(classes, proba)}
        result["probability_at_risk"] = round(proba_map.get(1, 0.0), 4)
        result["probability_pass"] = round(proba_map.get(0, 0.0), 4)

    if values is not None:
        result["insights"] = build_insights(values, is_at_risk)

    return result


def _page(**kwargs):
    return render_template(
        "index.html",
        app_config=APP_CONFIG,
        features=FEATURE_DEFS,
        feature_groups=grouped_features(),
        **kwargs,
    )


@app.route("/", methods=["GET"])
def index():
    return _page(values=empty_form_values(), result=None, errors=None)


@app.route("/predict", methods=["POST"])
def predict():
    values, errors = validate_and_parse(request.form)
    result = None

    if not errors:
        try:
            frame = values_to_dataframe(values)
            result = predict_row(frame, values)
        except Exception as exc:  # noqa: BLE001 - show friendly UI error
            errors = [
                "We couldn’t run the prediction with those answers. "
                f"Technical detail: {exc}"
            ]

    return _page(values=values, result=result, errors=errors)


@app.route("/reset", methods=["POST"])
def reset():
    flash("Form cleared. Fill in the missing fields when you’re ready.", "info")
    return _page(values=empty_form_values(), result=None, errors=None)


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    host = APP_CONFIG["server"].get("host", "127.0.0.1")
    port = int(APP_CONFIG["server"].get("port", 5000))
    debug = bool(APP_CONFIG["server"].get("debug", False))
    print(f"Loaded model: {MODEL_PATH}")
    print(f"Open http://{host}:{port}/  |  Press Ctrl+C to stop")
    app.run(host=host, port=port, debug=debug)
