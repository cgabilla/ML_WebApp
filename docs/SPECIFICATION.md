# Specification — Student At-Risk Prediction Web App

## 1. Purpose

Provide a local web interface for the Laboratory 02 supervised learning model so a user can enter student attributes and receive an **AtRisk** prediction without Google Colab.

## 2. Problem statement

| Item | Value |
|------|--------|
| Dataset | UCI Student Performance — Portuguese course (`student_performance.csv`) |
| Problem type | Binary classification |
| Target | `AtRisk` = 1 if `G3 < 10`, else 0 |
| Positive meaning | At risk of failing Portuguese |
| Negative meaning | Pass |
| Excluded inputs | `G1`, `G2`, `G3` (early-warning setup; Lab 02 / Cortez & Silva setup C) |
| Selected model | Gaussian Naive Bayes full Pipeline (Lab 02 best F1 for AtRisk) |

## 3. Functional requirements

1. Display application title and short description.
2. Show an input form for all 30 training features.
3. Validate required fields and numeric ranges / categorical choices.
4. On submit, load the local joblib Pipeline and generate a prediction.
5. Display Pass / At risk, predicted class, and class probabilities.
6. Provide Clear / Reset to start a new prediction.
7. Run fully offline (no Colab, no external prediction API).

## 4. Non-functional requirements

- Configurable via YAML (no hard-coded form field list in business logic beyond loading config).
- Reproducible install via `requirements.txt` and `docs/SETUP_RUN_EXIT.md`.
- Same preprocessing at inference as at training (saved Pipeline).

## 5. System architecture

```
Browser form
    |  POST /predict
Flask (app.py)
    |  validate using features.yaml
    |  build one-row DataFrame (feature_order.json)
joblib Pipeline
    |  StandardScaler + OneHotEncoder
    |  GaussianNB
Result panel (Pass / At risk + probabilities)
```

## 6. Routes

| Method | Path | Behavior |
|--------|------|----------|
| GET | `/` | Render empty/default form |
| POST | `/predict` | Validate, predict, re-render with result or errors |
| POST | `/reset` | Flash message and redirect to `/` |

## 7. Configuration files

### `config/app_config.yaml`

- App title, subtitle, description, author
- Model path and whether to show probabilities
- Result labels and server host/port

### `config/features.yaml`

For each feature:

- `name`, `label`, `type` (`categorical` \| `integer`)
- `required`, `default`, optional `help`
- categorical: `choices[{value,label}]`
- integer: `min`, `max`

## 8. Model artifact

| File | Role |
|------|------|
| `model/naive_bayes_pipeline.joblib` | Fitted sklearn Pipeline |
| `model/feature_order.json` | Column order + metadata |
| `model/MODEL_INFO.txt` | Versions and test metrics |

Export locally:

```bash
python scripts/export_model.py
```

Training parity with Lab 02 notebook: `random_state=42`, `stratify=y`, `test_size=0.20`, drop grades, `GaussianNB()`.

## 9. Security notes

- Joblib/pickle can execute code on load; only load model files you created or trust.
- This prototype is for local coursework, not public internet deployment.
- `app.secret_key` is for local flash messages only.

## 10. Out of scope

- User authentication
- Database persistence of predictions
- Retraining from the web UI
- Cloud hosting
