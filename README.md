# Student At-Risk Prediction Web App

MCAIPAT Laboratory 03 — local Flask prototype that loads the Laboratory 02 **Naive Bayes** pipeline and predicts whether a Portuguese secondary-school student is **at risk of failing** (`AtRisk = 1` when final grade G3 &lt; 10).

Predictions run **offline** from a local `.joblib` file. No Google Colab and no cloud prediction API are required at runtime.

## Stack

- Python 3.12+
- Flask (backend)
- HTML / CSS / JavaScript (UI)
- scikit-learn + joblib (saved pipeline)
- PyYAML (configurable form and app settings)

## Project layout

```
Gabilla_Cyrus_ML_WebApp/
  app.py
  requirements.txt
  config/app_config.yaml
  config/features.yaml
  model/naive_bayes_pipeline.joblib
  model/MODEL_INFO.txt
  model/feature_order.json
  data/student_performance.csv
  scripts/export_model.py
  scripts/colab_export_cell.py
  templates/index.html
  static/css/style.css
  static/js/app.js
  docs/SPECIFICATION.md
  docs/SETUP_RUN_EXIT.md
  docs/AI_DISCLOSURE_NOTES.md
```

## Quick start

```bash
cd Gabilla_Cyrus_ML_WebApp
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/export_model.py     # only if model/ is missing
python app.py
```

Open **http://127.0.0.1:5000/**

Stop the server with **Ctrl+C** in the terminal.

More detail: [docs/SETUP_RUN_EXIT.md](docs/SETUP_RUN_EXIT.md)

## Configuration

| File | Purpose |
|------|---------|
| `config/app_config.yaml` | Title, description, model path, host/port, result labels |
| `config/features.yaml` | Form fields, labels, choices, min/max validation |

After changing training data or features, re-run `python scripts/export_model.py` and keep `features.yaml` aligned with the new feature set.

## Model

Selected algorithm from Lab 02: **Gaussian Naive Bayes** (best F1 / recall for the rare at-risk class).

The saved artifact is a full **Pipeline** (`StandardScaler` + `OneHotEncoder` + `GaussianNB`) so the web app applies the same preprocessing used in training.

Colab export snippet for assignment screenshots: run `python scripts/colab_export_cell.py` and paste the printed cell into Colab.

## Author

Cyrus Gabilla — MCAIPAT Laboratory 03
