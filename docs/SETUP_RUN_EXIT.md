# Setup, Run, and Exit

Instructions to install this web app on a clean machine, run it locally, and stop it safely.

## 1. Prerequisites

- Python **3.12+** (3.10+ may work; project was verified on 3.12)
- `pip` and `venv`
- A modern browser (Chrome, Firefox, Edge)

Check Python:

```bash
python3 --version
```

## 2. Install (fresh environment)

```bash
cd Gabilla_Cyrus_ML_WebApp
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Confirm packages:

```bash
python -c "import flask, sklearn, joblib, yaml, pandas; print('ok', sklearn.__version__)"
```

## 3. Ensure the model file exists

Runtime needs:

- `model/naive_bayes_pipeline.joblib`
- `model/feature_order.json`

If missing (or after changing data), regenerate:

```bash
python scripts/export_model.py
```

This retrains the Lab 02–parity Naive Bayes pipeline on `data/student_performance.csv` and writes the joblib file.

Optional Colab path for assignment Activity 2 screenshots:

1. Open the Lab 02 Colab notebook.
2. Run `python scripts/colab_export_cell.py` locally and paste the printed cell into Colab after training.
3. Download `trained_model.joblib`.
4. Copy it to `model/naive_bayes_pipeline.joblib` (keep the same scikit-learn major/minor version when possible).

## 4. Run the web application

```bash
source .venv/bin/activate
python app.py
```

Expected terminal message:

```text
Loaded model: .../model/naive_bayes_pipeline.joblib
Open http://127.0.0.1:5000/  |  Press Ctrl+C to stop
```

In the browser open:

**http://127.0.0.1:5000/**

Host and port can be changed in `config/app_config.yaml` under `server:`.

## 5. Use the app

1. Fill the student form (all fields required).
2. Click **Predict**.
3. Read the result panel (**Pass** or **At risk**, plus probabilities).
4. Click **Clear / Reset** to start another profile.

Validation: leave a field empty or enter an out-of-range number (for example age `99`) and submit — the page shows error messages and does not invent a prediction.

## 6. Exit / stop

In the terminal where Flask is running:

1. Press **Ctrl+C** once.
2. Wait until the process ends and the prompt returns.
3. Optionally deactivate the virtual environment:

```bash
deactivate
```

Closing the browser tab does **not** stop the server. Always stop Flask with Ctrl+C.

## 7. Reinstall checklist (other computer)

1. Copy the whole `Gabilla_Cyrus_ML_WebApp` folder (include `model/` and `data/`).
2. Create a new `.venv` and `pip install -r requirements.txt`.
3. Prefer matching **scikit-learn 1.5.x** with the saved joblib (see `model/MODEL_INFO.txt`).
4. Run `python app.py`.
5. Stop with Ctrl+C.

## 8. Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Activate `.venv` and reinstall `requirements.txt` |
| Joblib / sklearn load error | Re-run `python scripts/export_model.py` on this machine |
| Port 5000 in use | Change `server.port` in `config/app_config.yaml` |
| Form validation errors | Check min/max and allowed choices in `config/features.yaml` |
