# AI-Assisted Coding Disclosure Notes

Answers below match the laboratory required fields and reflect what was actually done on this project.

## 1. AI Tool Used

**Google Gemini**

(No Copilot or ChatGPT were used for this Laboratory 03 web app.)

## 2. Purpose

Gemini was used to support:

- Planning a local Flask web app that loads the exported Naive Bayes `joblib` pipeline
- Structuring HTML/CSS/JavaScript for a clearer student input form
- Configuring feature fields (YAML) so the form matches the trained 30-feature set without G1/G2/G3
- Improving validation messages and Clear/empty-form behavior
- Drafting README / setup-run-exit wording and technical documentation narratives

## 3. Sample Prompts (at least three actual prompts used)

1. “Plan a local Flask web app that loads my Lab Naive Bayes joblib pipeline for student AtRisk prediction, with a form for all 30 features (no G1/G2/G3), Predict and Clear, and README install/run/exit steps.”

2. “Make the student form configurable with YAML, group the fields so they are easier to understand, remove default filled values so Clear really empties the form, and only keep a Surprise me random filler so users cannot easily game Pass vs At risk.”

3. “Help me write the technical documentation sections for model export, Flask integration, offline verification, and AI-assisted coding disclosure based on the Laboratory handout requirements.”

Related follow-up prompts also covered UI polish (About / How to use), post-prediction insight wording, and packaging the project zip for submission.

## 4. AI-Generated Assistance

AI assistance contributed to drafts or guidance for:

- Flask app layout (`app.py` routes, loading the pipeline with Joblib)
- HTML template structure and CSS styling for the form and result panel
- JavaScript for Surprise me + client-side missing-field checks
- Feature/app YAML configuration ideas
- Documentation wording (specification, setup guide, disclosure narrative)

Student-authored / verified work that was not left as raw AI output:

- Reusing the actual exported Naive Bayes pipeline from the prior Colab notebook
- Matching feature order and preprocessing to the trained Pipeline
- Choosing Naive Bayes only (best F1 from the prior evaluation)
- Local testing of Pass / At risk / mixed-probability / missing-field cases
- Final edits to remove grading-game hints and Lab 02 wording from the UI

## 5. Verification

AI suggestions were reviewed and modified before keeping them. Verification included:

- Installing dependencies in a local `.venv` and running `python app.py`
- Confirming the app loads `model/naive_bayes_pipeline.joblib` (Colab export aligned to scikit-learn 1.6.1)
- Testing at least three successful predictions with different inputs
- Testing validation when required fields are empty
- Confirming predictions change with inputs and are not hard-coded
- Checking that Clear resets the form to blank fields

## 6. Screenshots of AI interactions

Chat screenshots of the Gemini sessions were not retained during development, so they are not attached as separate figures. The working application screenshots elsewhere in the documentation (environment, UI, live predictions, validation) are the evidence that the assisted code was tested and runs locally with the saved model.

## Short statement for the PDF

AI-assisted coding with Google Gemini was used for Flask/web structure ideas, UI wording, configuration, and documentation drafts. At least three development prompts are listed above. Assisted parts were reviewed, adapted to the real feature set and exported Naive Bayes pipeline, and verified through local run/predict/validation tests. Predictions are produced by the local joblib model, not hard-coded results.
