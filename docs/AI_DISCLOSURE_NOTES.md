# AI-Assisted Coding Disclosure Notes

Based on the actual Cursor chat used to build Laboratory 03.

## 1. AI Tool Used

**Cursor** (AI coding assistant in the editor)

## 2. Purpose

Cursor was used to plan and implement the Flask web app, improve the UI/validation, integrate the exported Naive Bayes `joblib` model, and draft the technical documentation / submission package.

## 3. Sample Prompts (from this chat — at least three)

These are real prompts from the Lab 03 build conversation:

1. “Our next laboratory is local web development based on our google collab supervised machine learning. I am not sure what is the best framework to use which is highly compatible with google collab training data model. I need you to plan for me and let me know what you need. it should be configurable and very easy to configure for the data. maybe a simple web dev and choose the best technology stack to implement it. i will attach the requirement. … as you design and implement, create a documentation as well so that there is step to reinstall this in other environment. how to run and how to exit. … select only the best model to consider, Naive bayes.”

2. “can you make it more pretty looking design and very easy to understand? align forms, and add some about and information. do not make text so robotic , make it as human friendly. maybe add a button to randomly generate input like very practical, and a random input that will tend to pass or failed or something in between. if this is not allowed in the requirements, then we cannot do it.”

3. “meaning the input is predictable? what can you suggest to enhance? it should be surprising the result. it seems that user knows how to input to make it pass. what if we do not give the user a hint? what do you think? and only suprise me only. if you think this is a better idea or you have a better idea then do it.”

Additional related prompts from the same chat (optional extras):

- “why there are defaults values, when you clear, it is not cleared. just tell the user that there are missing data to fill out if missing.”
- “let's us build this all, just let me know what you need so that i can give you what you need as we create, i will just paste the screenshot here”
- “After reading the requirements, ai disclosure requires questions to answer. Can you update the ai disclosure only based on our what we did … and create another version archive for submission”

## 4. AI-Generated Assistance

Assisted with Cursor:

- Flask + Joblib model-loading structure
- HTML/CSS/JS form, Surprise me, missing-field validation
- YAML configuration for features/app settings
- Documentation PDF narratives and packaging

Student-reviewed / finalized:

- Naive Bayes-only model from Colab export
- Feature order / preprocessing consistency
- UI changes (no outcome-hint buttons, empty Clear behavior)
- Live prediction and validation testing

## 5. Verification

- `python app.py` local run
- Model file loaded from `model/`
- 3 successful prediction tests + 1 missing-field validation test
- Confirmed predictions are from the joblib model (not hard-coded)

## 6. Appendix needed for prompts?

**No.** At least three prompts in the AI disclosure section are enough. An appendix is optional.

## 7. AI interaction screenshots

Three actual Cursor chat screenshots from this project are included:

- `screenshots/activity08-ai/01-ai-interaction-plan.png` — plan Flask web app + Naive Bayes
- `screenshots/activity08-ai/02-ai-interaction-ui.png` — prettier UI / random input helpers
- `screenshots/activity08-ai/03-ai-interaction-surprise.png` — Surprise me only / remove hints

These are embedded in the documentation PDF AI disclosure section.
