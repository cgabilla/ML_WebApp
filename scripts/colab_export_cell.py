"""Pasteable Google Colab cell for Activity 2 model export screenshots.

Run this in the Lab 02 Colab notebook AFTER the Naive Bayes pipeline
(`trained["Naive Bayes"]`) has been fitted. It saves the full
preprocessing + GaussianNB Pipeline and downloads it.
"""

COLAB_CELL = r'''
# === Lab 03 Activity 2: export selected Naive Bayes pipeline ===
import joblib
from google.colab import files

# Prefer the already-trained Lab 02 winner
best_model = trained["Naive Bayes"]

joblib.dump(best_model, "trained_model.joblib")
print("Model successfully saved!")
print("File: trained_model.joblib")

files.download("trained_model.joblib")
'''

if __name__ == "__main__":
    print(COLAB_CELL)
