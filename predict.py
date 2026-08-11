"""
predict.py
-----------
The bridge between a raw URL string and a model prediction.

WHY THIS FILE EXISTS SEPARATELY FROM app.py:
Keeping prediction logic out of app.py means it can be tested on its
own (see test_prediction.py) without spinning up a Streamlit server,
and it means app.py only has to worry about UI — not model loading,
not feature ordering, not probability math. If you ever build a second
interface (a CLI tool, an API endpoint) you'd import this same file
instead of duplicating logic inside app.py.

Loads the trained Random Forest exactly once (module-level cache) so
repeated predictions in a running Streamlit session don't reload the
model from disk every time the user clicks "Analyze."
"""

import os
import joblib
import pandas as pd

from feature_extraction import extract_features, FEATURE_COLUMNS

# Flat layout: model file sits in the same directory as this script
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_random_forest.joblib")
_model = None


def _get_model():
    global _model
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Could not find {_MODEL_PATH}. Make sure Train.py has been run "
                f"and models_random_forest.joblib is in the same folder as predict.py."
            )
        _model = joblib.load(_MODEL_PATH)
    return _model


def predict_url(url: str) -> dict:
    """
    Runs the full pipeline on a single URL string:
    feature extraction -> model prediction -> structured result.

    Returns a dict with:
        label: "Phishing" or "Legitimate"
        confidence: float 0-100, confidence in the predicted class
        phishing_probability: float 0-100
        legitimate_probability: float 0-100
        features: dict of the 16 extracted feature values
        top_features: list of (feature_name, importance) tuples,
                       sorted by the model's global importance
    """
    model = _get_model()

    features = extract_features(url)
    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    proba = model.predict_proba(X)[0]  # [P(class=0, phishing), P(class=1, legitimate)]
    pred_class = model.predict(X)[0]

    phishing_prob = proba[0] * 100
    legit_prob = proba[1] * 100
    confidence = legit_prob if pred_class == 1 else phishing_prob

    importances = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "label": "Legitimate" if pred_class == 1 else "Phishing",
        "confidence": round(confidence, 1),
        "phishing_probability": round(phishing_prob, 1),
        "legitimate_probability": round(legit_prob, 1),
        "features": features,
        "top_features": importances[:5],
    }