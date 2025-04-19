#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import base64
import pickle
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Import your preprocessing & helper from main.py
from main import _preprocess, fig_to_base64

# ---- App setup ----
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost"}})

# ---- Logging ----
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ---- Load model + encoders ----
try:
    with open('lgbm_model.pkl', 'rb') as f:
        MODEL = pickle.load(f)
        logging.info(f"Loaded LightGBM model with classes: {MODEL.classes_.tolist()}")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

try:
    with open('encoders.pkl', 'rb') as f:
        ENCODERS = pickle.load(f)
        logging.info("Loaded encoders.pkl")
except Exception as e:
    logging.error(f"Failed to load encoders: {e}")
    raise

# ---- Reconstruct the “numeric → string” mappings from main._preprocess ----
_REVERSE_MAPPINGS = {
    'sex': {'female': 0, 'male': 1},
    'chest_pain_type': {'typical angina': 0, 'atypical angina': 1, 'non-anginal pain': 2, 'asymptomatic': 3},
    'fasting_blood_sugar': {'lower than 120mg/ml': 0, 'greater than 120mg/ml': 1},
    'resting_electrocardiogram': {'normal': 0, 'ST-T wave abnormality': 1, 'left ventricular hypertrophy': 2},
    'exercise_induced_angina': {'no': 0, 'yes': 1},
    'st_slope': {'upsloping': 0, 'flat': 1, 'downsloping': 2},
    'thalassemia': {'fixed defect': 1, 'normal': 2, 'reversible defect': 3}
}

# These are the raw numeric column names that _preprocess expects:
_NUMERIC_COLUMNS = {
    'age': 'age',
    'resting_blood_pressure': 'trestbps',
    'cholesterol': 'chol',
    'max_heart_rate_achieved': 'thalach',
    'st_depression': 'oldpeak',
    'num_major_vessels': 'ca'
}

def preprocess_input(user_input: dict) -> pd.DataFrame:
    """
    1) Map form-friendly JSON (string categories) back into the raw numeric codes,
       in the exact DataFrame shape _preprocess expects.
    2) Call _preprocess(...) to get the cleaned, categorized DataFrame.
    """
    # 1a) Check presence of every field
    required = (
        list(_NUMERIC_COLUMNS.keys()) +
        list(_REVERSE_MAPPINGS.keys())
    )
    missing = [f for f in required if f not in user_input]
    if missing:
        raise ValueError(f"Missing inputs for: {missing}")

    # 1b) Build one-row raw DataFrame
    raw = {}
    # Numeric pass-through:
    for form_name, raw_col in _NUMERIC_COLUMNS.items():
        raw[raw_col] = user_input[form_name]
    # Categorical invert:
    for form_name, inv_map in _REVERSE_MAPPINGS.items():
        raw[form_name] = inv_map.get(user_input[form_name], None)
        if raw[form_name] is None:
            valid_options = list(inv_map.keys())
            raise ValueError(f"Invalid category for '{form_name}': {user_input[form_name]}. Valid options: {valid_options}")

    df_raw = pd.DataFrame([raw])
    logging.debug(f"Raw DataFrame for _preprocess(): {df_raw.to_dict(orient='records')}")

    # 2) Call your existing _preprocess
    df_clean = _preprocess(df_raw)
    if df_clean.empty:
        raise ValueError("Preprocessing resulted in empty DataFrame")

    # Reorder columns to match model's expected order
    expected_columns = MODEL.feature_name_
    if set(df_clean.columns) != set(expected_columns):
        raise ValueError(f"Column mismatch: expected {expected_columns}, got {df_clean.columns.tolist()}")
    df_clean = df_clean[expected_columns]

    logging.debug(f"Output of _preprocess(): cols={df_clean.columns.tolist()}, dtypes={df_clean.dtypes.to_dict()}")

    return df_clean

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_in = request.get_json(force=True)
        logging.debug(f"Received JSON: {data_in}")

        # 1) Run through _preprocess to replicate exactly
        df = preprocess_input(data_in)

        # 2) Now apply your saved LabelEncoders
        for col, le in ENCODERS.items():
            try:
                df[col] = le.transform(df[col].astype(str))
                logging.debug(f"Encoded {col} with LabelEncoder, values={df[col].tolist()}")
            except Exception as e:
                raise ValueError(f"Encoding failed for '{col}': {e}")

        # 3) Predict
        # predict_proba[:, 1] is probability of class 1 (diseased)
        prob = float(MODEL.predict_proba(df)[0, 1])
        logging.debug(f"Prediction probabilities: healthy={MODEL.predict_proba(df)[0, 0]:.4f}, diseased={prob:.4f}")
        risk = "Risque élevé" if prob >= 0.5 else "Risque faible"

        # 4) SHAP bar-plot
        explainer = shap.TreeExplainer(MODEL)
        shap_values = explainer.shap_values(df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (diseased)
        shap.summary_plot(shap_values, df, feature_names=df.columns,
                          plot_type="bar", show=False)
        fig = plt.gcf()
        shap_img = fig_to_base64(fig)

        return jsonify({
            "probability": prob,
            "risk_category": risk,
            "shap_plot": shap_img
        })

    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        return jsonify(error=str(ve)), 400

    except Exception as ex:
        logging.exception("Unhandled error during prediction")
        return jsonify(error="Erreur interne du serveur"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)