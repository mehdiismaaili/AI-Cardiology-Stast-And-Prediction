#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_export.py

Loads the cleaned dataset via main._get_processed_data(),
fits per-column LabelEncoders, trains a LightGBMClassifier,
and exports both the model and the encoders for your API.
"""

import logging
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

from main import _get_processed_data, CAT_FEATS

# ---- Logging ----
logging.basicConfig(
    filename='train.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ---- 1. Load & preprocess the full dataset ----
try:
    df = _get_processed_data()
    logging.info(f"Loaded and preprocessed dataset with {len(df)} rows")
except Exception:
    logging.exception("Failed to load or preprocess data")
    raise

# ---- 2. Identify categorical columns (exclude 'target') ----
cat_cols = [c for c in CAT_FEATS if c != 'target']
for c in cat_cols:
    if c not in df.columns:
        logging.error(f"Expected categorical column '{c}' not found in DataFrame")
        raise KeyError(f"Missing column: {c}")

# ---- 3. Fit LabelEncoder on each categorical column ----
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    logging.debug(f"Fitted LabelEncoder for '{col}', classes={le.classes_.tolist()}")

# ---- 4. Split into train/validation ----
X = df.drop(columns=['target'])
y = df['target']
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.25,
    random_state=0,
    stratify=y
)
logging.info(f"Split data: X_train={X_train.shape}, X_val={X_val.shape}, columns={X_train.columns.tolist()}, dtypes={X_train.dtypes.to_dict()}")

# ---- 5. Train the LightGBM classifier ----
model = LGBMClassifier(
    num_leaves=20,
    max_depth=5,
    min_data_in_leaf=80,
    random_state=0,
    verbose=-1
)
model.fit(X_train, y_train, categorical_feature=cat_cols)
logging.info(f"Trained LightGBM model with classes: {model.classes_.tolist()}")

# ---- 6. Save the trained model & encoders ----
try:
    with open('lgbm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        logging.info("Saved model to lgbm_model.pkl")
except Exception as e:
    logging.error(f"Failed to save model: {e}")
    raise

try:
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
        logging.info("Saved encoders to encoders.pkl")
except Exception as e:
    logging.error(f"Failed to save encoders: {e}")
    raise

# ---- 7. Quick validation score ----
val_acc = model.score(X_val, y_val)
val_preds = model.predict(X_val)
val_probs = model.predict_proba(X_val)[:, 1]
logging.info(f"Validation accuracy: {val_acc:.4f}")
logging.debug(f"Validation predictions sample: {val_preds[:5].tolist()}, probabilities: {val_probs[:5].tolist()}")
print(f"Training complete. Validation accuracy: {val_acc:.4f}")