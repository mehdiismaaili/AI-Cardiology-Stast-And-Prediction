import logging
import io
import base64
import mysql.connector
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(filename='charts.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection
from config import get_db_connection

# Consistent styling
MYPAL = ['#FC05FB', '#FEAEFE', '#FCD2FC', '#F3FEFA', '#B4FFE4', '#3FFEBA']
TARGET_PALETTE = {0: MYPAL[-1], 1: MYPAL[0]}  # 0: healthy, 1: diseased

# Feature lists
NUM_FEATS = ['age', 'cholesterol', 'resting_blood_pressure', 'max_heart_rate_achieved', 'st_depression', 'num_major_vessels']
BIN_FEATS = ['sex', 'fasting_blood_sugar', 'exercise_induced_angina', 'target']
NOM_FEATS = ['chest_pain_type', 'resting_electrocardiogram', 'st_slope', 'thalassemia']
CAT_FEATS = NOM_FEATS + BIN_FEATS

# -------------------- 1. Data Loading --------------------
def _load_raw() -> pd.DataFrame:
    """Pull the whole `heart_disease_stats` table as a DataFrame."""
    logging.debug("Attempting to load data from heart_disease_stats")
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM heart_disease_stats", conn)
            logging.info(f"Loaded {len(df)} rows from heart_disease_stats")
            if df.empty:
                logging.warning("heart_disease_stats table is empty")
            return df
    except mysql.connector.Error as e:
        logging.error(f"Database error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error while loading data: {e}")
        return pd.DataFrame()

# -------------------- 2. Preprocessing --------------------
def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans, renames, and maps categorical variables."""
    if df.empty:
        logging.warning("Empty DataFrame provided for preprocessing")
        return df

    # Filter invalid values
    if 'ca' in df.columns:
        df = df[df['ca'] < 4]
    if 'thal' in df.columns:
        df = df[df['thal'] > 0]

    # Rename columns
    col_map = {
        'cp': 'chest_pain_type',
        'trestbps': 'resting_blood_pressure',
        'chol': 'cholesterol',
        'fbs': 'fasting_blood_sugar',
        'restecg': 'resting_electrocardiogram',
        'thalach': 'max_heart_rate_achieved',
        'exang': 'exercise_induced_angina',
        'oldpeak': 'st_depression',
        'slope': 'st_slope',
        'ca': 'num_major_vessels',
        'thal': 'thalassemia'
    }
    rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    # Categorical mappings
    mappings = {
        'sex': {0: 'female', 1: 'male'},
        'chest_pain_type': {0: 'typical angina', 1: 'atypical angina', 2: 'non-anginal pain', 3: 'asymptomatic'},
        'fasting_blood_sugar': {0: 'lower than 120mg/ml', 1: 'greater than 120mg/ml'},
        'resting_electrocardiogram': {0: 'normal', 1: 'ST-T wave abnormality', 2: 'left ventricular hypertrophy'},
        'exercise_induced_angina': {0: 'no', 1: 'yes'},
        'st_slope': {0: 'upsloping', 1: 'flat', 2: 'downsloping'},
        'thalassemia': {1: 'fixed defect', 2: 'normal', 3: 'reversible defect'}
    }
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            if df[col].isna().any():
                logging.warning(f"NaN values in {col} after mapping; filling with mode")
                df[col] = df[col].fillna(df[col].mode()[0]).astype('category')
            else:
                df[col] = df[col].astype('category')

    # Ensure target is binary and invert labels (assuming database has 0=diseased, 1=healthy)
    if 'target' in df.columns:
        df['target'] = df['target'].astype(int)
        df = df[df['target'].isin([0, 1])]
        # Invert target: 0=healthy, 1=diseased
        df['target'] = 1 - df['target']
        logging.info(f"Target distribution after inversion: {df['target'].value_counts().to_dict()}")

    # Handle numeric columns
    for col in NUM_FEATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                logging.warning(f"NaN values in {col}; filling with median")
                df[col] = df[col].fillna(df[col].median())

    # Final NaN check
    if df.isna().any().any():
        logging.warning(f"NaN values remaining in DataFrame: {df.isna().sum()}")
        for col in df.columns:
            if df[col].isna().any():
                if col in NUM_FEATS:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

    logging.debug(f"Preprocessed DataFrame shape: {df.shape}, columns: {df.columns.tolist()}, dtypes: {df.dtypes.to_dict()}")
    return df

def _get_processed_data():
    """Loads and preprocesses data."""
    df_raw = _load_raw()
    return _preprocess(df_raw)

def _split_encode_data(df: pd.DataFrame, test_size=0.25, random_state=0):
    """Splits data and applies label encoding."""
    if df.empty or 'target' not in df.columns:
        logging.error("DataFrame is empty or missing 'target' column")
        raise ValueError("DataFrame is empty or missing 'target' column")

    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in CAT_FEATS:
        if col in df_encoded.columns:
            try:
                df_encoded[col] = label_encoder.fit_transform(df_encoded[col].astype(str))
                logging.debug(f"Encoded {col} with classes: {label_encoder.classes_.tolist()}")
            except Exception as e:
                logging.error(f"Error encoding {col}: {e}")
                raise ValueError(f"Failed to encode {col}")

    X = df_encoded.drop(columns=['target'], errors='ignore')
    y = df_encoded['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Ensure numeric data
    for col in X.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_val[col] = pd.to_numeric(X_val[col], errors='coerce')

    # Check for NaNs
    if X_train.isna().any().any() or X_val.isna().any().any():
        logging.warning("NaN values in X_train or X_val; imputing with median")
        X_train = X_train.fillna(X_train.median())
        X_val = X_val.fillna(X_val.median())

    # Convert to float
    X_train = X_train.astype(float)
    X_val = X_val.astype(float)

    # Final validation
    if not np.all(X_train.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        logging.error("Non-numeric columns in X_train after encoding")
        raise ValueError("Non-numeric columns in X_train")
    if not np.all(X_val.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        logging.error("Non-numeric columns in X_val after encoding")
        raise ValueError("Non-numeric columns in X_val")

    logging.debug(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, dtypes: {X_train.dtypes.tolist()}")
    return X_train, X_val, y_train, y_val, X.columns

# -------------------- 3. Figure Helper --------------------
def fig_to_base64(fig):
    """Converts a Matplotlib figure to a Base64-encoded PNG string."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
        buf.seek(0)
        payload = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return payload
    except Exception as e:
        logging.error(f"Error converting figure to base64: {e}")
        return ""