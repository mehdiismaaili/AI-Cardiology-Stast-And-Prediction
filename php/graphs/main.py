
import logging
import warnings
import io
import base64
import mysql.connector
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    import shap
except ImportError:
    shap = None
try:
    import eli5
    from eli5.sklearn import PermutationImportance
except ImportError:
    eli5 = None
    PermutationImportance = None

warnings.filterwarnings("ignore")

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
        'thalassemia': {1: 'fixed defect', 2: 'normal', 3: 'reversable defect'}
    }
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            if df[col].isna().any():
                logging.warning(f"NaN values in {col} after mapping; filling with mode")
                df[col] = df[col].fillna(df[col].mode()[0]).astype('category')
            else:
                df[col] = df[col].astype('category')

    # Ensure target is binary
    if 'target' in df.columns:
        df['target'] = df['target'].astype(int)
        df = df[df['target'].isin([0, 1])]

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
        df = df.fillna(0)

    logging.debug(f"Preprocessed DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
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

# -------------------- 4. Graph Functions --------------------
def chart_target_distribution():
    """Plots distribution of the target variable."""
    df = _get_processed_data()
    if df.empty or 'target' not in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title('Erreur: Données indisponibles')
        return fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#F6F5F4')
    ax.set_facecolor('#F6F5F4')
    total = len(df)
    sns.countplot(x=df['target'], palette=MYPAL[1::4], ax=ax)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 3, f'{height/total*100:.1f}%', ha="center",
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))
    ax.set_title('Distribution de la Variable Cible', fontsize=20, y=1.05)
    ax.set_xlabel('État Cardiaque', fontsize=12)
    ax.set_ylabel('Effectif', fontsize=12)
    ax.set_xticklabels(['Sain (0)', 'Malade (1)'])
    sns.despine(right=True, offset=5, trim=True)
    plt.tight_layout()
    return fig_to_base64(fig)

def chart_numerical_distributions():
    """Plots KDE/count distributions of numerical features."""
    df = _get_processed_data()
    if df.empty or not any(f in df.columns for f in NUM_FEATS):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Données indisponibles')
        return fig_to_base64(fig)

    L = len(NUM_FEATS)
    ncol = 2
    nrow = int(np.ceil(L / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 14), facecolor='#F6F5F4')
    fig.subplots_adjust(top=0.92)
    axes = axes.flatten()

    for i, col in enumerate(NUM_FEATS):
        if col not in df.columns:
            continue
        ax = axes[i]
        ax.set_facecolor('#F6F5F4')
        if col == 'num_major_vessels':
            sns.countplot(data=df, x=col, hue='target', palette=MYPAL[1::4], ax=ax)
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2., height + 3, f'{int(height)}', ha="center",
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))
        else:
            sns.kdeplot(data=df, x=col, hue='target', multiple='stack', palette=MYPAL[1::4], ax=ax)
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=20)
        ax.set_ylabel('Densité' if col != 'num_major_vessels' else 'Effectif', fontsize=20)
        ax.legend(['Sain (0)', 'Malade (1)'], title='État')
        sns.despine(right=True, offset=0, trim=False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Distribution des Variables Numériques', fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig_to_base64(fig)

def chart_pairplot():
    """Generates a pairplot of numerical features."""
    df = _get_processed_data()
    if df.empty or 'target' not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Données indisponibles')
        return fig_to_base64(fig)

    feats = [f for f in ['age', 'cholesterol', 'resting_blood_pressure', 'max_heart_rate_achieved', 'st_depression', 'target'] if f in df.columns]
    if len(feats) <= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Pas assez de variables')
        return fig_to_base64(fig)

    g = sns.pairplot(df[feats], hue='target', corner=True, diag_kind='hist', palette=MYPAL[1::4])
    g.fig.suptitle('Pairplot des Variables Numériques', fontsize=24, y=1.02)
    plt.subplots_adjust(top=0.95)
    return fig_to_base64(g.fig)

def chart_regression_plots():
    """Plots regression plots of age vs other indicators."""
    df = _get_processed_data()
    if df.empty or 'target' not in df.columns or 'age' not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Données indisponibles')
        return fig_to_base64(fig)

    targets_y = ['cholesterol', 'max_heart_rate_achieved', 'resting_blood_pressure', 'st_depression']
    targets_y = [t for t in targets_y if t in df.columns]
    if not targets_y:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Aucune variable Y')
        return fig_to_base64(fig)

    fig, axes = plt.subplots(1, len(targets_y), figsize=(20, 4))
    if len(targets_y) == 1:
        axes = [axes]
    
    for ax, target_col in zip(axes, targets_y):
        sns.regplot(data=df[df['target'] == 1], x='age', y=target_col, ax=ax, color=MYPAL[0], label='Malade (1)')
        sns.regplot(data=df[df['target'] == 0], x='age', y=target_col, ax=ax, color=MYPAL[5], label='Sain (0)')
        ax.set_xlabel('Âge', fontsize=12)
        ax.set_ylabel(target_col.replace('_', ' ').title(), fontsize=12)
        ax.legend(title='État Cardiaque')
    
    fig.suptitle('Régressions Âge vs Indicateurs Médicaux', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig_to_base64(fig)

def chart_categorical_distributions():
    """Plots count plots for categorical features."""
    df = _get_processed_data()
    if df.empty or 'target' not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Données indisponibles')
        return fig_to_base64(fig)

    plot_feats = [f for f in CAT_FEATS[:-1] if f in df.columns]
    if not plot_feats:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Aucune variable catégorielle')
        return fig_to_base64(fig)

    L = len(plot_feats)
    ncol = 2
    nrow = int(np.ceil(L / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 24), facecolor='#F6F5F4')
    fig.subplots_adjust(top=0.92)
    axes = axes.flatten()

    for i, col in enumerate(plot_feats):
        ax = axes[i]
        ax.set_facecolor('#F6F5F4')
        sns.countplot(data=df, x=col, hue='target', palette=MYPAL[1::4], ax=ax)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3, f'{int(height)}', ha="center",
                    bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=20)
        ax.set_ylabel('Effectif', fontsize=20)
        if i == 0:
            ax.legend(['Sain (0)', 'Malade (1)'], title='État Cardiaque')
        else:
            ax.get_legend().remove()
        sns.despine(right=True, offset=0, trim=False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Distribution des Variables Catégorielles', fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig_to_base64(fig)

def chart_pearson_heatmap():
    """Plots Pearson correlation heatmap."""
    df = _get_processed_data()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Données indisponibles')
        return fig_to_base64(fig)

    num_df = df[[f for f in NUM_FEATS if f in df.columns]].select_dtypes(include=np.number)
    if num_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Aucune variable numérique')
        return fig_to_base64(fig)

    corr = num_df.corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, mask=mask, cmap=sns.color_palette(MYPAL, as_cmap=True), vmax=1.0, vmin=-1.0,
                center=0, annot=True, square=False, linewidths=.5, cbar_kws={"shrink": 0.75}, ax=ax)
    ax.set_title("Corrélation de Pearson (Variables Numériques)", fontsize=20, y=1.05)
    plt.tight_layout()
    return fig_to_base64(fig)

def chart_pointbiserial_heatmap():
    """Plots point-biserial correlation heatmap."""
    df = _get_processed_data()
    if df.empty or 'target' not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Données indisponibles')
        return fig_to_base64(fig)

    feats = [f for f in NUM_FEATS if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if not feats:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Aucune variable numérique')
        return fig_to_base64(fig)

    rows = []
    for x in feats + ['target']:
        col = []
        for y in feats + ['target']:
            if x == y:
                col.append(np.nan)
            else:
                pb = stats.pointbiserialr(df[x], df[y])[0]
                col.append(round(pb, 2))
        rows.append(col)

    corr = pd.DataFrame(rows, columns=feats + ['target'], index=feats + ['target'])
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr.mask(mask), cmap=sns.color_palette(MYPAL, as_cmap=True), vmax=1.0, vmin=-1,
                center=0, annot=True, square=False, linewidths=.5, cbar_kws={"shrink": 0.75}, ax=ax)
    ax.set_title("Corrélation Point-Biserial (Numérique vs Cible)", fontsize=20, y=1.05)
    plt.tight_layout()
    return fig_to_base64(fig)

def chart_cramersv_heatmap():
    """Plots Cramér's V correlation heatmap."""
    df = _get_processed_data()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Données indisponibles')
        return fig_to_base64(fig)

    def cramers_v(x, y):
        cm = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(cm)[0]
        n = cm.sum().sum()
        phi2 = chi2 / n
        r, k = cm.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
        rcorr = r - ((r-1)**2) / (n-1)
        kcorr = k - ((k-1)**2) / (n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))) if min(kcorr-1, rcorr-1) > 0 else np.nan

    cat_df = df[[f for f in CAT_FEATS if f in df.columns]]
    if cat_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Aucune variable catégorielle')
        return fig_to_base64(fig)

    rows = []
    for x in cat_df:
        col = [cramers_v(cat_df[x], cat_df[y]) for y in cat_df]
        rows.append([round(c, 2) for c in col])

    corr = pd.DataFrame(rows, columns=cat_df.columns, index=cat_df.columns)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr.mask(mask), cmap=sns.color_palette(MYPAL, as_cmap=True), vmax=1.0, vmin=0,
                center=0, annot=True, square=False, linewidths=.01, cbar_kws={"shrink": 0.75}, ax=ax)
    ax.set_title("Corrélation de Cramér V (Variables Catégorielles)", fontsize=20, y=1.05)
    plt.tight_layout()
    return fig_to_base64(fig)

def chart_roc_curves():
    """Plots ROC curves for multiple classifiers."""
    df = _get_processed_data()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Données indisponibles')
        return fig_to_base64(fig)

    try:
        X_train, X_val, y_train, y_val, _ = _split_encode_data(df)
    except ValueError as e:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f'Erreur: {e}')
        return fig_to_base64(fig)

    names = ['Logistic Regression', 'Nearest Neighbors', 'Support Vectors', 'Nu SVC',
             'Decision Tree', 'Random Forest', 'AdaBoost', 'Gradient Boosting',
             'Naive Bayes', 'Linear DA', 'Quadratic DA', 'Neural Net']
    classifiers = [
        LogisticRegression(solver="liblinear", random_state=0),
        KNeighborsClassifier(2),
        SVC(probability=True, random_state=0),
        NuSVC(probability=True, random_state=0),
        DecisionTreeClassifier(random_state=0),
        RandomForestClassifier(random_state=0),
        AdaBoostClassifier(random_state=0),
        GradientBoostingClassifier(random_state=0),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        MLPClassifier(random_state=0)
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    for name, clf in zip(names, classifiers):
        try:
            clf.fit(X_train, y_train)
            pred_proba = clf.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, pred_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=3, label=f'{name} (AUC = {roc_auc:.2f})')
        except Exception as e:
            logging.error(f"Error with {name} in chart_roc_curves: {e}")

    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Taux de Faux Positifs', fontsize=12)
    ax.set_ylabel('Taux de Vrais Positifs', fontsize=12)
    ax.set_title('Courbes ROC des Classifieurs', fontsize=20)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig_to_base64(fig)

def chart_confusion_matrices():
    """Plots confusion matrices for multiple classifiers."""
    df = _get_processed_data()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Données indisponibles')
        return fig_to_base64(fig)

    try:
        X_train, X_val, y_train, y_val, _ = _split_encode_data(df)
    except ValueError as e:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f'Erreur: {e}')
        return fig_to_base64(fig)

    names = ['Logistic Regression', 'Nearest Neighbors', 'Support Vectors', 'Nu SVC',
             'Decision Tree', 'Random Forest', 'AdaBoost', 'Gradient Boosting',
             'Naive Bayes', 'Linear DA', 'Quadratic DA', 'Neural Net']
    classifiers = [
        LogisticRegression(solver="liblinear", random_state=0),
        KNeighborsClassifier(2),
        SVC(probability=True, random_state=0),
        NuSVC(probability=True, random_state=0),
        DecisionTreeClassifier(random_state=0),
        RandomForestClassifier(random_state=0),
        AdaBoostClassifier(random_state=0),
        GradientBoostingClassifier(random_state=0),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        MLPClassifier(random_state=0)
    ]

    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))
    axes = axes.flatten()

    for clf, ax, name in zip(classifiers, axes, names):
        try:
            clf.fit(X_train, y_train)
            cm = confusion_matrix(y_val, clf.predict(X_val))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Sain (0)', 'Malade (1)'],
                        yticklabels=['Sain (0)', 'Malade (1)'], ax=ax)
            ax.set_title(name)
            ax.set_xlabel('Prédiction')
            ax.set_ylabel('Réel')
        except Exception as e:
            logging.error(f"Error with {name} in chart_confusion_matrices: {e}")
            ax.set_title(f'Erreur: {name}')

    plt.tight_layout()
    return fig_to_base64(fig)

def chart_lgbm_confusion_matrix():
    """Plots confusion matrix for tuned LightGBM."""
    logging.debug("Starting chart_lgbm_confusion_matrix")
    df = _get_processed_data()
    if df.empty:
        logging.error("Empty DataFrame in chart_lgbm_confusion_matrix")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Données indisponibles')
        return fig_to_base64(fig)

    try:
        X_train, X_val, y_train, y_val, _ = _split_encode_data(df)
        params = {'num_leaves': 20, 'max_depth': 5, 'min_data_in_leaf': 80}
        import contextlib
        import os
        lgbm = LGBMClassifier(**params, random_state=0, verbose=-1)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
                lgbm.fit(X_train, y_train, eval_set=(X_val, y_val))
        cm = confusion_matrix(y_val, lgbm.predict(X_val))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Sain (0)', 'Malade (1)'],
                    yticklabels=['Sain (0)', 'Malade (1)'], ax=ax)
        ax.set_title('Matrice de Confusion (LightGBM Optimisé)', fontsize=14)
        ax.set_xlabel('Prédiction', fontsize=12)
        ax.set_ylabel('Valeur Réelle', fontsize=12)
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        logging.error(f"Error in chart_lgbm_confusion_matrix: {e}")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title(f'Erreur: {e}')
        return fig_to_base64(fig)

def chart_permutation_importance():
    """Plots permutation importance for tuned LightGBM."""
    logging.debug("Starting chart_permutation_importance")
    if eli5 is None or PermutationImportance is None:
        logging.error("eli5 or PermutationImportance not installed")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title('Erreur: eli5 non installé')
        return fig_to_base64(fig)

    df = _get_processed_data()
    if df.empty:
        logging.error("Empty DataFrame in chart_permutation_importance")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title('Erreur: Données indisponibles')
        return fig_to_base64(fig)

    try:
        X_train, X_val, y_train, y_val, features = _split_encode_data(df)
        # Validate data
        if X_train.isna().any().any() or y_train.isna().any():
            logging.error("NaN values in X_train or y_train")
            raise ValueError("NaN values in input data")
        if not np.all(X_train.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            logging.error("Non-numeric columns in X_train")
            raise ValueError("Non-numeric columns in X_train")

        params = {'num_leaves': 20, 'max_depth': 5, 'min_data_in_leaf': 80}
        import contextlib
        import os
        lgbm = LGBMClassifier(**params, random_state=0, verbose=-1)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
                lgbm.fit(X_train, y_train)
        perm_imp = PermutationImportance(lgbm, random_state=0).fit(X_train, y_train)

        importances = pd.DataFrame({
            'feature': features,
            'importance': perm_imp.feature_importances_
        }).sort_values('importance', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importances['feature'], importances['importance'], color=MYPAL[4])
        ax.set_title('Importance des Variables par Permutation (LightGBM)', fontsize=14)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Variable', fontsize=12)
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        logging.error(f"Error in chart_permutation_importance: {str(e)}")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f'Erreur: {str(e)}')
        return fig_to_base64(fig)

def chart_shap_bar():
    """Plots SHAP bar plot for tuned LightGBM."""
    logging.debug("Starting chart_shap_bar")
    if shap is None:
        logging.error("shap not installed")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: shap non installé')
        return fig_to_base64(fig)

    df = _get_processed_data()
    if df.empty:
        logging.error("Empty DataFrame in chart_shap_bar")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Données indisponibles')
        return fig_to_base64(fig)

    try:
        X_train, X_val, y_train, y_val, features = _split_encode_data(df)
        # Validate data
        if X_val.isna().any().any() or y_val.isna().any():
            logging.error("NaN values in X_val or y_val")
            raise ValueError("NaN values in input data")
        if not np.all(X_val.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            logging.error("Non-numeric columns in X_val")
            raise ValueError("Non-numeric columns in X_val")

        params = {'num_leaves': 20, 'max_depth': 5, 'min_data_in_leaf': 80}
        import contextlib
        import os
        lgbm = LGBMClassifier(**params, random_state=0)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
                lgbm.fit(X_train, y_train)
        explainer = shap.TreeExplainer(lgbm)
        shap_values = explainer.shap_values(X_val)
        logging.debug(f"X_val shape: {X_val.shape}, shap_values type: {type(shap_values)}, shap_values shape: {[v.shape for v in shap_values] if isinstance(shap_values, list) else shap_values.shape}")
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 for binary classification
            logging.debug(f"Selected shap_values[1] shape: {shap_values.shape}")
        if shap_values.ndim == 1:
            logging.warning(f"shap_values is 1D with shape {shap_values.shape}; reshaping to (1, -1)")
            shap_values = shap_values.reshape(1, -1)
        shap.summary_plot(shap_values, X_val, feature_names=features, plot_type="bar", show=False)

        fig = plt.gcf()
        fig.suptitle('Importance des Variables SHAP (LightGBM)', fontsize=14, y=1.05)
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        logging.error(f"Error in chart_shap_bar: {str(e)}")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f'Erreur: {str(e)}')
        return fig_to_base64(fig)

def chart_shap_summary():
    """Plots SHAP summary plot for tuned LightGBM."""
    logging.debug("Starting chart_shap_summary")
    if shap is None:
        logging.error("shap not installed")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: shap non installé')
        return fig_to_base64(fig)

    df = _get_processed_data()
    if df.empty:
        logging.error("Empty DataFrame in chart_shap_summary")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Erreur: Données indisponibles')
        return fig_to_base64(fig)

    try:
        X_train, X_val, y_train, y_val, features = _split_encode_data(df)
        # Validate data
        if X_val.isna().any().any() or y_val.isna().any():
            logging.error("NaN values in X_val or y_val")
            raise ValueError("NaN values in input data")
        if not np.all(X_val.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            logging.error("Non-numeric columns in X_val")
            raise ValueError("Non-numeric columns in X_val")

        params = {'num_leaves': 20, 'max_depth': 5, 'min_data_in_leaf': 80}
        import contextlib
        import os
        lgbm = LGBMClassifier(**params, random_state=0)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
                lgbm.fit(X_train, y_train)
        explainer = shap.TreeExplainer(lgbm)
        shap_values = explainer.shap_values(X_val)
        logging.debug(f"X_val shape: {X_val.shape}, shap_values type: {type(shap_values)}, shap_values shape: {[v.shape for v in shap_values] if isinstance(shap_values, list) else shap_values.shape}")
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 for binary classification
            logging.debug(f"Selected shap_values[1] shape: {shap_values.shape}")
        if shap_values.ndim == 1:
            logging.warning(f"shap_values is 1D with shape {shap_values.shape}; reshaping to (1, -1)")
            shap_values = shap_values.reshape(1, -1)
        elif shap_values.ndim > 2:
            logging.warning(f"shap_values has {shap_values.ndim} dimensions; flattening to 2D")
            shap_values = shap_values.reshape(shap_values.shape[0], -1)
        logging.debug(f"Final shap_values shape for summary_plot: {shap_values.shape}")
        shap.summary_plot(shap_values, X_val, feature_names=features, show=False)

        fig = plt.gcf()
        fig.suptitle('Résumé SHAP (LightGBM)', fontsize=14, y=1.05)
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        logging.error(f"Error in chart_shap_summary: {str(e)}")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f'Erreur: {str(e)}')
        return fig_to_base64(fig)

# -------------------- 5. Performance Table --------------------
def model_performance_table():
    """Generates HTML performance table."""
    logging.debug("Starting model_performance_table")
    df = _get_processed_data()
    if df.empty:
        logging.error("Empty DataFrame in model_performance_table")
        return "<p class='text-danger'>Erreur: Données indisponibles</p>"

    try:
        X_train, X_val, y_train, y_val, _ = _split_encode_data(df)
        # Validate data
        if X_train.isna().any().any() or X_val.isna().any().any() or y_train.isna().any() or y_val.isna().any():
            logging.error("NaN values in X_train, X_val, y_train, or y_val")
            raise ValueError("NaN values in input data")
        if not np.all(X_train.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            logging.error("Non-numeric columns in X_train")
            raise ValueError("Non-numeric columns in X_train")

        names = ['Logistic Regression', 'Nearest Neighbors', 'Support Vectors', 'Nu SVC',
                 'Decision Tree', 'Random Forest', 'AdaBoost', 'Gradient Boosting',
                 'Naive Bayes', 'Linear DA', 'Quadratic DA', 'Neural Net',
                 'Catboost', 'XGBoost', 'LightGBM']
        classifiers = [
            LogisticRegression(solver="liblinear", random_state=0),
            KNeighborsClassifier(2),
            SVC(probability=True, random_state=0),
            NuSVC(probability=True, random_state=0),
            DecisionTreeClassifier(random_state=0),
            RandomForestClassifier(random_state=0),
            AdaBoostClassifier(random_state=0),
            GradientBoostingClassifier(random_state=0),
            GaussianNB(),
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis(),
            MLPClassifier(random_state=0, max_iter=1000),
            CatBoostClassifier(random_state=0, verbose=0) if 'CatBoostClassifier' in globals() else None,
            XGBClassifier(objective='binary:logistic', random_state=0) if 'XGBClassifier' in globals() else None,
            LGBMClassifier(num_leaves=20, max_depth=5, min_data_in_leaf=80, random_state=0, verbose=-1) if 'LGBMClassifier' in globals() else None
        ]

        cols = ["Classifier", "Accuracy", "ROC_AUC", "Recall", "Precision", "F1"]
        data_rows = []

        for name, clf in zip(names, classifiers):
            if clf is None:
                logging.warning(f"{name} classifier not available; skipping")
                data_rows.append({
                    'Classifier': name,
                    'Accuracy': 'Non installé',
                    'ROC_AUC': 'Non installé',
                    'Recall': 'Non installé',
                    'Precision': 'Non installé',
                    'F1': 'Non installé'
                })
                continue
            try:
                logging.debug(f"Training {name}")
                clf.fit(X_train, y_train)
                pred = clf.predict(X_val)
                pred_proba = clf.predict_proba(X_val)[:, 1]
                cm = confusion_matrix(y_val, pred)
                data_rows.append({
                    'Classifier': name,
                    'Accuracy': accuracy_score(y_val, pred) * 100,
                    'ROC_AUC': auc(*roc_curve(y_val, pred_proba)[:2]),
                    'Recall': cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0,
                    'Precision': cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0,
                    'F1': f1_score(y_val, pred, zero_division=0)
                })
            except Exception as e:
                logging.error(f"Error evaluating {name}: {str(e)}")
                data_rows.append({
                    'Classifier': name,
                    'Accuracy': 'Erreur',
                    'ROC_AUC': 'Erreur',
                    'Recall': 'Erreur',
                    'Precision': 'Erreur',
                    'F1': 'Erreur'
                })

        data_table = pd.DataFrame(data_rows, columns=cols)
        logging.info("Generated model performance table")
        return data_table.to_html(
            classes='table table-striped table-hover table-sm',
            index=False,
            float_format='%.2f',
            border=0
        )
    except Exception as e:
        logging.error(f"Error in model_performance_table: {str(e)}")
        return f"<p class='text-danger'>Erreur: {str(e)}</p>"

if __name__ == "__main__":
    print("--- Testing Chart Generation ---")
    for func in [chart_target_distribution, chart_pearson_heatmap, chart_roc_curves, chart_shap_summary, chart_shap_bar, chart_permutation_importance, model_performance_table]:
        print(f"\n{func.__name__}:")
        result = func()
        if isinstance(result, str) and "<table" in result:
            print(result[:100] + "...")
        else:
            print(result[:100] + "...")
