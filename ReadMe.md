Below is the regenerated setup guide as a Markdown file, suitable for inclusion in your Git repository. The content is identical to the provided guide, formatted in clean Markdown with proper headings, lists, code blocks, and structure. The guide reflects the latest project folder structure (with `predict.php` and `app/` inside `php/`) and is wrapped in an `<xaiArtifact>` tag with the same `artifact_id` (`688d6111-bbd7-4ad5-91fd-51c66a6e7f74`) since it’s an update. The Markdown is optimized for readability in a Git repository (e.g., rendering correctly in GitHub or GitLab).

---


# AI-Card-Stats-Pred Setup and Run Guide

**Project**: AI-Card-Stats-Pred  
**Date**: April 19, 2025  
**Purpose**: Guide to set up and run the AI-Card-Stats-Pred application for heart disease data analysis and prediction.

## Introduction

The AI-Card-Stats-Pred project is a web-based application for analyzing heart disease data and predicting cardiac risk using machine learning. It uses a MySQL database to store data, a Python-based prediction API, and PHP for the frontend, served via XAMPP. This guide provides step-by-step instructions to set up the project environment, configure the database, and run the application.

## Prerequisites

Before starting, ensure you have the following installed:

- **XAMPP**: Web server with Apache and MySQL (download from [https://www.apachefriends.org](https://www.apachefriends.org)).
- **MySQL Workbench** or **phpMyAdmin**: For database management (phpMyAdmin is included with XAMPP).
- **Python 3.8+**: For the virtual environment and API (download from [https://www.python.org](https://www.python.org)).
- **Git** (optional): To clone the project repository if not already downloaded.
- **Text Editor**: For editing configuration files (e.g., VS Code, Notepad++).
- **Project Files**: The `AI-Card-Stats-Pred` folder, containing:
  - `php/` (with `styles.css`, `index.php`, `predict.php`, `requirements.txt`, `db_files/`, `graphs/`, `app/`).
  - `env/` (to be created for the virtual environment).

## Project Folder Structure

Place the `AI-Card-Stats-Pred` folder in XAMPP’s `htdocs` directory (e.g., `C:\xampp\htdocs`). The structure is:

```
AI-Card-Stats-Pred/
├── env/                    # Python virtual environment (to be created)
├── php/                    # Frontend, database scripts, graphs, and prediction API
│   ├── db_files/           # Database setup scripts
│   │   ├── table.sql       # SQL for creating the cardiology table
│   │   ├── insert_data.py  # Script to insert data into the database
│   ├── graphs/             # Graph generation scripts
│   │   ├── config.py       # Database connection settings
│   │   ├── [graph scripts] # Python scripts for generating graphs
│   ├── app/                # Prediction API
│   │   ├── app.py          # Flask API for predictions
│   │   ├── config.py       # Database connection settings
│   ├── styles.css          # CSS for styling the web interface
│   ├── index.php           # Main webpage for data analysis
│   ├── predict.php         # Webpage for heart disease prediction
│   ├── requirements.txt    # Python dependencies
```

## Setup Instructions

Follow these steps to set up and run the application. Commands assume a Windows environment (since XAMPP is mentioned); adjust for macOS/Linux if needed.

### Step 1: Set Up the MySQL Database

1. **Start XAMPP**:
   - Open the XAMPP Control Panel.
   - Start the **Apache** and **MySQL** modules.

2. **Access phpMyAdmin**:
   - Open a browser and navigate to `http://localhost/phpmyadmin`.
   - Alternatively, use MySQL Workbench to connect to `localhost` (default user: `root`, password: empty or set during XAMPP installation).

3. **Create the Database**:
   - In phpMyAdmin, click **New** in the left sidebar.
   - Enter `cardiology` as the database name and click **Create**.

4. **Create the Table**:
   - Select the `cardiology` database in phpMyAdmin.
   - Go to the **SQL** tab.
   - Copy the SQL from `php/db_files/table.sql` (e.g., `C:\xampp\htdocs\AI-Card-Stats-Pred\php\db_files\table.sql`).
   - Paste and execute the SQL to create the table (e.g., `heart_disease_stats`).
   - Example content of `table.sql` (verify the exact structure):
     ```sql
     CREATE TABLE heart_disease_stats (
         id INT AUTO_INCREMENT PRIMARY KEY,
         age INT,
         sex VARCHAR(10),
         chest_pain_type VARCHAR(50),
         resting_blood_pressure INT,
         cholesterol INT,
         fasting_blood_sugar VARCHAR(50),
         resting_electrocardiogram VARCHAR(50),
         max_heart_rate_achieved INT,
         exercise_induced_angina VARCHAR(10),
         st_depression FLOAT,
         st_slope VARCHAR(20),
         num_major_vessels INT,
         thalassemia VARCHAR(50),
         target INT
     );
     ```

5. **Insert Data**:
   - Open a terminal (e.g., Command Prompt or PowerShell).
   - Navigate to the `php/db_files` directory:
     ```bash
     cd C:\xampp\htdocs\AI‑Card‑Stats‑Pred\php\db_files
     ```  
   - Lancez :  
     ```bash
     python insert_data.py
     ```  

### B. Configurer les accès MySQL

- Modifier **php/graphs/config.py** et **php/app/config.py** :

  ```python
  DB_HOST = 'localhost'
  DB_USER = 'root'
  DB_PASS = ''          # ou votre mot de passe MySQL
  DB_NAME = 'cardiology'
  ```

---

### C. Environnement Python & dépendances

1. **Créer & activer l’environnement**  
   ```bash
   cd C:\xampp\htdocs\AI‑Card‑Stats‑Pred
   python -m venv env
   env\Scripts\activate
   ```
2. **Installer les paquets**  
   ```bash
   cd php
   pip install -r requirements.txt
   ```

---

### D. Former et exporter le modèle

1. **Lancer le script**  
   ```bash
   cd C:\xampp\htdocs\AI‑Card‑Stats‑Pred
   python train_export.py
   ```  
   → Cela génère `lgbm_model.pkl` et `encoders.pkl`.  

---

### E. Démarrer l’API de prédiction

1. **Activer l’environnement** si besoin (`env\Scripts\activate`).  
2. **Lancer Flask**  
   ```bash
   cd php\app
   python app.py
   ```  
   → API disponible sur `http://localhost:5000/predict`.  

---

### F. Lancer le site web

1. Vérifiez qu’**Apache** (XAMPP) tourne toujours.  
2. Ouvrez votre navigateur à :  
   ```
   http://localhost/AI‑Card‑Stats‑Pred/php/index.php
   ```  
3. Pour la prédiction :  
   ```
   http://localhost/AI‑Card‑Stats‑Pred/php/predict.php
   ```  
   → Remplissez le formulaire, soumettez, et voyez la probabilité + le graphique SHAP.

---

## 5. Résolution de problèmes courants

- **Erreur de connexion MySQL** : revérifiez `DB_USER`/`DB_PASS` dans les deux `config.py`.  
- **Module manquant** : activez `env` puis `pip install -r requirements.txt`.  
- **Port 5000 occupé** : changez le port dans `app.py` et mettez à jour l’URL fetch dans `predict.php`.  
- **404 sur la page PHP** : vérifiez que `AI‑Card‑Stats‑Pred` est bien dans `htdocs` et qu’Apache est en route.
