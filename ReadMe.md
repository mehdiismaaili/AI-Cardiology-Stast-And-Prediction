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
│   ├── style.css          # CSS for styling the web interface
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
     cd C:\xampp\htdocs\AI-Card-Stats-Pred\php\db_files
     ```
   - Run the data insertion script:
     ```bash
     python insert_data.py
     ```
   - Ensure the script uses the correct database credentials (e.g., `user=root`, `password=`). If errors occur, update `insert_data.py` with your MySQL credentials (see Step 2).

### Step 2: Configure Database Connections

1. **Edit `php/graphs/config.py`**:
   - Open `php/graphs/config.py` (e.g., `C:\xampp\htdocs\AI-Card-Stats-Pred\php\graphs\config.py`) in a text editor.
   - Update the database connection details:
     ```python
     DB_CONFIG = {
         'host': 'localhost',
         'user': 'root',           # Your MySQL username
         'password': '',           # Your MySQL password
         'database': 'cardiology'
     }
     ```
   - Save the file.

2. **Edit `php/app/config.py`**:
   - Open `php/app/config.py` (e.g., `C:\xampp\htdocs\AI-Card-Stats-Pred\php\app\config.py`).
   - Update the same database connection details:
     ```python
     DB_CONFIG = {
         'host': 'localhost',
         'user': 'root',           # Your MySQL username
         'password': '',           # Your MySQL password
         'database': 'cardiology'
     }
     ```
   - Save the file.

**Note**: If your MySQL `root` user has a password, update both files accordingly. Test the connection in phpMyAdmin or MySQL Workbench to confirm credentials.

### Step 3: Set Up the Python Virtual Environment

1. **Create the Virtual Environment**:
   - Open a terminal and navigate to the project root:
     ```bash
     cd C:\xampp\htdocs\AI-Card-Stats-Pred
     ```
   - Create a virtual environment named `env`:
     ```bash
     python -m venv env
     ```

2. **Activate the Virtual Environment**:
   - Activate the environment:
     ```bash
     env\Scripts\activate
     ```
   - Your terminal should show `(env)` to indicate the virtual environment is active.

3. **Install Dependencies**:
   - Navigate to the `php` folder:
     ```bash
     cd php
     ```
   - Install the required Python packages:
     ```bash
     pip install -r requirements.txt
     ```
   - Example `requirements.txt` dependencies (verify the file):
     ```
     flask
     pandas
     numpy
     scikit-learn
     matplotlib
     seaborn
     mysql-connector-python
     ```
   - If installation fails, ensure `pip` is updated:
     ```bash
     pip install --upgrade pip
     ```

### Step 4: Run the Prediction API

1. **Navigate to the `php/app` Folder**:
   - In the terminal (with `env` activated):
     ```bash
     cd C:\xampp\htdocs\AI-Card-Stats-Pred\php\app
     ```

2. **Run the Flask API**:
   - Start the prediction API:
     ```bash
     python app.py
     ```
   - The API should start on `http://localhost:5000` (or another port if specified in `app.py`).
   - Keep the terminal open to keep the API running.
   - If errors occur (e.g., `ModuleNotFoundError`), ensure all dependencies are installed and `config.py` credentials are correct.

### Step 5: Run the Web Application

1. **Start Apache**:
   - In the XAMPP Control Panel, ensure the **Apache** module is running (MySQL should already be running from Step 1).

2. **Access the Application**:
   - Open a browser and navigate to:
     ```
     http://localhost/AI-Card-Stats-Pred/php/index.php
     ```
   - The main page should load, displaying heart disease analysis graphs and a link to the prediction form.

3. **Test the Prediction Feature**:
   - Click the “Faire une Prédiction de Maladie Cardiaque” button or navigate to:
     ```
     http://localhost/AI-Card-Stats-Pred/php/predict.php
     ```
   - Fill out the form (e.g., `age: 29`, `sex: male`, `thalassemia: normal`) and submit.
   - Verify that the API returns a prediction (e.g., probability and risk category) and a SHAP plot.

## Troubleshooting

- **Database Errors**:
  - **“Access denied for user”**: Verify `user` and `password` in `php/graphs/config.py` and `php/app/config.py`. Test credentials in phpMyAdmin.
  - **“Table doesn’t exist”**: Ensure `table.sql` was executed correctly in phpMyAdmin.
  - **No data**: Run `insert_data.py` and check for errors in the terminal.

- **Python Errors**:
  - **“ModuleNotFoundError”**: Ensure `requirements.txt` dependencies are installed in the `env` virtual environment.
  - **“Port already in use”**: Stop other applications using port 5000 or change the port in `php/app/app.py` (e.g., `app.run(port=5001)`).
  - **“Permission denied”**: Run the terminal as Administrator.

- **Webpage Errors**:
  - **“404 Not Found”**: Verify `AI-Card-Stats-Pred` is in `C:\xampp\htdocs` and Apache is running.
  - **Graphs not loading**: Check `php/graphs/config.py` credentials and ensure `php/graphs/` scripts (e.g., `chart_target_distribution.py`) are present.
  - **Prediction fails**: Ensure `php/app/app.py` is running and the API is accessible at `http://localhost:5000/predict`.

- **Styling Issues**:
  - If `index.php` or `predict.php` looks unstyled, verify `styles.css` is in `php/` and referenced correctly.
  - Clear browser cache or hard refresh (`Ctrl+F5`).

## Additional Notes

- **File Paths**: Adjust paths if XAMPP is installed elsewhere (e.g., `D:\xampp`) or if using macOS/Linux (e.g., `/opt/lampp/htdocs`).
- **Python Version**: Ensure Python 3.8+ is used, as some dependencies (e.g., `scikit-learn`) may not support older versions.
- **API Port**: If `http://localhost:5000` is inaccessible, check `php/app/app.py` for the port and test with `curl http://localhost:5000/predict`.
- **Database Backup**: Before running `table.sql` or `insert_data.py`, back up any existing `cardiology` database.
- **Logs**: Check `app.log`, `charts.log`, or `train.log` in the project root for debugging API or graph issues.
- **Testing**: Use test cases to verify predictions (e.g., `age: 29`, `sex: male`, `thalassemia: normal` should yield low risk).

## Support

For issues not covered, contact the project maintainer or check the project documentation. Provide:

- Screenshots of errors.
- Logs from `app.log`, `charts.log`, or terminal output.
- MySQL error messages from phpMyAdmin or `insert_data.py`.
- Browser Console errors (F12 > Console).

This guide ensures you can set up and run the AI-Card-Stats-Pred application successfully. Happy analyzing!

---
