# Guide d’installation et d’utilisation de AI‑Card‑Stats‑Pred

**Projet** : AI‑Card‑Stats‑Pred   
**But** : Installer et lancer l’application pour analyser des données cardiaques et prédire le risque.

---

## 1. Présentation rapide

AI‑Card‑Stats‑Pred est une application web qui :

1. **Analyse** un jeu de données sur les maladies cardiaques (graphiques et statistiques).  
2. **Prédit** la probabilité de maladie à partir d’un formulaire (machine learning).  

Elle utilise :
- **MySQL** (via XAMPP) pour la base de données  
- Une **API Flask** (Python) pour la prédiction  
- Un frontend **PHP** pour l’interface  

---

## 2. Prérequis

- **XAMPP** (Apache + MySQL)  
- **phpMyAdmin** (inclus dans XAMPP)  
- **Python 3.8+**  
- **pip**, **venv** (fournis avec Python)  
- **Éditeur de texte** (VS Code, Notepad++, etc.)  
- **Dossier projet** `AI‑Card‑Stats‑Pred` copié dans `C:\xampp\htdocs\`

---

## 3. Structure du projet

```
C:\xampp\htdocs\AI‑Card‑Stats‑Pred\
│
├─ env\              # (sera créé) environnement Python
├─ php\
│   ├─ db_files\     # scripts SQL et insertion de données
│   ├─ graphs\       # scripts Python pour générer les graphiques
│   ├─ app\          # API Flask pour la prédiction
│   ├─ styles.css    # styles du frontend
│   ├─ index.php     # page d’analyse des graphiques
│   └─ predict.php   # page du formulaire de prédiction
└─ train_export.py   # script de formation et export du modèle
```

---

## 4. Installation pas à pas

### Étape 1 : Créer l’environnement Python et installer les bibliothèques

1. Ouvrez un terminal (CMD ou PowerShell).  
2. Placez‑vous à la racine du projet :
   ```bash
   cd C:\xampp\htdocs\AI‑Card‑Stats‑Pred
   ```
3. Créez un environnement virtuel `env` :
   ```bash
   python -m venv env
   ```
4. Activez‑le :
   ```bash
   env\Scripts\activate
   ```
   → Vous devez voir `(env)` devant l’invite de commande.

5. Installez les dépendances Python :
   ```bash
   cd php
   pip install -r requirements.txt
   ```
   Le fichier `requirements.txt` contient, par exemple :
   ```
   flask
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   mysql-connector-python
   lightgbm
   shap
   eli5
   ```

---

### Étape 2 : Configurer et remplir la base MySQL

1. **Démarrez XAMPP** : lancez Apache et MySQL.  
2. **Créer la base**  
   - Ouvrez phpMyAdmin (`http://localhost/phpmyadmin`).  
   - Cliquez sur **Nouvelle**, nommez-la **cardiology**, puis **Créer**.  
3. **Créer la table**  
   - Sélectionnez **cardiology**, onglet **SQL**.  
   - Copiez et exécutez le SQL de `php/db_files/table.sql`.  
4. **Insérer les données**  
   - Dans le même terminal (avec `env` activé) :  
     ```bash
     cd C:\xampp\htdocs\AI‑Card‑Stats‑Pred\php\db_files
     python insert_data.py
     ```
   - Vérifiez qu’aucune erreur n’apparaît.

---

### Étape 3 : Mettre à jour les accès MySQL

1. **php/graphs/config.py** et **php/app/config.py** → renseignez :
   ```python
   DB_HOST = 'localhost'
   DB_USER = 'root'
   DB_PASS = ''          # ou votre mot de passe MySQL
   DB_NAME = 'cardiology'
   ```
2. Sauvegardez.

---

### Étape 4 : Former et exporter le modèle

1. Dans le terminal (toujours activé) :
   ```bash
   cd C:\xampp\htdocs\AI‑Card‑Stats‑Pred
   python train_export.py
   ```
2. Cela crée :
   - **lgbm_model.pkl**  
   - **encoders.pkl**  

---

### Étape 5 : Lancer l’API Flask

1. Activez l’env (si besoin).  
2. Lancez l’API :
   ```bash
   cd php\app
   python app.py
   ```
3. L’API écoute sur `http://localhost:5000/predict`.

---

### Étape 6 : Démarrer le site web

1. Vérifiez qu’**Apache** tourne.  
2. Ouvrez dans un navigateur :
   ```
   http://localhost/AI‑Card‑Stats‑Pred/php/index.php
   ```
3. Pour la prédiction :
   ```
   http://localhost/AI‑Card‑Stats‑Pred/php/predict.php
   ```

---

## 5. Vérification et dépannage

- **Erreur MySQL** : vérifiez `DB_USER`/`DB_PASS` dans les deux `config.py`.  
- **Module manquant** : `pip install -r requirements.txt` dans `php`.  
- **Port 5000 occupé** : changez le port dans `app.py` et dans le fetch JS.  
- **404 page PHP** : assurez-vous que `AI‑Card‑Stats‑Pred` est dans `htdocs` et qu’Apache tourne.
