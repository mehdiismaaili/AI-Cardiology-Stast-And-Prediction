# Guide d’installation et d’utilisation de AI‑Card‑Stats‑Pred

**Projet** : AI‑Card‑Stats‑Pred  
**But** : Mettre en place et lancer l’application pour analyser des données cardiaques et prédire le risque.

---

## 1. Présentation rapide

AI‑Card‑Stats‑Pred est une application web qui :

1. **Analyse** un jeu de données sur les maladies cardiaques (graphiques, statistiques).  
2. **Prédit** la probabilité de maladie à partir d’un formulaire (machine learning).  

Elle s’appuie sur :
- **MySQL** (via XAMPP) pour la base de données  
- Une **API Flask** (Python) pour la prédiction  
- Un frontend **PHP** pour l’interface  

---

## 2. Prérequis

- **XAMPP** (Apache + MySQL)  
- **phpMyAdmin** ou **MySQL Workbench**  
- **Python 3.8+** (avec **venv**, **pip**)  
- **Éditeur de texte** (VS Code, Notepad++, etc.)  
- Le dossier `AI‑Card‑Stats‑Pred` copié dans `C:\xampp\htdocs\`

---

## 3. Structure du projet

```
C:\xampp\htdocs\AI‑Card‑Stats‑Pred\
│
├─ env\              # (sera créé) environnement Python
├─ php\
│   ├─ db_files\     # scripts SQL + insertion de données
│   ├─ graphs\       # scripts Python pour générer les graphiques
│   ├─ app\          # API Flask pour la prédiction
│   ├─ style.css    # CSS du frontend
│   ├─ index.php     # page d’analyse des graphiques
│   └─ predict.php   # page du formulaire de prédiction
```

---

## 4. Installation pas à pas

### Étape 1 : Créer l’environnement Python et installer les bibliothèques

1. Ouvrez un terminal (CMD ou PowerShell).  
2. Allez à la racine du projet :
   ```bash
   cd C:\xampp\htdocs\AI‑Card‑Stats‑Pred
   ```
3. Créez et activez l’environnement virtuel :
   ```bash
   python -m venv env
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   env\Scripts\activate
   ```
4. Installez les dépendances :
   ```bash
   cd php
   pip install -r requirements.txt
   ```
   *(flask, pandas, numpy, scikit‑learn, matplotlib, seaborn, mysql‑connector‑python, lightgbm, shap, eli5, etc.)*

---

### Étape 2 : Configurer et remplir la base MySQL

1. **Démarrez XAMPP** : lancez Apache et MySQL.  
2. **Créer la base**  
   - Ouvrez **phpMyAdmin** (`http://localhost/phpmyadmin`) ou MySQL Workbench.  
   - Créez une base nommée **cardiology**.  
3. **Créer la table**  
   - Sélectionnez **cardiology**, onglet **SQL**.  
   - Copiez-collez le contenu de `php/db_files/table.sql` et exécutez.  
4. **Insérer les données**  
   - Dans le même terminal (avec l’env activé) :
     ```bash
     cd php\db_files
     python insert_data.py
     ```
   - Vérifiez qu’il n’y a pas d’erreur.

---

### Étape 3 : Mettre à jour les accès MySQL

Dans les deux fichiers `php/graphs/config.py` et `php/app/config.py`, ajustez :
```python
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASS = ''          # votre mot de passe MySQL, si défini
DB_NAME = 'cardiology'
```

---

### Étape 4 : Former et exporter le modèle

1. Dans votre terminal (env activé) :
   ```bash
   cd C:\xampp\htdocs\AI‑Card‑Stats‑Pred
   python train_export.py
   ```
2. Cela génère deux fichiers :
   - `lgbm_model.pkl`  
   - `encoders.pkl`  

---

### Étape 5 : Lancer l’API Flask

1. Activez l’environnement si nécessaire.  
2. Dans `php/app/` :
   ```bash
   cd php\app
   python app.py
   ```
3. L’API est accessible sur `http://localhost:5000/predict`.

---

### Étape 6 : Démarrer le site web

1. Assurez-vous qu’**Apache** tourne toujours (XAMPP).  
2. Ouvrez votre navigateur à :
   ```
   http://localhost/AI‑Card‑Stats‑Pred/php/index.php
   ```
3. Pour tester la prédiction :
   ```
   http://localhost/AI‑Card‑Stats‑Pred/php/predict.php
   ```
   → Remplissez le formulaire, soumettez, et obtenez la probabilité, la catégorie de risque et le graphique SHAP.

---

## 5. Vérification et dépannage

- **Erreur MySQL** : revérifiez `DB_USER`/`DB_PASS` dans les deux fichiers `config.py`.  
- **Paquet manquant** : `pip install -r requirements.txt` dans `php`.  
- **Port 5000 occupé** : changez le port dans `app.py` et mettez à jour l’URL de fetch dans `predict.php`.  
- **404 page PHP** : vérifiez que le dossier est bien dans `htdocs` et qu’Apache est actif.
