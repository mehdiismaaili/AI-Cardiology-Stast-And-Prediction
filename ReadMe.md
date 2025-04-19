# Guide d’installation et d’utilisation de AI‑Card‑Stats‑Pred

**Projet** : AI‑Card‑Stats‑Pred   
**But** : Installer et lancer l’application web qui analyse des données cardiaques et prédit le risque de maladie.

---

## 1. Présentation rapide

AI‑Card‑Stats‑Pred est une application web qui :

1. **Analyse** un jeu de données sur les maladies cardiaques (graphiques et statistiques).  
2. **Prédit** la probabilité de maladie à partir d’un formulaire (machine learning).  

Elle s’appuie sur :
- Une base MySQL (XAMPP) pour stocker les données.
- Une API Python (Flask) pour la prédiction.
- Un frontend PHP pour l’affichage.

---

## 2. Ce dont vous avez besoin

- **XAMPP** (Apache + MySQL)  
- **phpMyAdmin** (inclus dans XAMPP)  
- **Python 3.8+**  
- **pip**, **venv** (venv est fourni avec Python)  
- **Éditeur de texte** (VS Code, Notepad++, etc.)  
- **Dossier du projet** `AI‑Card‑Stats‑Pred` copié dans `C:\xampp\htdocs\`

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

### A. Base de données MySQL

1. **Démarrer XAMPP** : lancez Apache et MySQL dans le panneau de contrôle.  
2. **Créer la base**  
   - Ouvrez phpMyAdmin (`http://localhost/phpmyadmin`).  
   - Cliquez sur **Nouvelle** → nommez-la **cardiology** → **Créer**.  
3. **Créer la table**  
   - Sélectionnez **cardiology** → onglet **SQL**.  
   - Copiez le contenu de `php/db_files/table.sql` et exécutez.  
4. **Insérer les données**  
   - Ouvrez un terminal CMD/PowerShell.  
   - Allez dans `...\php\db_files` :  
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
