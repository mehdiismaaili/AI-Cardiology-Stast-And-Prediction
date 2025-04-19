<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Prédiction de Maladie Cardiaque</title>
  <link rel="stylesheet" href="styles.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css">
  <style>
    :root {
      --primary: #007bff;
      --dark: #0056b3;
      --light: #e6f0ff;
      --success: #28a745;
      --danger: #dc3545;
      --muted: #6c757d;
      --white: #fff;
      --shadow: rgba(0, 0, 0, 0.1);
      --border: #d3e4ff;
    }

    body {
    font-family: system-ui, -apple-system, sans-serif;
    background-color: var(--light);
    color: #212529;
  }

    /* Header Styles */
    .site-header {
      background-color: var(--primary, #007bff);
      color: var(--white, #fff);
      padding: 1.5rem 0;
      box-shadow: 0 2px 4px var(--shadow, rgba(0, 0, 0, 0.1));
    }

    .site-header h1 {
      font-weight: 600;
      font-size: 1.75rem;
      margin-bottom: 0.25rem;
    }

    .site-header .text-muted {
      font-size: 0.875rem;
      color: rgba(255, 255, 255, 0.7);
    }

    /* Sidebar Styles */
    #toc {
      position: sticky;
      top: 1rem;
      padding: 1rem;
      background-color: var(--white, #fff);
      border: 1px solid var(--border, #d3e4ff);
      border-radius: 8px;
      box-shadow: 0 2px 4px var(--shadow, rgba(0, 0, 0, 0.1));
    }

    #toc h5 {
      color: var(--primary, #007bff);
      font-weight: 600;
      margin-bottom: 0.75rem;
    }

    #toc .nav-link {
      color: var(--dark, #0056b3);
      font-weight: 500;
      padding: 0.5rem 0;
      transition: color 0.3s ease;
    }

    #toc .nav-link:hover,
    #toc .nav-link.active {
      color: var(--primary, #007bff);
    }

    /* Form Styles (Unchanged) */
    .prediction-form .form-control {
      border: 1px solid var(--primary, #007bff);
      border-radius: 6px;
      padding: 0.5rem 0.75rem;
      transition: border-color 0.3s ease, box-shadow 0.3s ease;
      background-color: #fff;
      height: calc(2.25rem + 2px);
    }

    .prediction-form .form-control:focus {
      border-color: var(--dark, #0056b3);
      box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
      outline: none;
    }

    .prediction-form .form-control.is-invalid {
      border-color: var(--danger, #dc3545);
    }

    .prediction-form .form-label {
      color: var(--dark, #0056b3);
      font-weight: 600;
      margin-bottom: 0.5rem;
    }

    .prediction-form .btn-primary {
      background-color: #fff;
      color: var(--primary, #007bff);
      border: 1px solid var(--primary, #007bff);
      padding: 0.75rem 2rem;
      font-size: 1.1rem;
      font-weight: 600;
      border-radius: 8px;
      box-shadow: 0 2px 8px var(--shadow, rgba(0, 0, 0, 0.1));
      transition: background-color 0.3s ease, color 0.3s ease, transform 0.2s ease;
      display: block;
      margin: 1.5rem auto 0;
    }

    .prediction-form .btn-primary:hover {
      background-color: var(--dark, #0056b3);
      color: #fff;
      transform: scale(1.05);
    }

    .prediction-form .invalid-feedback {
      font-size: 0.875rem;
      color: var(--danger, #dc3545);
    }

    /* Card and Container Styles */
    .chart-card {
      background-color: var(--white, #fff);
      border: 1px solid var(--border, #d3e4ff);
      border-radius: 10px;
      box-shadow: 0 2px 8px var(--shadow, rgba(0, 0, 0, 0.1));
      transition: box-shadow 0.3s ease;
    }

    .chart-card:hover {
      box-shadow: 0 4px 12px var(--shadow, rgba(0, 0, 0, 0.15));
    }

    .chart-container {
      padding: 1.5rem;
    }

    .risk-low {
      color: var(--success, #28a745);
      font-weight: 600;
    }

    .risk-high {
      color: var(--danger, #dc3545);
      font-weight: 600;
    }

    .text-muted-small {
      font-size: 0.875rem;
      color: var(--muted, #6c757d);
    }

    .alert-danger {
      background-color: #f8d7da;
      border: 1px solid #f5c6cb;
      border-radius: 8px;
      color: #721c24;
      padding: 1rem;
      font-size: 0.875rem;
    }

    .subsection-title {
      color: var(--primary, #007bff);
      font-weight: 600;
      margin-bottom: 1rem;
    }

    /* Footer Styles */
    footer {
      background-color: var(--dark, #0056b3);
      color: var(--white, #fff);
      padding: 1.5rem 0;
      margin-top: 2rem;
    }

    footer .text-muted-small {
      color: rgba(255, 255, 255, 0.7);
      text-align: center;
    }

    /* Responsive Adjustments */
    @media (max-width: 767.98px) {
      .navbar-nav {
        background-color: var(--primary, #007bff);
        padding: 0.5rem;
      }
      .site-header {
        padding: 1rem 0;
      }
      .site-header h1 {
        font-size: 1.5rem;
      }
      .prediction-form .form-control {
        font-size: 0.875rem;
        height: calc(2rem + 2px);
      }
      .prediction-form .btn-primary {
        font-size: 1rem;
        padding: 0.5rem 1.5rem;
      }
      .chart-container {
        padding: 1rem;
      }
      #toc {
        position: static;
        margin-bottom: 1rem;
      }
    }
  </style>
</head>
<body>
  <!-- Header -->
  <header class="site-header">
    <div class="container-md">
      <h1 class="h4 mb-0">Analyse des Données Cardiaques</h1>
      <p class="text-muted">Prédiction des Risques Cardiaques</p>
    </div>
  </header>

  <!-- Main Content -->
  <div class="container-md my-4">
    <div class="row">
      <aside id="toc" class="col-md-3">
        <h5>Navigation</h5>
        <nav class="nav flex-column">
          <a class="nav-link" href="index.php">Analyse des Graphiques</a>
          <a class="nav-link active" href="predict.php">Prédiction</a>
        </nav>
      </aside>
      <main class="col-md-9">
        <section id="prediction">
          <h2 class="section-title">Prédiction de Maladie Cardiaque</h2>
          <div class="chart-card mb-4">
            <div class="chart-container">
              <form id="predictionForm" class="prediction-form needs-validation" novalidate>
                <div class="row">
                  <!-- Age -->
                  <div class="col-md-6 mb-3">
                    <label for="age" class="form-label">Âge</label>
                    <input type="number" class="form-control" id="age" name="age" min="18" max="100" required>
                    <div class="invalid-feedback">Veuillez entrer un âge entre 18 et 100 ans.</div>
                  </div>
                  <!-- Sex -->
                  <div class="col-md-6 mb-3">
                    <label for="sex" class="form-label">Sexe</label>
                    <select class="form-control" id="sex" name="sex" required>
                      <option value="" disabled selected>Choisir...</option>
                      <option value="male">Homme</option>
                      <option value="female">Femme</option>
                    </select>
                    <div class="invalid-feedback">Veuillez sélectionner un sexe.</div>
                  </div>
                  <!-- Chest pain -->
                  <div class="col-md-6 mb-3">
                    <label for="chest_pain_type" class="form-label">Type de douleur thoracique</label>
                    <select class="form-control" id="chest_pain_type" name="chest_pain_type" required>
                      <option value="" disabled selected>Choisir...</option>
                      <option value="typical angina">Angine typique</option>
                      <option value="atypical angina">Angine atypique</option>
                      <option value="non-anginal pain">Douleur non-angineuse</option>
                      <option value="asymptomatic">Asymptomatique</option>
                    </select>
                    <div class="invalid-feedback">Veuillez sélectionner un type de douleur.</div>
                  </div>
                  <!-- Resting BP -->
                  <div class="col-md-6 mb-3">
                    <label for="resting_blood_pressure" class="form-label">Pression artérielle au repos (mmHg)</label>
                    <input type="number" class="form-control" id="resting_blood_pressure" name="resting_blood_pressure" min="80" max="200" required>
                    <div class="invalid-feedback">Veuillez entrer une valeur entre 80 et 200.</div>
                  </div>
                  <!-- Cholesterol -->
                  <div class="col-md-6 mb-3">
                    <label for="cholesterol" class="form-label">Cholestérol (mg/dL)</label>
                    <input type="number" class="form-control" id="cholesterol" name="cholesterol" min="100" max="400" required>
                    <div class="invalid-feedback">Veuillez entrer une valeur entre 100 et 400.</div>
                  </div>
                  <!-- Fasting blood sugar -->
                  <div class="col-md-6 mb-3">
                    <label for="fasting_blood_sugar" class="form-label">Glycémie à jeun (>120 mg/dL)</label>
                    <select class="form-control" id="fasting_blood_sugar" name="fasting_blood_sugar" required>
                      <option value="" disabled selected>Choisir...</option>
                      <option value="greater than 120mg/ml">Supérieure à 120</option>
                      <option value="lower than 120mg/ml">Inférieure à 120</option>
                    </select>
                    <div class="invalid-feedback">Veuillez sélectionner une option.</div>
                  </div>
                  <!-- Resting ECG -->
                  <div class="col-md-6 mb-3">
                    <label for="resting_electrocardiogram" class="form-label">Résultat ECG au repos</label>
                    <select class="form-control" id="resting_electrocardiogram" name="resting_electrocardiogram" required>
                      <option value="" disabled selected>Choisir...</option>
                      <option value="normal">Normal</option>
                      <option value="ST-T wave abnormality">Anomalie ST-T</option>
                      <option value="left ventricular hypertrophy">Hypertrophie VG</option>
                    </select>
                    <div class="invalid-feedback">Veuillez sélectionner une option.</div>
                  </div>
                  <!-- Max heart rate -->
                  <div class="col-md-6 mb-3">
                    <label for="max_heart_rate_achieved" class="form-label">Fréquence cardiaque max</label>
                    <input type="number" class="form-control" id="max_heart_rate_achieved" name="max_heart_rate_achieved" min="60" max="220" required>
                    <div class="invalid-feedback">Veuillez entrer une valeur entre 60 et 220.</div>
                  </div>
                  <!-- Exercise induced angina -->
                  <div class="col-md-6 mb-3">
                    <label for="exercise_induced_angina" class="form-label">Angine à l’effort</label>
                    <select class="form-control" id="exercise_induced_angina" name="exercise_induced_angina" required>
                      <option value="" disabled selected>Choisir...</option>
                      <option value="yes">Oui</option>
                      <option value="no">Non</option>
                    </select>
                    <div class="invalid-feedback">Veuillez sélectionner une option.</div>
                  </div>
                  <!-- ST depression -->
                  <div class="col-md-6 mb-3">
                    <label for="st_depression" class="form-label">Dépression ST (mm)</label>
                    <input type="number" step="0.1" class="form-control" id="st_depression" name="st_depression" min="0" max="6" required>
                    <div class="invalid-feedback">Veuillez entrer une valeur entre 0 et 6.</div>
                  </div>
                  <!-- ST slope -->
                  <div class="col-md-6 mb-3">
                    <label for="st_slope" class="form-label">Pente du ST</label>
                    <select class="form-control" id="st_slope" name="st_slope" required>
                      <option value="" disabled selected>Choisir...</option>
                      <option value="upsloping">Ascendant</option>
                      <option value="flat">Plat</option>
                      <option value="downsloping">Descendant</option>
                    </select>
                    <div class="invalid-feedback">Veuillez sélectionner une option.</div>
                  </div>
                  <!-- Num major vessels -->
                  <div class="col-md-6 mb-3">
                    <label for="num_major_vessels" class="form-label">Nb vaisseaux majeurs</label>
                    <select class="form-control" id="num_major_vessels" name="num_major_vessels" required>
                      <option value="" disabled selected>Choisir...</option>
                      <option value="0">0</option>
                      <option value="1">1</option>
                      <option value="2">2</option>
                      <option value="3">3</option>
                    </select>
                    <div class="invalid-feedback">Veuillez sélectionner une option.</div>
                  </div>
                  <!-- Thalassemia -->
                  <div class="col-md-6 mb-3">
                    <label for="thalassemia" class="form-label">Thalassémie</label>
                    <select class="form-control" id="thalassemia" name="thalassemia" required>
                      <option value="" disabled selected>Choisir...</option>
                      <option value="normal">Normal</option>
                      <option value="fixed defect">Défaut fixé</option>
                      <option value="reversible defect">Défaut réversible</option>
                    </select>
                    <div class="invalid-feedback">Veuillez sélectionner une option.</div>
                  </div>
                </div>
                <button type="submit" class="btn btn-primary">Prédire</button>
              </form>
            </div>
          </div>

          <!-- Result -->
          <div id="result" class="chart-card mt-4" style="display:none">
            <div class="chart-container">
              <h3 class="subsection-title">Résultat de la Prédiction</h3>
              <p>Probabilité : <strong id="probability"></strong></p>
              <p>Risque : <strong id="risk_category"></strong></p>
              <img id="shap_plot" src="" alt="SHAP Plot" style="max-width:100%; display:none" />
              <p class="chart-explanation">Ce graphique montre comment chaque variable contribue à votre risque.</p>
              <p class="text-muted-small mt-2">Usage éducatif uniquement. Consultez un médecin.</p>
            </div>
          </div>

          <!-- Error -->
          <div id="error" class="alert alert-danger mt-4" style="display:none"></div>
        </section>
      </main>
    </div>
  </div>

  <!-- Footer -->
  <footer>
    <div class="container-md text-muted-small py-3">
      Projet AI in Cardiology © 2025
    </div>
  </footer>

  <!-- JS: Validation & fetch -->
  <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.0/dist/jquery.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    // Valid categories based on app.py's _REVERSE_MAPPINGS
    const VALID_CATEGORIES = {
        sex: ['female', 'male'],
        chest_pain_type: ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'],
        fasting_blood_sugar: ['lower than 120mg/ml', 'greater than 120mg/ml'],
        resting_electrocardiogram: ['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'],
        exercise_induced_angina: ['no', 'yes'],
        st_slope: ['upsloping', 'flat', 'downsloping'],
        thalassemia: ['fixed defect', 'normal', 'reversible defect']
    };

    // API endpoint
    const API_ENDPOINT = 'http://localhost:5000/predict';

    // Bootstrap validation
    (function() {
        'use strict';
        window.addEventListener('load', function() {
            var forms = document.getElementsByClassName('needs-validation');
            Array.prototype.forEach.call(forms, function(form) {
                form.addEventListener('submit', function(e) {
                    if (!form.checkValidity()) {
                        e.preventDefault();
                        e.stopPropagation();
                    }
                    form.classList.add('was-validated');
                }, false);
            });
        }, false);
    })();

    // Prediction fetch
    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        if (!this.checkValidity()) return;

        const fd = new FormData(this);
        const payload = {
            age: Number(fd.get('age')),
            sex: fd.get('sex'),
            chest_pain_type: fd.get('chest_pain_type'),
            resting_blood_pressure: Number(fd.get('resting_blood_pressure')),
            cholesterol: Number(fd.get('cholesterol')),
            fasting_blood_sugar: fd.get('fasting_blood_sugar'),
            resting_electrocardiogram: fd.get('resting_electrocardiogram'),
            max_heart_rate_achieved: Number(fd.get('max_heart_rate_achieved')),
            exercise_induced_angina: fd.get('exercise_induced_angina'),
            st_depression: parseFloat(fd.get('st_depression')),
            st_slope: fd.get('st_slope'),
            num_major_vessels: Number(fd.get('num_major_vessels')),
            thalassemia: fd.get('thalassemia')
        };

        // Client-side validation for categorical inputs
        for (const [field, value] of Object.entries(payload)) {
            if (VALID_CATEGORIES[field] && !VALID_CATEGORIES[field].includes(value)) {
                const errDiv = document.getElementById('error');
                errDiv.textContent = `Valeur invalide pour ${field}: ${value}. Options valides: ${VALID_CATEGORIES[field].join(', ')}`;
                errDiv.style.display = 'block';
                return;
            }
        }

        console.log('Sending payload:', payload);

        const resDiv = document.getElementById('result');
        const errDiv = document.getElementById('error');
        resDiv.style.display = 'none';
        errDiv.style.display = 'none';

        try {
            const res = await fetch(API_ENDPOINT, {
                method: 'POST',
                mode: 'cors',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            console.log('Response status:', res.status);
            const data = await res.json();
            console.log('Response data:', data);

            if (res.ok) {
                document.getElementById('probability').textContent = (data.probability * 100).toFixed(1) + ' %';
                const rc = document.getElementById('risk_category');
                rc.textContent = data.risk_category;
                rc.className = data.risk_category === 'Risque élevé' ? 'risk-high' : 'risk-low';

                const img = document.getElementById('shap_plot');
                if (data.shap_plot) {
                    img.src = 'data:image/png;base64,' + data.shap_plot;
                    img.style.display = 'block';
                } else {
                    img.style.display = 'none';
                }
                resDiv.style.display = 'block';
            } else {
                errDiv.textContent = 'Erreur : ' + (data.error || 'Réponse invalide');
                errDiv.style.display = 'block';
            }
        } catch (err) {
            console.error('Network error:', err);
            errDiv.textContent = 'Erreur réseau : ' + err.message;
            errDiv.style.display = 'block';
        }
    });
  </script>
</body>
</html>