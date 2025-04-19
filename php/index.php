<?php
/******************************************************************
 * 0.  CONFIG – adjust only these two lines
 ******************************************************************/
const PYTHON_BIN = 'C:\\xampp\\htdocs\\AI-Card-Stats-Pred\\env\\Scripts\\python.exe';
const GRAPH_DIR  = __DIR__ . '\\graphs\\';
const DEBUG_LOG  = __DIR__ . '\\debug.log';
const CACHE_DIR  = __DIR__ . '\\cache\\';
/* Timeour for maximum execution time */
set_time_limit(3600);

if (!is_dir(CACHE_DIR)) mkdir(CACHE_DIR, 0750, true);

/******************************************************************
 * 1.  run_graph() – executes a python file, logs, caches, validates
 ******************************************************************/
function run_graph(string $script, bool $cache = true): ?string
{
    $cacheFile = CACHE_DIR . "$script.b64";
    if ($cache && is_file($cacheFile) && (time() - filemtime($cacheFile) < 3600)) {
        return file_get_contents($cacheFile);
    }

    $cmd = escapeshellarg(PYTHON_BIN) . ' ' . escapeshellarg(GRAPH_DIR."$script.py") . ' 2>&1';
    $out = shell_exec($cmd);
    $len = $out === null ? 'NULL' : strlen($out);

    file_put_contents(DEBUG_LOG,date('[c] ')."${script} LEN=$len\n$cmd\n\n",FILE_APPEND);

    if ($out === null || !base64_decode($out, true) || strlen($out) < 1000) return null;
    if ($cache) file_put_contents($cacheFile, $out);
    return $out;
}

/******************************************************************
 * 2.  Small helpers – embed images & the performance table
 ******************************************************************/
function graph_img(string $script, string $alt, int $fig): string
{
    $img = run_graph($script);
    return $img
        ? '<img src="data:image/png;base64,'.$img.'" class="img-fluid" alt="'.htmlspecialchars($alt).'">'
        : '<div class="alert alert-danger mb-0">Erreur : '.$alt.' (Figure&nbsp;'.$fig.')</div>';
}

function perf_table(): string
{
    $html = shell_exec(escapeshellarg(PYTHON_BIN).' '.escapeshellarg(GRAPH_DIR.'table_performance.py').' 2>&1');
    file_put_contents(DEBUG_LOG,date('[c] ')."table_performance LEN=".strlen($html)."\n",FILE_APPEND);
    return (strlen($html) > 50 && stripos($html,'<table')!==false)
        ? $html
        : '<div class="alert alert-danger mb-0">Erreur génération table de performance</div>';
}
?>
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Rapport sur les maladies cardiaques</title>

    <!-- CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link rel="stylesheet" href="style.css">
</head>

<body>
<header class="site-header">
    <div class="container-md d-flex justify-content-between align-items-center">
        <h1 class="h4 mb-0">Rapport sur les maladies cardiaques</h1>
        <span class="text-muted no-print">Rafraîchissez pour recalculer les graphiques</span>
    </div>
</header>

<div class="container-md my-4">
    <div class="card border-0 bg-primary text-white shadow">
        <div class="card-body py-5 text-center">
            <h1 class="display-5 fw-bold mb-3">Heart Diseases : indicateurs et prédictions</h1>
            <p class="lead mb-0">Analyse du jeu de données Cleveland (296 patients)</p>
            <br>
            <a href="predict.php" class="btn btn-primary btn-lg predict-btn">Faire une Prédiction de Maladie Cardiaque</a>
        </div>
    </div>
</div>

<div class="container-md">
    <div class="row g-4">
        <!-- Sidebar -->
        <aside class="col-lg-3 d-none d-lg-block">
            <div id="toc">
                <h5><i class="fas fa-list me-2"></i>Sommaire</h5>
                <nav class="nav flex-column">
                    <a class="nav-link" href="predict.php">Prédiction</a>
                    <a class="nav-link" href="#introduction">1 Introduction</a>
                    <a class="nav-link" href="#objectifs">1.1 Objectifs</a>
                    <a class="nav-link" href="#target">2.2 Cible</a>
                    <a class="nav-link" href="#distributions">2.3 Distributions</a>
                    <a class="nav-link" href="#correlations">2.4 Corrélations</a>
                    <a class="nav-link" href="#modelisation">3 Modélisation</a>
                    <a class="nav-link" href="#conclusion">4 Conclusion</a>
                </nav>
            </div>
        </aside>

        <!-- Main -->
        <div class="col-lg-9">
            <main>
                <!-- Prediction button -->
                <!-- Introduction -->
                <section id="introduction">
                    <h2 class="section-title">1. Introduction</h2>
                    <div class="card shadow-sm mb-4"><div class="card-body">
                        Étude basée sur le dataset Cleveland (296 cas après nettoyage) : 13 variables cliniques + cible
                        indiquant la présence d’un rétrécissement coronarien &gt; 50 %.
                    </div></div>
                </section>
                
                <!-- Objectifs -->
                <section id="objectifs">
                    <h3 class="subsection-title">1.1 Objectifs</h3>
                    <ul>
                        <li>Explorer les distributions et corrélations des indicateurs cardiaques.</li>
                        <li>Construire des modèles de classification pour détecter la maladie.</li>
                        <li>Identifier les variables les plus déterminantes (Permutation, SHAP).</li>
                    </ul>
                </section>

                <!-- Target distribution -->
                <section id="target">
                    <h2 class="section-title">2.2 Distribution de la cible</h2>
                    <div class="card chart-card shadow-sm mb-4">
                        <div class="card-body"><?= graph_img('chart_target_distribution','Distribution cible',1); ?></div>
                        <div class="card-footer text-muted-small">
                            <span class="figure-label">Figure 1.</span>
                            Jeu de données équilibré : <b>54 %</b> de patients atteints (maladie) contre <b>46 %</b> sains.
                        </div>
                    </div>
                </section>

                <!-- Distributions -->
                <section id="distributions">
                    <h2 class="section-title">2.3 Distributions des variables</h2>

                    <h3 class="subsection-title">2.3.1 Variables numériques – KDE</h3>
                    <div class="card chart-card shadow-sm mb-4">
                        <div class="card-body"><?= graph_img('chart_numerical_distributions','KDE numériques',2); ?></div>
                        <div class="card-footer text-muted-small">
                            <span class="figure-label">Figure 2.</span>
                            Âge moyen <b>54,5 ans</b> (29–77). Cholestérol médian <b>240 mg/dl</b>.  
                            Les patients malades présentent : <br>
                            • plus de vaisseaux obstrués (<code>num_major_vessels</code>)<br>
                            • fréquence cardiaque maximale plus basse.
                        </div>
                    </div>

                    <h3 class="subsection-title mt-4">2.3.2 Variables catégorielles</h3>
                    <div class="card chart-card shadow-sm mb-4">
                        <div class="card-body"><?= graph_img('chart_categorical_distributions','Catégorielles',3); ?></div>
                        <div class="card-footer text-muted-small">
                            <span class="figure-label">Figure 3.</span>
                            Les douleurs <em>atypical / non‑anginales</em> et une pente ST <em>downsloping</em>
                            sont fortement associées à la maladie.
                        </div>
                    </div>

                    <h3 class="subsection-title mt-4">2.3.3 Relations sélectionnées</h3>
                    <div class="card chart-card shadow-sm mb-4">
                        <div class="card-body"><?= graph_img('chart_regression_plots','Régressions sélectionnées',4); ?></div>
                        <div class="card-footer text-muted-small">
                            <span class="figure-label">Figure 4.</span>
                            Chez les malades : cholestérol, tension au repos et dépression ST
                            augmentent avec l’âge ; la fréquence cardiaque max diminue.
                        </div>
                    </div>

                    <h3 class="subsection-title mt-4">2.3.4 Pair‑plot</h3>
                    <p class="text-muted-small mb-2">(Affichage lourd – peut prendre quelques secondes.)</p>
                    <div class="card chart-card shadow-sm mb-4">
                        <div class="card-body"><?= graph_img('chart_pairplot','Pair‑plot',5); ?></div>
                        <div class="card-footer text-muted-small">
                            <span class="figure-label">Figure 5.</span>
                            Vue globale : faibles corrélations internes, confirmant la nécessité
                            d’un modèle multivarié.
                        </div>
                    </div>
                </section>

                <!-- Correlations -->
                <section id="correlations">
                    <h2 class="section-title">2.4 Corrélations</h2>
                    <div class="row g-4">
                        <div class="col-md-6">
                            <div class="card chart-card shadow-sm mb-4">
                                <div class="card-body"><?= graph_img('chart_pearson_heatmap','Pearson',6); ?></div>
                                <div class="card-footer text-muted-small">
                                    <span class="figure-label">Figure 6.</span>
                                    Corrélations faibles ; la plus forte est
                                    <code>num_major_vessels ↔ st_depression</code> (≈ 0,6).
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card chart-card shadow-sm mb-4">
                                <div class="card-body"><?= graph_img('chart_pointbiserial_heatmap','Point‑biserial',7); ?></div>
                                <div class="card-footer text-muted-small">
                                    <span class="figure-label">Figure 7.</span>
                                    Variables numériques les plus liées à la cible :<br>
                                    • <b>num_major_vessels</b> (‑0,47)<br>
                                    • <b>max_heart_rate_achieved</b> (+0,43)<br>
                                    • <b>st_depression</b> (‑0,43)
                                </div>
                            </div>
                        </div>
                        <div class="col-12">
                            <div class="card chart-card shadow-sm mb-4">
                                <div class="card-body"><?= graph_img('chart_cramersv_heatmap','Cramér V',8); ?></div>
                                <div class="card-footer text-muted-small">
                                    <span class="figure-label">Figure 8.</span>
                                    Plus fortes associations catégorielles :  
                                    <b>chest_pain_type</b>, <b>st_slope</b>, <b>thalassemia</b>.
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                <!-- Modelling -->
                <section id="modelisation">
                    <h2 class="section-title">3. Modélisation</h2>

                    <h3 class="subsection-title">3.1 Performance des modèles</h3>
                    <div class="card shadow-sm mb-4">
                        <div class="card-body"><?= perf_table(); ?></div>
                        <div class="card-footer text-muted-small">
                            <span class="figure-label">Table 1.</span>
                            Le modèle <b>LGBM</b> optimisé obtient un rappel de <b>94 %</b>
                            et la meilleure F1‑score.
                        </div>
                    </div>

                    <h3 class="subsection-title">3.2 Courbes ROC</h3>
                    <div class="card chart-card shadow-sm mb-4">
                        <div class="card-body"><?= graph_img('chart_roc_curves','ROC',9); ?></div>
                        <div class="card-footer text-muted-small">
                            <span class="figure-label">Figure 9.</span>
                            LGBM présente l’AUC la plus élevée (« zone » ≈ 0,97).
                        </div>
                    </div>

                    <h3 class="subsection-title mt-4">3.3 Matrices de confusion</h3>
                    <div class="card chart-card shadow-sm mb-4">
                        <div class="card-body"><?= graph_img('chart_confusion_matrices','Confusion',10); ?></div>
                        <div class="card-footer text-muted-small">
                            <span class="figure-label">Figure 10.</span>
                            Comparaison visuelle des faux positifs / faux négatifs de chaque modèle.
                        </div>
                    </div>

                    <h3 class="subsection-title mt-4">3.4 Importance des variables</h3>
                    <div class="card chart-card shadow-sm mb-4">
                        <div class="card-body"><?= graph_img('chart_permutation_importance','Permutation importance',11); ?></div>
                        <div class="card-footer text-muted-small">
                            <span class="figure-label">Figure 11.</span>
                            Top 3 : <b>num_major_vessels</b>, <b>chest_pain_type</b>, <b>st_slope</b>.
                        </div>
                    </div>

                    <h3 class="subsection-title mt-4">3.5 Valeurs SHAP</h3>
                    <div class="row g-4">
                        <div class="col-md-6">
                            <div class="card chart-card shadow-sm mb-4">
                                <div class="card-body"><?= graph_img('chart_shap_bar','SHAP bar',12); ?></div>
                                <div class="card-footer text-muted-small">
                                    <span class="figure-label">Figure 12.</span>
                                    Importance moyenne des variables (SHAP) – même trio de tête.
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card chart-card shadow-sm mb-4">
                                <div class="card-body"><?= graph_img('chart_shap_summary','SHAP summary',13); ?></div>
                                <div class="card-footer text-muted-small">
                                    <span class="figure-label">Figure 13.</span>
                                    Distribution des impacts individuels : faible dispersion
                                    pour les variables secondaires.
                                </div>
                            </div>
                        </div>
                    </div>

                    <h3 class="subsection-title mt-4">3.6 Matrice de confusion LightGBM</h3>
                    <div class="card chart-card shadow-sm mb-4">
                        <div class="card-body"><?= graph_img('chart_lgbm_confusion_matrix','LGBM confusion',14); ?></div>
                        <div class="card-footer text-muted-small">
                            <span class="figure-label">Figure 14.</span>
                            LGBM optimisé : 2 faux négatifs et 1 faux positif sur le set de validation.
                        </div>
                    </div>
                </section>

                <!-- Conclusion -->
                <section id="conclusion">
                    <h2 class="section-title">4. Conclusion</h2>
                    <div class="card shadow-sm"><div class="card-body">
                        <ul>
                            <li>Le jeu de données est équilibré et exempt de valeurs manquantes après
                                nettoyage (296 cas).</li>
                            <li>Les variables les plus prédictives sont : 
                                <b>num_major_vessels</b>, <b>chest_pain_type</b>,
                                <b>st_slope</b>, <b>max_heart_rate_achieved</b> et
                                <b>st_depression</b>.</li>
                            <li>Le modèle LightGBM, après tuning, atteint un rappel de
                                <b>94 %</b> (AUC ≈ 0,97) avec très peu de faux négatifs, gage de
                                sécurité dans un contexte clinique.</li>
                            <li>Cholestérol total montre une corrélation étonnamment faible avec la
                                maladie dans cet échantillon : prudence quant à sa valeur isolée.</li>
                            <li>Pour des recommandations cliniques robustes, un
                                échantillon plus large et des données longitudinales seraient
                                nécessaires.</li>
                        </ul>
                    </div></div>
                </section>
            </main>
        </div><!-- /col-lg-9 -->
    </div><!-- /row -->
</div><!-- /container -->

<footer><div class="container-md text-center text-muted-small py-3">
    © 2025 Analyse des maladies cardiaques
</div></footer>
<script>    
    async function sendPrediction() {
        const payload = {
            age: parseInt(document.getElementById('age').value,10),
            sex: document.querySelector('input[name="sex"]:checked').value,
            chest_pain_type: document.getElementById('cp').value,
            resting_blood_pressure: parseInt(document.getElementById('rbp').value,10),
            cholesterol: parseInt(document.getElementById('chol').value,10),
            fasting_blood_sugar: document.getElementById('fbs').value,
            resting_electrocardiogram: document.getElementById('recg').value,
            max_heart_rate_achieved: parseInt(document.getElementById('mhr').value,10),
            exercise_induced_angina: document.getElementById('exang').value,
            st_depression: parseFloat(document.getElementById('oldpeak').value),
            st_slope: document.getElementById('slope').value,
            num_major_vessels: parseInt(document.getElementById('ca').value,10),
            thalassemia: document.getElementById('thal').value
        };

        const res = await fetch("http://localhost:8000/predict",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify(payload)
        });

        if(!res.ok){
            alert("Erreur : "+(await res.text()));
            return;
        }
        const data = await res.json();
        document.getElementById('result')
            .innerHTML = `Probabilité de maladie : <b>${(data.probability*100).toFixed(1)} %</b> 
                            — risque <b>${data.risk.toUpperCase()}</b>`;
    }
</script>
</body>
</html>
