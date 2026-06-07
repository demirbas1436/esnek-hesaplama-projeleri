"""
Asama 5: Performans Analizi ve Karsilastirma - Hem Desharnais Hem Albrecht
- Sugeno modelleri vs Linear Regression vs Decision Tree vs KMeans Regressor
- 5-Fold Cross-Validation (Klasik ML modelleri icin)
- Wilcoxon istatistiksel anlamsallik testi
Calistirma: python src/05_performance_comparison.py
"""
import pathlib
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans

# Existing imports and code...
# (Assume other necessary imports are already present above)

# -------------------------------------------------------------------------
# KMeansRegressor: simple clustering based regressor
# -------------------------------------------------------------------------
class KMeansRegressor(BaseEstimator, RegressorMixin):
    """K-Means based regressor.

    The model clusters the feature space into *n_clusters* groups and predicts
    the mean target value of the cluster to which a sample belongs.
    """
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.cluster_means_ = None

    def fit(self, X, y):
        # Fit KMeans on features
        self.kmeans.fit(X)
        # Assign each sample to a cluster
        labels = self.kmeans.labels_
        # Compute mean target per cluster
        self.cluster_means_ = {}
        for lbl in np.unique(labels):
            self.cluster_means_[lbl] = y[labels == lbl].mean()
        return self

    def predict(self, X):
        labels = self.kmeans.predict(X)
        # Map cluster label to mean target
        preds = np.array([self.cluster_means_[lbl] for lbl in labels])
        return preds

# -------------------------------------------------------------------------
# Helper: compute boxplot (IQR) categories for a column
# -------------------------------------------------------------------------
def get_iqr_category(value, series: pd.Series):
    """Return 'Düşük', 'Orta', or 'Yüksek' based on IQR quartiles.

    Low  <= Q1
    Medium between Q1 and Q3
    High >= Q3
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    if value <= q1:
        return "Düşük"
    elif value >= q3:
        return "Yüksek"
    else:
        return "Orta"

# -------------------------------------------------------------------------
# Updated evaluation routine to include KMeansRegressor
# -------------------------------------------------------------------------
def evaluate_models(X_train, X_test, y_train, y_test, dataset_name: str):
    results = []
    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lr = lin_reg.predict(X_test)
    results.append(("LinearRegression", y_pred_lr))
    # Decision Tree
    dt_reg = DecisionTreeRegressor(random_state=42)
    dt_reg.fit(X_train, y_train)
    y_pred_dt = dt_reg.predict(X_test)
    results.append(("DecisionTree", y_pred_dt))
    # KMeans Regressor (baseline clustering)
    km_reg = KMeansRegressor(n_clusters=4, random_state=42)
    km_reg.fit(X_train, y_train)
    y_pred_km = km_reg.predict(X_test)
    results.append(("KMeansRegressor", y_pred_km))
    # Sugeno models are already evaluated elsewhere; they are added separately.
    return results

# The rest of the script (loading data, computing metrics, generating tables) should
# incorporate the new "KMeansRegressor" entry where other model results are processed.

# -------------------------------------------------------------------------
# NOTE: The above additions are placed near the top of the file after imports.
# Ensure that later sections that compute metric tables iterate over the new entry.
# -------------------------------------------------------------------------
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import wilcoxon
import joblib
import shutil

# --- Dizin Yapisi ---
BASE_DIR   = pathlib.Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = {
    'desharnais': {
        'train_file': 'desharnais_train_clean.csv',
        'test_file': 'desharnais_test_clean.csv',
        'selected_features': ['PointsAjust', 'Length', 'TeamExp'],
        'title': 'Desharnais'
    },
    'albrecht': {
        'train_file': 'albrecht_train_clean.csv',
        'test_file': 'albrecht_test_clean.csv',
        'selected_features': ['Input', 'Output', 'File'],
        'title': 'Albrecht'
    }
}

# ============================
# K-MEANS REGRESSOR MODELİ
# ============================
class KMeansRegressor:
    """K-Means kumeleme tabanli regresyon modeli.
    Egitim verisini kume merkezlerine ayirir ve her kumedeki efor ortalamasini tutar.
    Yeni veriler icin en yakin kumenin ortalama efor degerini doner.
    """
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        self.cluster_means = {}

    def fit(self, X, y):
        clusters = self.kmeans.fit_predict(X)
        for i in range(self.n_clusters):
            mask = (clusters == i)
            if np.sum(mask) > 0:
                self.cluster_means[i] = np.mean(y[mask])
            else:
                self.cluster_means[i] = np.mean(y)
        return self

    def predict(self, X):
        clusters = self.kmeans.predict(X)
        return np.array([self.cluster_means[c] for c in clusters])

# ============================
# SUGENO TAHMİN FONKSİYONLARI
# ============================
def trimf_membership(x, params):
    a, b, c = params
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    else:
        return (c - x) / (c - b) if c != b else 1.0

def get_firing_levels(X, rules, features, mf_params, dataset_name):
    n_samples, n_rules = X.shape[0], len(rules)
    W = np.zeros((n_samples, n_rules))
    synonyms = {"Short": "Low", "Long": "High"} if dataset_name == "desharnais" else {}
    for i, rule in enumerate(rules):
        for j in range(n_samples):
            levels = []
            for k, feat in enumerate(features):
                val     = X[j, k]
                mf_name = synonyms.get(rule['antecedent'][k], rule['antecedent'][k])
                params  = mf_params['triangular'][feat][mf_name]
                levels.append(trimf_membership(val, params))
            W[j, i] = np.prod(levels)
    return W

def sugeno_predict(X, rules, coefficients, features, mf_params, dataset_name):
    W     = get_firing_levels(X, rules, features, mf_params, dataset_name)
    W_sum = W.sum(axis=1, keepdims=True)
    W_sum[W_sum == 0] = 1e-10
    W_norm = W / W_sum
    Z = np.zeros((X.shape[0], len(rules)))
    for i in range(len(rules)):
        p1, p2, p3, c = coefficients[i]
        Z[:, i] = p1*X[:,0] + p2*X[:,1] + p3*X[:,2] + c
    return np.sum(W_norm * Z, axis=1)

def load_sugeno_predictions(model_name, dataset_name, rule_sets, X_train, X_test, mf_params, features):
    set_name = model_name.split('_')[0]
    rules  = rule_sets[set_name]
    coeffs = np.load(MODELS_DIR / f"coefficients_{dataset_name}_{model_name}.npy")
    y_ptr  = sugeno_predict(X_train, rules, coeffs, features, mf_params, dataset_name)
    y_pte  = sugeno_predict(X_test,  rules, coeffs, features, mf_params, dataset_name)
    return y_ptr, y_pte

def evaluate_model(y_true, y_pred, model_name):
    rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
    mae      = mean_absolute_error(y_true, y_pred)
    mape     = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2       = r2_score(y_true, y_pred)
    accuracy = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)) <= 0.25) * 100
    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape,
            'R2': r2, 'Accuracy (+-25pct)': accuracy}

# ============================
# ANA DEĞERLENDİRME DÖNGÜSÜ
# ============================
for dname, config in DATASETS.items():
    print("\n" + "=" * 70)
    print(f"VERI SETI: {dname.upper()} - KLASIK VE SUGENO MODELLERI KARSILASTIRMASI")
    print("=" * 70)

    train_df = pd.read_csv(MODELS_DIR / config['train_file'])
    test_df  = pd.read_csv(MODELS_DIR / config['test_file'])
    selected_features = config['selected_features']
    target = 'Effort'

    X_train = train_df[selected_features].values
    y_train = train_df[target].values
    X_test  = test_df[selected_features].values
    y_test  = test_df[target].values

    scaler_y = joblib.load(MODELS_DIR / f"{dname}_scaler_y.pkl")
    y_train_real = scaler_y.inverse_transform(y_train.reshape(-1,1)).flatten()
    y_test_real  = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()

    with open(MODELS_DIR / f'{dname}_mf_parameters.json', 'r') as f:
        mf_params = json.load(f)
    with open(MODELS_DIR / f'{dname}_rule_sets.json', 'r') as f:
        rule_sets = json.load(f)

    # Klasik ML modelleri
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr_test_real = scaler_y.inverse_transform(lr.predict(X_test).reshape(-1,1)).flatten()

    dt = DecisionTreeRegressor(max_depth=5, min_samples_split=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt_test_real = scaler_y.inverse_transform(dt.predict(X_test).reshape(-1,1)).flatten()

    # K-Means Kümeleme Modeli
    km = KMeansRegressor(n_clusters=3, random_state=42)
    km.fit(X_train, y_train)
    y_pred_km_test_real = scaler_y.inverse_transform(km.predict(X_test).reshape(-1,1)).flatten()

    all_results = []
    sugeno_models = []
    for sn in rule_sets:
        if len(rule_sets[sn]) > 0:
            sugeno_models.extend([f"{sn}_ModelA", f"{sn}_ModelB"])

    # Sugeno Modelleri Degerlendir
    for sm in sugeno_models:
        try:
            _, y_pte = load_sugeno_predictions(sm, dname, rule_sets, X_train, X_test, mf_params, selected_features)
            y_pte_real = scaler_y.inverse_transform(y_pte.reshape(-1,1)).flatten()
            all_results.append(evaluate_model(y_test_real, y_pte_real, sm))
        except Exception as e:
            print(f"[Hata] {sm} yuklenirken hata: {e}")

    # Klasik modeller ekle
    all_results.append(evaluate_model(y_test_real, y_pred_lr_test_real, 'LinearRegression'))
    all_results.append(evaluate_model(y_test_real, y_pred_dt_test_real, 'DecisionTree'))
    all_results.append(evaluate_model(y_test_real, y_pred_km_test_real, 'KMeansRegressor'))

    results_df = pd.DataFrame(all_results)

    print("\nTEST SETI PERFORMANS TABLOSU")
    print(results_df.round(3).to_string(index=False))

    # Cross-Validation (Klasik modeller için normalized uzayda)
    print("\n5-FOLD CROSS-VALIDATION (Klasik Modeller)")
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for ModelClass, params, name in [
        (LinearRegression, {}, 'LinearRegression'),
        (DecisionTreeRegressor, {'max_depth': 5, 'min_samples_split': 5, 'random_state': 42}, 'DecisionTree'),
        (KMeansRegressor, {'n_clusters': 3, 'random_state': 42}, 'KMeansRegressor')
    ]:
        rmse_scores = []
        for tr_idx, val_idx in kf.split(X_all):
            m = ModelClass(**params)
            m.fit(X_all[tr_idx], y_all[tr_idx])
            preds = m.predict(X_all[val_idx])
            rmse_scores.append(np.sqrt(mean_squared_error(y_all[val_idx], preds)))
        cv_results.append({'Model': name, 'CV_RMSE_Mean': np.mean(rmse_scores), 'CV_RMSE_Std': np.std(rmse_scores)})
        print(f"   {name}: CV RMSE = {np.mean(rmse_scores):.4f} (+/- {np.std(rmse_scores):.4f})")

    cv_df = pd.DataFrame(cv_results)

    # Wilcoxon Istatistiksel Test
    print("\nWILCOXON ISTATISTIKSEL TEST")
    sugeno_only = results_df[results_df['Model'].str.contains('Model')]
    if not sugeno_only.empty:
        best_sugeno_name = sugeno_only.loc[sugeno_only['R2'].idxmax(), 'Model']
        _, y_pte_best = load_sugeno_predictions(best_sugeno_name, dname, rule_sets, X_train, X_test, mf_params, selected_features)
        y_pte_best_real = scaler_y.inverse_transform(y_pte_best.reshape(-1,1)).flatten()
        errors_sugeno = np.abs(y_test_real - y_pte_best_real)
        errors_lr     = np.abs(y_test_real - y_pred_lr_test_real)
        if len(errors_sugeno) == len(errors_lr):
            stat, p_val = wilcoxon(errors_sugeno, errors_lr)
            print(f"   Wilcoxon Test ({best_sugeno_name} vs LinearRegression):")
            print(f"     Istatistik: {stat:.4f} | p-degeri: {p_val:.4f}")
            if p_val < 0.05:
                print("     [OK] Fark istatistiksel olarak ANLAMLI (p < 0.05)")
            else:
                print("     [!] Fark istatistiksel olarak ANLAMLISIZ (p >= 0.05)")

    # Kaydet
    results_df.to_csv(OUTPUT_DIR / f"{dname}_performance_results.csv", index=False)
    cv_df.to_csv(OUTPUT_DIR / f"{dname}_cv_results.csv", index=False)
    print(f"\n[OK] {dname}_performance_results.csv kaydedildi.")

    # Görselleştirme
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. RMSE Bar
    ax = axes[0, 0]
    ax.bar(range(len(results_df)), results_df['RMSE'],
           color=['#E74C3C' if 'Model' in n else '#3498DB' for n in results_df['Model']])
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('RMSE'); ax.set_title('RMSE (Test Seti)'); ax.grid(True, alpha=0.3, axis='y')

    # 2. R2 Bar
    ax = axes[0, 1]
    colors_r2 = ['#27AE60' if v >= 0 else '#E74C3C' for v in results_df['R2']]
    ax.bar(range(len(results_df)), results_df['R2'], color=colors_r2)
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=7)
    ax.axhline(0, color='black', lw=1); ax.set_ylabel('R2')
    ax.set_title('R2 Karsilastirmasi'); ax.grid(True, alpha=0.3, axis='y')

    # 3. Accuracy Bar
    ax = axes[0, 2]
    ax.bar(range(len(results_df)), results_df['Accuracy (+-25pct)'],
           color=['#F39C12' if 'Model' in n else '#9B59B6' for n in results_df['Model']])
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Accuracy (%)'); ax.set_title('+-25% Hata Bantinda Dogruluk')
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.3, axis='y')

    # 4. Residual Plot
    if not sugeno_only.empty:
        ax = axes[1, 0]
        residuals_s = y_test_real - y_pte_best_real
        residuals_lr = y_test_real - y_pred_lr_test_real
        ax.scatter(y_pte_best_real, residuals_s, alpha=0.7, s=60, label=best_sugeno_name, color='tomato')
        ax.scatter(y_pred_lr_test_real, residuals_lr, alpha=0.7, s=60, label='LR', color='steelblue')
        ax.axhline(0, color='black', lw=2, ls='--')
        ax.set_xlabel('Tahmin'); ax.set_ylabel('Residual')
        ax.set_title('Residual Plot'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5. MAE Bar
    ax = axes[1, 1]
    ax.bar(range(len(results_df)), results_df['MAE'],
           color=['#E67E22' if 'Model' in n else '#1ABC9C' for n in results_df['Model']])
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('MAE'); ax.set_title('MAE Karsilastirmasi'); ax.grid(True, alpha=0.3, axis='y')

    # 6. Feature Importance (Decision Tree)
    ax = axes[1, 2]
    importance = dt.feature_importances_
    ax.bar(selected_features, importance, color=['#E74C3C', '#3498DB', '#27AE60'], edgecolor='black')
    for i, (feat, imp) in enumerate(zip(selected_features, importance)):
        ax.text(i, imp + 0.01, f'{imp:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Onem'); ax.set_title('Decision Tree - Ozellik Onemi'); ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Kapsamli Performans Analizi ({config["title"]} Seti)\n(Sugeno vs Klasik ML Modelleri)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{dname}_performance_comparison.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[OK] {dname}_performance_comparison.png kaydedildi.")

# Geriye donuk uyumluluk dosyalari
shutil.copyfile(OUTPUT_DIR / "desharnais_performance_results.csv", OUTPUT_DIR / "performance_results.csv")
shutil.copyfile(OUTPUT_DIR / "desharnais_cv_results.csv", OUTPUT_DIR / "cv_results.csv")
shutil.copyfile(OUTPUT_DIR / "desharnais_performance_comparison.png", OUTPUT_DIR / "performance_comparison.png")
print("\n[OK] Geriye donuk uyumluluk dosyalari olusturuldu (performance_results.csv vb.)")
