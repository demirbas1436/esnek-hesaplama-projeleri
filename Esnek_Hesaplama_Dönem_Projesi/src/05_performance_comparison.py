"""
Asama 5: Performans Analizi ve Karsilastirma
- Sugeno modelleri vs Linear Regression vs Decision Tree
- 5-Fold Cross-Validation
- Wilcoxon istatistiksel anlamsallik testi
Calistirma: python src/05_performance_comparison.py
"""
import pathlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import wilcoxon
import joblib

# --- Dizin Yapisi ---
BASE_DIR   = pathlib.Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================
# 1. VERİYİ YÜKLE
# ============================
train_df = pd.read_csv(MODELS_DIR / "desharnais_train_clean.csv")
test_df  = pd.read_csv(MODELS_DIR / "desharnais_test_clean.csv")

selected_features = ['PointsAjust', 'Length', 'TeamExp']
target = 'Effort'

X_train = train_df[selected_features].values
y_train = train_df[target].values
X_test  = test_df[selected_features].values
y_test  = test_df[target].values

scaler_y = joblib.load(MODELS_DIR / "scaler_y.pkl")
y_train_real = scaler_y.inverse_transform(y_train.reshape(-1,1)).flatten()
y_test_real  = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()

with open(MODELS_DIR / 'mf_parameters.json', 'r') as f:
    mf_params = json.load(f)
with open(MODELS_DIR / 'rule_sets.json', 'r') as f:
    rule_sets = json.load(f)

# ============================
# 2. SUGENO TAHMİN FONKSİYONU (yerel kopya)
# ============================
synonyms = {"Short": "Low", "Long": "High"}

def trimf_membership(x, params):
    a, b, c = params
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    else:
        return (c - x) / (c - b) if c != b else 1.0

def get_firing_levels(X, rules, features):
    n_samples, n_rules = X.shape[0], len(rules)
    W = np.zeros((n_samples, n_rules))
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

def sugeno_predict(X, rules, coefficients):
    W     = get_firing_levels(X, rules, selected_features)
    W_sum = W.sum(axis=1, keepdims=True)
    W_sum[W_sum == 0] = 1e-10
    W_norm = W / W_sum
    Z = np.zeros((X.shape[0], len(rules)))
    for i in range(len(rules)):
        p1, p2, p3, c = coefficients[i]
        Z[:, i] = p1*X[:,0] + p2*X[:,1] + p3*X[:,2] + c
    return np.sum(W_norm * Z, axis=1)

def load_sugeno_predictions(model_name):
    set_name = model_name.split('_')[0]
    rules  = rule_sets[set_name]
    coeffs = np.load(MODELS_DIR / f"coefficients_{model_name}.npy")
    y_ptr  = sugeno_predict(X_train, rules, coeffs)
    y_pte  = sugeno_predict(X_test,  rules, coeffs)
    return y_ptr, y_pte

# ============================
# 3. KLASİK ML MODELLERİ
# ============================
print("=" * 70)
print("KLASIK ML MODELLERINI EGIT")
print("=" * 70)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr_test_real = scaler_y.inverse_transform(lr.predict(X_test).reshape(-1,1)).flatten()
print("[-] Linear Regression egitildi.")

dt = DecisionTreeRegressor(max_depth=5, min_samples_split=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt_test_real = scaler_y.inverse_transform(dt.predict(X_test).reshape(-1,1)).flatten()
print("[-] Decision Tree egitildi.")

# ============================
# 4. TÜM MODELLERİ DEĞERLENDİR
# ============================
def evaluate_model(y_true, y_pred, model_name):
    rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
    mae      = mean_absolute_error(y_true, y_pred)
    mape     = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2       = r2_score(y_true, y_pred)
    accuracy = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)) <= 0.25) * 100
    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape,
            'R2': r2, 'Accuracy (+-25pct)': accuracy}

all_results = []

# Sugeno modelleri
sugeno_models = []
for sn in rule_sets:
    if len(rule_sets[sn]) > 0:
        sugeno_models.extend([f"{sn}_ModelA", f"{sn}_ModelB"])

print("\n" + "=" * 70)
print("SUGENO MODELLERI DEGERLENDIRILIYOR")
print("=" * 70)

for sm in sugeno_models:
    try:
        _, y_pte = load_sugeno_predictions(sm)
        y_pte_real = scaler_y.inverse_transform(y_pte.reshape(-1,1)).flatten()
        all_results.append(evaluate_model(y_test_real, y_pte_real, sm))
        print(f"[OK] {sm} degerlendirdi.")
    except Exception as e:
        print(f"[Hata] {sm}: {e}")

# Klasik modeller
all_results.append(evaluate_model(y_test_real, y_pred_lr_test_real, 'LinearRegression'))
all_results.append(evaluate_model(y_test_real, y_pred_dt_test_real, 'DecisionTree'))
print("[OK] LinearRegression & DecisionTree degerlendirdi.")

results_df = pd.DataFrame(all_results)

print("\n" + "=" * 70)
print("TEST SETI PERFORMANS TABLOSU")
print("=" * 70)
print(results_df.round(3).to_string(index=False))

# ============================
# 5. CROSS-VALIDATION
# ============================
print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION (Klasik Modeller)")
print("=" * 70)

X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for ModelClass, params, name in [
    (LinearRegression, {}, 'LinearRegression'),
    (DecisionTreeRegressor, {'max_depth': 5, 'min_samples_split': 5, 'random_state': 42}, 'DecisionTree')
]:
    rmse_scores = []
    for tr_idx, val_idx in kf.split(X_all):
        m = ModelClass(**params)
        m.fit(X_all[tr_idx], y_all[tr_idx])
        preds = m.predict(X_all[val_idx])
        rmse_scores.append(np.sqrt(mean_squared_error(y_all[val_idx], preds)))
    cv_results.append({'Model': name, 'CV_RMSE_Mean': np.mean(rmse_scores), 'CV_RMSE_Std': np.std(rmse_scores)})
    print(f"{name}: CV RMSE = {np.mean(rmse_scores):.4f} (+/- {np.std(rmse_scores):.4f})")

cv_df = pd.DataFrame(cv_results)

# ============================
# 6. İSTATİSTİKSEL TEST
# ============================
print("\n" + "=" * 70)
print("WILCOXON ISTATISTIKSEL TEST")
print("=" * 70)

sugeno_only = results_df[results_df['Model'].str.contains('Model')]
if not sugeno_only.empty:
    best_sugeno_name = sugeno_only.loc[sugeno_only['R2'].idxmax(), 'Model']
    _, y_pte_best = load_sugeno_predictions(best_sugeno_name)
    y_pte_best_real = scaler_y.inverse_transform(y_pte_best.reshape(-1,1)).flatten()
    errors_sugeno = np.abs(y_test_real - y_pte_best_real)
    errors_lr     = np.abs(y_test_real - y_pred_lr_test_real)
    if len(errors_sugeno) == len(errors_lr):
        stat, p_val = wilcoxon(errors_sugeno, errors_lr)
        print(f"\nWilcoxon Test ({best_sugeno_name} vs LinearRegression):")
        print(f"  Istatistik: {stat:.4f}")
        print(f"  p-degeri  : {p_val:.4f}")
        if p_val < 0.05:
            print("  [OK] Fark istatistiksel olarak ANLAMLI (p < 0.05)")
        else:
            print("  [!] Fark istatistiksel olarak ANLAMLISIZ (p >= 0.05)")

# ============================
# 7. KAYDET
# ============================
results_df.to_csv(OUTPUT_DIR / "performance_results.csv", index=False)
cv_df.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)
print(f"\n[OK] performance_results.csv kaydedildi.")
print(f"[OK] cv_results.csv kaydedildi.")

# ============================
# 8. GÖRSELLEŞTİRME
# ============================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 1. RMSE bar
ax = axes[0, 0]
ax.bar(range(len(results_df)), results_df['RMSE'],
       color=['#E74C3C' if 'Model' in n else '#3498DB' for n in results_df['Model']])
ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('RMSE'); ax.set_title('RMSE (Test Seti)'); ax.grid(True, alpha=0.3, axis='y')

# 2. R2 bar
ax = axes[0, 1]
colors_r2 = ['#27AE60' if v >= 0 else '#E74C3C' for v in results_df['R2']]
bars = ax.bar(range(len(results_df)), results_df['R2'], color=colors_r2)
ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=7)
ax.axhline(0, color='black', lw=1); ax.set_ylabel('R2')
ax.set_title('R2 Karsilastirmasi'); ax.grid(True, alpha=0.3, axis='y')

# 3. Accuracy bar
ax = axes[0, 2]
ax.bar(range(len(results_df)), results_df['Accuracy (+-25pct)'],
       color=['#F39C12' if 'Model' in n else '#9B59B6' for n in results_df['Model']])
ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Accuracy (%)'); ax.set_title('+-25% Hata Bantinda Dogruluk')
ax.set_ylim(0, 100); ax.grid(True, alpha=0.3, axis='y')

# 4. Residual plot
if not sugeno_only.empty:
    ax = axes[1, 0]
    residuals_s = y_test_real - y_pte_best_real
    residuals_lr = y_test_real - y_pred_lr_test_real
    ax.scatter(y_pte_best_real, residuals_s, alpha=0.7, s=60, label=best_sugeno_name, color='tomato')
    ax.scatter(y_pred_lr_test_real, residuals_lr, alpha=0.7, s=60, label='LR', color='steelblue')
    ax.axhline(0, color='black', lw=2, ls='--')
    ax.set_xlabel('Tahmin'); ax.set_ylabel('Residual')
    ax.set_title('Residual Plot'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 5. MAE karsilastirma
ax = axes[1, 1]
ax.bar(range(len(results_df)), results_df['MAE'],
       color=['#E67E22' if 'Model' in n else '#1ABC9C' for n in results_df['Model']])
ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('MAE'); ax.set_title('MAE Karsilastirmasi'); ax.grid(True, alpha=0.3, axis='y')

# 6. Feature importance (DT)
ax = axes[1, 2]
importance = dt.feature_importances_
ax.bar(selected_features, importance, color=['#E74C3C', '#3498DB', '#27AE60'], edgecolor='black')
for i, (feat, imp) in enumerate(zip(selected_features, importance)):
    ax.text(i, imp + 0.01, f'{imp:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Onem'); ax.set_title('Decision Tree - Ozellik Onemi'); ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Kapsamli Performans Analizi\n(Sugeno vs Klasik ML Modelleri)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "performance_comparison.png", dpi=100, bbox_inches='tight')
plt.close()
print("[OK] performance_comparison.png kaydedildi.")
