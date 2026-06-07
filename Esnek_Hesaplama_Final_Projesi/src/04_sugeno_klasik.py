"""
Asama 4: Sugeno Modeli Egitimi
- Model A: En Kucuk Kareler (Least Squares) - analitik cozum
- Model B: Gradient Descent + L2 Regularizasyon
- Her LLM kural seti icin iki model egitilir
Calistirma: python src/04_sugeno_models.py
"""
import pathlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

with open(MODELS_DIR / 'mf_parameters.json', 'r') as f:
    mf_params = json.load(f)
with open(MODELS_DIR / 'rule_sets.json', 'r') as f:
    rule_sets = json.load(f)

# ============================
# 2. ÜYELİK FONKSİYONLARI
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

# ============================
# 3. SUGENO TAHMİN FONKSİYONU
# ============================
def sugeno_predict(X, rules, coefficients):
    W     = get_firing_levels(X, rules, selected_features)
    W_sum = W.sum(axis=1, keepdims=True)
    W_sum[W_sum == 0] = 1e-10
    W_norm = W / W_sum
    Z = np.zeros((X.shape[0], len(rules)))
    for i in range(len(rules)):
        p1, p2, p3, c = coefficients[i]
        Z[:, i] = p1 * X[:, 0] + p2 * X[:, 1] + p3 * X[:, 2] + c
    return np.sum(W_norm * Z, axis=1)

# ============================
# 4. MODEL A: EN KÜÇÜK KARELER
# ============================
class SugenoModelA_LeastSquares:
    """Cikis katsayilarini En Kucuk Kareler yontemiyle ogrenme."""
    def __init__(self, rules):
        self.rules = rules
        self.coefficients = None
        self.is_fitted = False

    def fit(self, X, y):
        n_samples, n_rules = X.shape[0], len(self.rules)
        W     = get_firing_levels(X, self.rules, selected_features)
        W_sum = W.sum(axis=1, keepdims=True)
        W_sum[W_sum == 0] = 1e-10
        W_norm = W / W_sum
        A = np.zeros((n_samples, n_rules * 4))
        for i in range(n_rules):
            cs = i * 4
            A[:, cs]   = W_norm[:, i] * X[:, 0]
            A[:, cs+1] = W_norm[:, i] * X[:, 1]
            A[:, cs+2] = W_norm[:, i] * X[:, 2]
            A[:, cs+3] = W_norm[:, i]
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        self.coefficients = coeffs.reshape(n_rules, 4)
        self.is_fitted = True
        return self

    def predict(self, X):
        return sugeno_predict(X, self.rules, self.coefficients)

# ============================
# 5. MODEL B: GRADIENT DESCENT + L2
# ============================
class SugenoModelB_GradientDescent:
    """Gradient Descent + L2 regularizasyon ile katsayi ogrenmesi."""
    def __init__(self, rules, learning_rate=0.5, n_epochs=10000, lambda_reg=0.001, patience=500):
        self.rules      = rules
        self.lr         = learning_rate
        self.n_epochs   = n_epochs
        self.lambda_reg = lambda_reg
        self.patience   = patience
        self.coefficients = None
        self.loss_history = []

    def fit(self, X, y):
        n_rules, n_samples = len(self.rules), X.shape[0]
        init_coeffs = np.array([[r['consequent']['p1'], r['consequent']['p2'],
                                  r['consequent']['p3'], r['consequent']['c']]
                                 for r in self.rules])
        coeffs = init_coeffs.flatten().copy()
        W     = get_firing_levels(X, self.rules, selected_features)
        W_sum = W.sum(axis=1, keepdims=True)
        W_sum[W_sum == 0] = 1e-10
        W_norm = W / W_sum
        best_loss, best_coeffs, patience_counter = float('inf'), coeffs.copy(), 0

        for epoch in range(self.n_epochs):
            C = coeffs.reshape(n_rules, 4)
            Z = np.zeros((n_samples, n_rules))
            for i in range(n_rules):
                Z[:, i] = C[i,0]*X[:,0] + C[i,1]*X[:,1] + C[i,2]*X[:,2] + C[i,3]
            y_pred = np.sum(W_norm * Z, axis=1)
            error  = y_pred - y
            mse    = np.mean(error ** 2)
            reg    = self.lambda_reg * np.sum(C[:, :3] ** 2)
            loss   = mse + reg
            self.loss_history.append(loss)
            if loss < best_loss:
                best_loss, best_coeffs, patience_counter = loss, coeffs.copy(), 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch}")
                break
            grad = np.zeros_like(coeffs)
            for i in range(n_rules):
                grad[i*4]   = (2/n_samples)*np.sum(error*W_norm[:,i]*X[:,0]) + 2*self.lambda_reg*C[i,0]
                grad[i*4+1] = (2/n_samples)*np.sum(error*W_norm[:,i]*X[:,1]) + 2*self.lambda_reg*C[i,1]
                grad[i*4+2] = (2/n_samples)*np.sum(error*W_norm[:,i]*X[:,2]) + 2*self.lambda_reg*C[i,2]
                grad[i*4+3] = (2/n_samples)*np.sum(error*W_norm[:,i])
            coeffs -= self.lr * grad
            if epoch % 2000 == 0:
                print(f"  Epoch {epoch}: Loss={loss:.6f}, MSE={mse:.6f}")

        self.coefficients = best_coeffs.reshape(n_rules, 4)
        return self

    def predict(self, X):
        return sugeno_predict(X, self.rules, self.coefficients)

# ============================
# 6. MODELLERİ EĞİT
# ============================
print("=" * 70)
print("SUGENO MODEL EGITIMI")
print("=" * 70)

results = {}

for set_name, rules in rule_sets.items():
    if len(rules) == 0:
        print(f"\n[Atlandı] {set_name} bos.")
        continue

    print(f"\n{'='*70}")
    print(f"[*] KURAL SETI: {set_name} ({len(rules)} kural)")
    print(f"{'='*70}")

    # Model A
    print("\n[-] Model A: Least Squares")
    model_a = SugenoModelA_LeastSquares(rules)
    model_a.fit(X_train, y_train)
    y_pred_a_train = model_a.predict(X_train)
    y_pred_a_test  = model_a.predict(X_test)
    y_train_real   = scaler_y.inverse_transform(y_train.reshape(-1,1)).flatten()
    y_test_real    = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()
    y_pred_a_train_real = scaler_y.inverse_transform(y_pred_a_train.reshape(-1,1)).flatten()
    y_pred_a_test_real  = scaler_y.inverse_transform(y_pred_a_test.reshape(-1,1)).flatten()
    results[f"{set_name}_ModelA"] = {
        'model': model_a,
        'y_pred_train': y_pred_a_train_real,
        'y_pred_test' : y_pred_a_test_real,
        'rmse_train'  : np.sqrt(mean_squared_error(y_train_real, y_pred_a_train_real)),
        'rmse_test'   : np.sqrt(mean_squared_error(y_test_real,  y_pred_a_test_real)),
        'mae_test'    : mean_absolute_error(y_test_real, y_pred_a_test_real),
        'mape_test'   : np.mean(np.abs((y_test_real - y_pred_a_test_real) / y_test_real)) * 100,
        'r2_test'     : r2_score(y_test_real, y_pred_a_test_real),
    }
    r = results[f"{set_name}_ModelA"]
    print(f"   Train RMSE: {r['rmse_train']:.2f}  |  Test RMSE: {r['rmse_test']:.2f}")
    print(f"   Test MAE: {r['mae_test']:.2f}  |  Test MAPE: {r['mape_test']:.2f}%  |  R2: {r['r2_test']:.4f}")

    # Model B
    print("\n[-] Model B: Gradient Descent + L2")
    model_b = SugenoModelB_GradientDescent(rules)
    model_b.fit(X_train, y_train)
    y_pred_b_train = model_b.predict(X_train)
    y_pred_b_test  = model_b.predict(X_test)
    y_pred_b_train_real = scaler_y.inverse_transform(y_pred_b_train.reshape(-1,1)).flatten()
    y_pred_b_test_real  = scaler_y.inverse_transform(y_pred_b_test.reshape(-1,1)).flatten()
    results[f"{set_name}_ModelB"] = {
        'model'       : model_b,
        'y_pred_train': y_pred_b_train_real,
        'y_pred_test' : y_pred_b_test_real,
        'rmse_train'  : np.sqrt(mean_squared_error(y_train_real, y_pred_b_train_real)),
        'rmse_test'   : np.sqrt(mean_squared_error(y_test_real,  y_pred_b_test_real)),
        'mae_test'    : mean_absolute_error(y_test_real, y_pred_b_test_real),
        'mape_test'   : np.mean(np.abs((y_test_real - y_pred_b_test_real) / y_test_real)) * 100,
        'r2_test'     : r2_score(y_test_real, y_pred_b_test_real),
        'loss_history': model_b.loss_history,
    }
    r = results[f"{set_name}_ModelB"]
    print(f"   Train RMSE: {r['rmse_train']:.2f}  |  Test RMSE: {r['rmse_test']:.2f}")
    print(f"   Test MAE: {r['mae_test']:.2f}  |  Test MAPE: {r['mape_test']:.2f}%  |  R2: {r['r2_test']:.4f}")

# ============================
# 7. ÖZET TABLO
# ============================
print("\n" + "=" * 70)
print("OZET KARSILASTIRMA TABLOSU")
print("=" * 70)
summary = [{'Model': k, 'RMSE': f"{v['rmse_test']:.2f}",
            'MAE': f"{v['mae_test']:.2f}", 'MAPE(%)': f"{v['mape_test']:.2f}",
            'R2': f"{v['r2_test']:.4f}"} for k, v in results.items()]
print(pd.DataFrame(summary).to_string(index=False))

# ============================
# 8. GRAFİK
# ============================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curve (Student ModelB)
ax = axes[0, 0]
if 'Student_ModelB' in results and 'loss_history' in results['Student_ModelB']:
    ax.plot(results['Student_ModelB']['loss_history'], lw=1.5, color='darkblue')
    ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Model B Loss Egrisi (Student)'); ax.grid(True, alpha=0.3)

# RMSE karsilastirma
ax = axes[0, 1]
names  = list(results.keys())
rmse_v = [results[m]['rmse_test'] for m in names]
ax.bar(range(len(names)), rmse_v, color=['#E74C3C' if 'A' in n else '#27AE60' for n in names])
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('RMSE'); ax.set_title('Test RMSE Karsilastirmasi'); ax.grid(True, alpha=0.3, axis='y')

# Gercek vs Tahmin (en iyi ModelB)
ax = axes[1, 0]
best_b_key = min((k for k in results if 'ModelB' in k), key=lambda k: results[k]['rmse_test'])
ax.scatter(y_test_real, results[best_b_key]['y_pred_test'], alpha=0.7, s=60, edgecolors='black')
lims = [min(y_test_real.min(), results[best_b_key]['y_pred_test'].min()),
        max(y_test_real.max(), results[best_b_key]['y_pred_test'].max())]
ax.plot(lims, lims, 'r--', lw=2, label='y=x')
ax.set_xlabel('Gercek Effort'); ax.set_ylabel('Tahmin')
ax.set_title(f'{best_b_key}: Gercek vs Tahmin'); ax.legend(); ax.grid(True, alpha=0.3)

# R2 karsilastirma
ax = axes[1, 1]
r2_v = [results[m]['r2_test'] for m in names]
colors_r2 = ['#27AE60' if v > 0 else '#E74C3C' for v in r2_v]
ax.bar(range(len(names)), r2_v, color=colors_r2)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
ax.axhline(0, color='black', lw=1); ax.set_ylabel('R2')
ax.set_title('R2 Karsilastirmasi'); ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Sugeno Model Performans Analizi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sugeno_model_analysis.png", dpi=100, bbox_inches='tight')
plt.close()
print("\n[OK] sugeno_model_analysis.png kaydedildi.")

# ============================
# 9. KATSAYILARI KAYDET
# ============================
for key, res in results.items():
    np.save(MODELS_DIR / f"coefficients_{key}.npy", res['model'].coefficients)

print("[OK] Katsayilar kaydedildi (coefficients_*.npy)")
