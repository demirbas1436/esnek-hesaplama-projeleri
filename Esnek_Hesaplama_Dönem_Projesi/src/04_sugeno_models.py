"""
Asama 4: Sugeno Modeli Egitimi - Hem Desharnais Hem Albrecht
- Model A: En Kucuk Kareler (Least Squares) - analitik cozum
- Model B: Gradient Descent + L2 Regularizasyon
- Her iki veri setindeki her LLM kural seti icin iki model egitilir
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
# ÜYELİK FONKSİYONLARI VE YARDIMCILAR
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
        Z[:, i] = p1 * X[:, 0] + p2 * X[:, 1] + p3 * X[:, 2] + c
    return np.sum(W_norm * Z, axis=1)

# ============================
# MODEL A: EN KÜÇÜK KARELER
# ============================
class SugenoModelA_LeastSquares:
    """Cikis katsayilarini En Kucuk Kareler yontemiyle ogrenme."""
    def __init__(self, rules, features, mf_params, dataset_name):
        self.rules = rules
        self.features = features
        self.mf_params = mf_params
        self.dataset_name = dataset_name
        self.coefficients = None
        self.is_fitted = False

    def fit(self, X, y):
        n_samples, n_rules = X.shape[0], len(self.rules)
        W     = get_firing_levels(X, self.rules, self.features, self.mf_params, self.dataset_name)
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
        return sugeno_predict(X, self.rules, self.coefficients, self.features, self.mf_params, self.dataset_name)

# ============================
# MODEL B: GRADIENT DESCENT + L2
# ============================
class SugenoModelB_GradientDescent:
    """Gradient Descent + L2 regularizasyon ile katsayi ogrenmesi."""
    def __init__(self, rules, features, mf_params, dataset_name, learning_rate=0.5, n_epochs=10000, lambda_reg=0.001, patience=500):
        self.rules      = rules
        self.features   = features
        self.mf_params  = mf_params
        self.dataset_name = dataset_name
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
        W     = get_firing_levels(X, self.rules, self.features, self.mf_params, self.dataset_name)
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
                break
            grad = np.zeros_like(coeffs)
            for i in range(n_rules):
                grad[i*4]   = (2/n_samples)*np.sum(error*W_norm[:,i]*X[:,0]) + 2*self.lambda_reg*C[i,0]
                grad[i*4+1] = (2/n_samples)*np.sum(error*W_norm[:,i]*X[:,1]) + 2*self.lambda_reg*C[i,1]
                grad[i*4+2] = (2/n_samples)*np.sum(error*W_norm[:,i]*X[:,2]) + 2*self.lambda_reg*C[i,2]
                grad[i*4+3] = (2/n_samples)*np.sum(error*W_norm[:,i])
            coeffs -= self.lr * grad

        self.coefficients = best_coeffs.reshape(n_rules, 4)
        return self

    def predict(self, X):
        return sugeno_predict(X, self.rules, self.coefficients, self.features, self.mf_params, self.dataset_name)

# ============================
# EĞİTİM DÖNGÜSÜ
# ============================
for dname, config in DATASETS.items():
    print("\n" + "=" * 70)
    print(f"VERI SETI: {dname.upper()} - SUGENO MODEL TROLLING")
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

    with open(MODELS_DIR / f'{dname}_mf_parameters.json', 'r') as f:
        mf_params = json.load(f)
    with open(MODELS_DIR / f'{dname}_rule_sets.json', 'r') as f:
        rule_sets = json.load(f)

    results = {}

    for set_name, rules in rule_sets.items():
        if len(rules) == 0:
            continue

        print(f"\n[*] KURAL SETI: {set_name} ({len(rules)} kural)")

        # Model A
        model_a = SugenoModelA_LeastSquares(rules, selected_features, mf_params, dname)
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
            'mape_test'   : np.mean(np.abs((y_test_real - y_pred_a_test_real) / (y_test_real + 1e-10))) * 100,
            'r2_test'     : r2_score(y_test_real, y_pred_a_test_real),
        }
        r = results[f"{set_name}_ModelA"]
        print(f"   Model A (Least Squares) - Test RMSE: {r['rmse_test']:.2f} | Test R2: {r['r2_test']:.4f}")

        # Model B
        model_b = SugenoModelB_GradientDescent(rules, selected_features, mf_params, dname)
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
            'mape_test'   : np.mean(np.abs((y_test_real - y_pred_b_test_real) / (y_test_real + 1e-10))) * 100,
            'r2_test'     : r2_score(y_test_real, y_pred_b_test_real),
            'loss_history': model_b.loss_history,
        }
        r = results[f"{set_name}_ModelB"]
        print(f"   Model B (GradientDesc)  - Test RMSE: {r['rmse_test']:.2f} | Test R2: {r['r2_test']:.4f}")

    # ============================
    # ÖZET TABLO
    # ============================
    print("\n" + "=" * 70)
    print(f"PERFORMANS OZET TABLOSU ({config['title']})")
    print("=" * 70)
    summary = [{'Model': k, 'RMSE': f"{v['rmse_test']:.2f}",
                'MAE': f"{v['mae_test']:.2f}", 'MAPE(%)': f"{v['mape_test']:.2f}",
                'R2': f"{v['r2_test']:.4f}"} for k, v in results.items()]
    print(pd.DataFrame(summary).to_string(index=False))

    # ============================
    # GRAFİKLER
    # ============================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss Egrisi
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

    plt.suptitle(f'Sugeno Model Performans Analizi ({config["title"]})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{dname}_sugeno_model_analysis.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] {dname}_sugeno_model_analysis.png kaydedildi.")

    # Katsayilari kaydet
    for key, res in results.items():
        np.save(MODELS_DIR / f"coefficients_{dname}_{key}.npy", res['model'].coefficients)
        # Geriye donuk uyumluluk icin Desharnais katsayilarini ek olarak prefixsiz kaydet
        if dname == 'desharnais':
            np.save(MODELS_DIR / f"coefficients_{key}.npy", res['model'].coefficients)

print("\n[OK] Tüm Sugeno katsayilari kaydedildi.")

# Geriye donuk uyumluluk kopyalamalari
shutil.copyfile(OUTPUT_DIR / "desharnais_sugeno_model_analysis.png", OUTPUT_DIR / "sugeno_model_analysis.png")
print("[OK] sugeno_model_analysis.png kopyalandı.")
