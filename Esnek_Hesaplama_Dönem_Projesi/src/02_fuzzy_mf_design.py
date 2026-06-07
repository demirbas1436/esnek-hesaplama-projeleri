"""
Asama 2: Fuzzy Membership Function Tasarimi
- FCM ile Gaussian MF merkez noktalarini bul
- Percentile tabanli Triangular / Trapezoidal MF parametreleri
- Girdi ve Cikti (Effort) MF grafikleri
- Parametreleri dataset_mf_parameters.json olarak kaydet
Calistirma: python src/02_fuzzy_mf_design.py
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import membership as mf
import json
import shutil

# --- Dizin Yapisi ---
BASE_DIR   = pathlib.Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams['font.size'] = 10

DATASETS_CONFIG = {
    'desharnais': {
        'train_file': 'desharnais_train_clean.csv',
        'selected_features': ['PointsAjust', 'Length', 'TeamExp'],
        'title': 'Desharnais Veri Seti'
    },
    'albrecht': {
        'train_file': 'albrecht_train_clean.csv',
        'selected_features': ['Input', 'Output', 'File'],
        'title': 'Albrecht Veri Seti'
    }
}

# ============================
# 1. FCM İLE GAUSSIAN MF MERKEZLERİ
# ============================
def find_fcm_centers(data, n_clusters=3, m=2.0, error=1e-5, maxiter=1000):
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data.reshape(1, -1), n_clusters, m, error=error, maxiter=maxiter
    )
    return np.sort(cntr.flatten())

# ============================
# 2. PERCENTILE PARAMETRELER
# ============================
def get_all_params(data):
    p25, p40, p50, p60, p75 = np.percentile(data, [25, 40, 50, 60, 75])
    return {
        'tri_low':   [0,    p25,  p50],
        'tri_med':   [p25,  p50,  p75],
        'tri_high':  [p50,  p75,  1  ],
        'trap_low':  [0,    0,    p25,  p50],
        'trap_med':  [p25,  p40,  p60,  p75],
        'trap_high': [p50,  p75,  1,    1  ]
    }

for dataset_name, config in DATASETS_CONFIG.items():
    print("\n" + "=" * 60)
    print(f"VERI SETI: {dataset_name.upper()}")
    print("=" * 60)
    
    train_df = pd.read_csv(MODELS_DIR / config['train_file'])
    selected_features = config['selected_features']
    X_fuzzy = train_df[selected_features]
    y_effort = train_df['Effort']
    
    print("\nSECILEN DEGISKENLERIN ISTATISTIGI (NORMALIZE [0,1])")
    print(X_fuzzy.describe())
    
    # FCM merkezleri
    fcm_centers = {}
    for col in selected_features:
        centers = find_fcm_centers(X_fuzzy[col].values)
        fcm_centers[col] = centers
        print(f"{col}: Low={centers[0]:.3f}, Med={centers[1]:.3f}, High={centers[2]:.3f}")
        
    effort_centers = find_fcm_centers(y_effort.values)
    print(f"Effort (output): Low={effort_centers[0]:.3f}, Med={effort_centers[1]:.3f}, High={effort_centers[2]:.3f}")
    
    # Percentile parametreleri
    perc_params = {}
    for col in selected_features:
        params = get_all_params(X_fuzzy[col].values)
        perc_params[col] = params
        print(f"\n{col}:")
        for k, v in params.items():
            print(f"  {k}: {[round(x,3) for x in v]}")
            
    effort_params = get_all_params(y_effort.values)
    perc_params['Effort'] = effort_params
    
    # Girdi MF Grafikleri
    x = np.linspace(0, 1, 1000)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f'Giris Degiskenleri - Uyelik Fonksiyonlari\n({config["title"]})',
                 fontsize=16, fontweight='bold')
    colors = {'Low': '#E74C3C', 'Medium': '#F39C12', 'High': '#27AE60'}
    
    for row_idx, col_name in enumerate(selected_features):
        centers = fcm_centers[col_name]
        params  = perc_params[col_name]
        sigma   = max(np.mean(np.diff(centers)) / 2.5, 0.05)
        
        # Gaussian
        ax = axes[row_idx, 0]
        gauss_vals = {
            'Low':    np.exp(-0.5 * ((x - centers[0]) / sigma) ** 2),
            'Medium': np.exp(-0.5 * ((x - centers[1]) / sigma) ** 2),
            'High':   np.exp(-0.5 * ((x - centers[2]) / sigma) ** 2),
        }
        for label, vals in gauss_vals.items():
            ax.plot(x, vals, color=colors[label], linewidth=2.5, label=f'{label} (mu={centers[list(gauss_vals).index(label)]:.2f})')
            ax.fill_between(x, 0, vals, color=colors[label], alpha=0.12)
        ax.set_title(f'{col_name} - Gaussian', fontweight='bold')
        ax.set_ylim(0, 1.15); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        if row_idx == 2: ax.set_xlabel('Normalize Deger [0, 1]')
        
        # Triangular
        ax = axes[row_idx, 1]
        for label, key in [('Low','tri_low'), ('Medium','tri_med'), ('High','tri_high')]:
            y_mf = mf.trimf(x, params[key])
            ax.plot(x, y_mf, color=colors[label], linewidth=2.5, label=label)
            ax.fill_between(x, 0, y_mf, color=colors[label], alpha=0.12)
        ax.set_title(f'{col_name} - Triangular', fontweight='bold')
        ax.set_ylim(0, 1.15); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        if row_idx == 2: ax.set_xlabel('Normalize Deger [0, 1]')
        
        # Trapezoidal
        ax = axes[row_idx, 2]
        for label, key in [('Low','trap_low'), ('Medium','trap_med'), ('High','trap_high')]:
            y_mf = mf.trapmf(x, params[key])
            ax.plot(x, y_mf, color=colors[label], linewidth=2.5, label=label)
            ax.fill_between(x, 0, y_mf, color=colors[label], alpha=0.12)
        ax.set_title(f'{col_name} - Trapezoidal', fontweight='bold')
        ax.set_ylim(0, 1.15); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        if row_idx == 2: ax.set_xlabel('Normalize Deger [0, 1]')
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{dataset_name}_membership_functions_input.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] {dataset_name}_membership_functions_input.png kaydedildi.")
    
    # Cikis (Effort) MF Grafigi
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Cikis Degiskeni (Effort) - Uyelik Fonksiyonlari\n({config["title"]})',
                 fontsize=14, fontweight='bold')
    
    ep = effort_params
    ec = effort_centers
    e_sigma = max(np.mean(np.diff(ec)) / 2.5, 0.05)
    
    # Gaussian
    ax = axes[0]
    for label, c in zip(['Low','Medium','High'], ec):
        vals = np.exp(-0.5 * ((x - c) / e_sigma) ** 2)
        ax.plot(x, vals, color=colors[label], linewidth=2.5, label=f'{label} (mu={c:.2f})')
        ax.fill_between(x, 0, vals, color=colors[label], alpha=0.12)
    ax.set_title('Effort - Gaussian', fontweight='bold')
    ax.set_xlabel('Normalize Effort [0,1]'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    
    # Triangular
    ax = axes[1]
    for label, key in [('Low','tri_low'),('Medium','tri_med'),('High','tri_high')]:
        y_mf = mf.trimf(x, ep[key])
        ax.plot(x, y_mf, color=colors[label], linewidth=2.5, label=label)
        ax.fill_between(x, 0, y_mf, color=colors[label], alpha=0.12)
    ax.set_title('Effort - Triangular', fontweight='bold')
    ax.set_xlabel('Normalize Effort [0,1]'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    
    # Trapezoidal
    ax = axes[2]
    for label, key in [('Low','trap_low'),('Medium','trap_med'),('High','trap_high')]:
        y_mf = mf.trapmf(x, ep[key])
        ax.plot(x, y_mf, color=colors[label], linewidth=2.5, label=label)
        ax.fill_between(x, 0, y_mf, color=colors[label], alpha=0.12)
    ax.set_title('Effort - Trapezoidal', fontweight='bold')
    ax.set_xlabel('Normalize Effort [0,1]'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{dataset_name}_membership_functions_output.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] {dataset_name}_membership_functions_output.png kaydedildi.")
    
    # Parametreleri Kaydet
    mf_config = {'features': selected_features, 'gaussian': {}, 'triangular': {}, 'trapezoidal': {}}
    for col in selected_features:
        centers = fcm_centers[col]
        sigma   = max(np.mean(np.diff(centers)) / 2.5, 0.05)
        mf_config['gaussian'][col] = {
            'Low':    {'center': float(centers[0]), 'sigma': float(sigma)},
            'Medium': {'center': float(centers[1]), 'sigma': float(sigma)},
            'High':   {'center': float(centers[2]), 'sigma': float(sigma)},
        }
        p = perc_params[col]
        mf_config['triangular'][col]  = {'Low': p['tri_low'],  'Medium': p['tri_med'],  'High': p['tri_high']}
        mf_config['trapezoidal'][col] = {'Low': p['trap_low'], 'Medium': p['trap_med'], 'High': p['trap_high']}
        
    mf_config_clean = json.loads(json.dumps(mf_config, default=lambda o: o.item() if isinstance(o, np.generic) else o))
    
    # Veri setine ozel parametreleri kaydet
    with open(MODELS_DIR / f'{dataset_name}_mf_parameters.json', 'w') as f:
        json.dump(mf_config_clean, f, indent=2)
    print(f"[OK] {dataset_name}_mf_parameters.json kaydedildi.")

# Geriye donuk uyumluluk (backward compatibility) icin Desharnais sonuclarini ana dosya olarak kopyala
shutil.copyfile(MODELS_DIR / "desharnais_mf_parameters.json", MODELS_DIR / "mf_parameters.json")
shutil.copyfile(OUTPUT_DIR / "desharnais_membership_functions_input.png", OUTPUT_DIR / "membership_functions_input.png")
shutil.copyfile(OUTPUT_DIR / "desharnais_membership_functions_output.png", OUTPUT_DIR / "membership_functions_output.png")
print("\n[OK] Geriye donuk uyumluluk dosyalari olusturuldu (mf_parameters.json vb.)")
