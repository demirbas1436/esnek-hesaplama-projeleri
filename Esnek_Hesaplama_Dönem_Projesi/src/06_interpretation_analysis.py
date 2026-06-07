"""
Asama 6: Model Yorumlama ve Aciklanabilirlik Analizi - Hem Desharnais Hem Albrecht
- En iyi ateslenen kurallarin analizi (Baskin kurallar)
- Katsayilarin is isaretleri ve yorumu (Takim tecrubesi vs Dosya sayisi)
- Senaryo bazli tahmin aciklanabilirligi (Kucuk vs Buyuk projeler)
Calistirma: python src/06_interpretation_analysis.py
"""
import pathlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import shutil

# --- Dizin Yapisi ---
BASE_DIR   = pathlib.Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = {
    'desharnais': {
        'selected_features': ['PointsAjust', 'Length', 'TeamExp'],
        'scenarios': {
            "Kucuk_Proje": np.array([0.15, 0.20, 0.80]),  # [PointsAjust, Length, TeamExp]
            "Buyuk_Proje": np.array([0.85, 0.70, 0.15])   # [PointsAjust, Length, TeamExp]
        },
        'title': 'Desharnais'
    },
    'albrecht': {
        'selected_features': ['Input', 'Output', 'File'],
        'scenarios': {
            "Kucuk_Proje": np.array([0.15, 0.15, 0.15]),  # [Input, Output, File]
            "Buyuk_Proje": np.array([0.85, 0.85, 0.85])   # [Input, Output, File]
        },
        'title': 'Albrecht'
    }
}

# ============================
# YARDIMCI FONKSİYONLAR
# ============================
def trimf_membership(x, params):
    a, b, c = params
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    else:
        return (c - x) / (c - b) if c != b else 1.0

def get_iqr_category(value, series: pd.Series):
    """Return 'Düşük', 'Orta', or 'Yüksek' based on IQR quartiles.
    Low  <= Q1, Medium between Q1 and Q3, High >= Q3.
    The quartiles are obtained from the training data and correspond to the
    thresholds shown in the box‑plot visualisations.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    if value <= q1:
        return "Düşük"
    elif value >= q3:
        return "Yüksek"
    else:
        return "Orta"

def get_firing_levels(sample, rules, features, mf_params, dataset_name):
    n_rules = len(rules)
    W = np.zeros(n_rules)
    synonyms = {"Short": "Low", "Long": "High"} if dataset_name == "desharnais" else {}
    for i, rule in enumerate(rules):
        levels = []
        for k, feat in enumerate(features):
            val     = sample[k]
            mf_name = synonyms.get(rule['antecedent'][k], rule['antecedent'][k])
            params  = mf_params['triangular'][feat][mf_name]
            levels.append(trimf_membership(val, params))
        W[i] = np.prod(levels)
    return W

def explain_prediction(sample, rules, coeffs, features, mf_params, dataset_name, scaler_y):
    W     = get_firing_levels(sample, rules, features, mf_params, dataset_name)
    W_sum = W.sum()
    if W_sum == 0:
        W_sum = 1e-10
    W_norm = W / W_sum
    
    # Baskin kural
    best_idx = np.argmax(W_norm)
    best_rule = rules[best_idx]
    
    # Tahmin
    Z = np.zeros(len(rules))
    for i in range(len(rules)):
        p1, p2, p3, c = coeffs[i]
        Z[i] = p1*sample[0] + p2*sample[1] + p3*sample[2] + c
        
    pred_scaled = np.sum(W_norm * Z)
    pred_real   = scaler_y.inverse_transform([[pred_scaled]])[0,0]
    
    return pred_real, best_idx+1, best_rule, W_norm[best_idx]

# ============================
# ANA YORUMLAMA DÖNGÜSÜ
# ============================
for dname, config in DATASETS.items():
    print("\n" + "=" * 75)
    print(f"VERI SETI: {dname.upper()} - MODEL ACIKLANABILIRLIK RAPORU")
    print("=" * 75)

    selected_features = config['selected_features']
    scenarios         = config['scenarios']

    with open(MODELS_DIR / f'{dname}_mf_parameters.json', 'r') as f:
        mf_params = json.load(f)
    with open(MODELS_DIR / f'{dname}_rule_sets.json', 'r') as f:
        rule_sets = json.load(f)
        
    scaler_y = joblib.load(MODELS_DIR / f"{dname}_scaler_y.pkl")
    scaler_X = joblib.load(MODELS_DIR / f"{dname}_scaler_X.pkl")
    train_df = pd.read_csv(MODELS_DIR / f"{dname}_train_clean.csv")

    # Sadece Model B (Gradient Descent) uzerinden yorumlama yapalim (genelde daha iyi calistigi icin)
    model_b_keys = [k for k in rule_sets if len(rule_sets[k]) > 0]
    
    coeff_summary = []
    
    for sn in model_b_keys:
        try:
            coeffs = np.load(MODELS_DIR / f"coefficients_{dname}_{sn}_ModelB.npy")
            # Katsayi ortalamalari
            mean_c = coeffs.mean(axis=0)
            coeff_summary.append({
                'Set': sn,
                'p1 (Ort)': mean_c[0],
                'p2 (Ort)': mean_c[1],
                'p3 (Ort)': mean_c[2],
                'c (Ort)' : mean_c[3]
            })
        except FileNotFoundError:
            print(f"   [!] Katsayi dosyasi bulunamadi: coefficients_{dname}_{sn}_ModelB.npy")

    print("\nKURAL SETLERININ ORTALAMA KATSAYILARI (MODEL B)")
    coeff_df = pd.DataFrame(coeff_summary)
    print(coeff_df.round(4).to_string(index=False))

    print("\nKATSAYILARIN MEALİ:")
    # Desharnais p3 takim tecrubesi (-), Albrecht p3 dosya sayisi (+)
    if dname == 'desharnais':
        print("   - p1 (PointsAjust): Proje boyutu arttikca efor pozitif etkilenir.")
        print("   - p2 (Length): Sure uzadikca efor pozitif etkilenir.")
        print("   - p3 (TeamExp): Takim tecrubesi arttikca efor azalmalidir (Negatif katsayi beklenir).")
    else:
        print("   - p1 (Input): Giris sayisi arttikca efor pozitif etkilenir (Pozitif katsayi beklenir).")
        print("   - p2 (Output): Cikis sayisi arttikca efor pozitif etkilenir (Pozitif katsayi beklenir).")
        print("   - p3 (File): Dosya sayisi arttikca efor pozitif etkilenir (Pozitif katsayi beklenir).")

    # Senaryo Analizleri
    print("\nSENARYO BAZLI TAHMIN ACIKLAMASI")
    scenario_plots_data = []

    for name, sample in scenarios.items():
        print(f"\n>>> Senaryo: {name} (Normalize: {sample.round(2)})")
        for sn in model_b_keys:
            try:
                coeffs = np.load(MODELS_DIR / f"coefficients_{dname}_{sn}_ModelB.npy")
                # Gerçek fiziksel değerleri ters ölçekleme ile al
                real_vals = scaler_X.inverse_transform(sample.reshape(1, -1)).flatten()
                # Her özelliğin gerçek değeri ve IQR bazlı kategorisi
                categories = []
                for idx, feat in enumerate(selected_features):
                    series = train_df[feat]
                    cat = get_iqr_category(real_vals[idx], series)
                    categories.append(cat)
                # Yazdırma
                print(f"   * Gerçek Değerler: {dict(zip(selected_features, real_vals.round(2)))}")
                print(f"   * Kategoriler (IQR): {dict(zip(selected_features, categories))}")
                
                pred, r_id, rule, weight = explain_prediction(sample, rule_sets[sn], coeffs, selected_features, mf_params, dname, scaler_y)
                print(f"   * {sn} ModelB Tahmini: {pred:.1f} adam-saat")
                print(f"     Baskin Kural   : R{r_id} ({'-'.join(rule['antecedent'])})")
                print(f"     Ateslenme Gucu : {weight:.2%}")
                    'Senaryo': name,
                    'Model': sn,
                    'Tahmin': pred,
                    'Baskin_Kural': f"R{r_id}",
                    'Agirlik': weight
                })
            except Exception as e:
                pass

    # ============================
    # GÖRSELLEŞTİRME
    # ============================
    if scenario_plots_data:
        plot_df = pd.DataFrame(scenario_plots_data)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Kucuk Proje Tahminleri
        ax = axes[0]
        kp_df = plot_df[plot_df['Senaryo'] == 'Kucuk_Proje']
        if not kp_df.empty:
            bars = ax.bar(kp_df['Model'], kp_df['Tahmin'], color='#3498DB', edgecolor='black')
            ax.set_ylabel('Efor Tahmini (Adam-Saat)')
            ax.set_title('Küçük Proje Senaryosu - Model Tahminleri')
            ax.grid(True, alpha=0.3, axis='y')
            for bar, rule, w in zip(bars, kp_df['Baskin_Kural'], kp_df['Agirlik']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (kp_df['Tahmin'].max()*0.02),
                        f"{rule} ({w:.0%})", ha='center', fontsize=9, fontweight='bold')

        # Buyuk Proje Tahminleri
        ax = axes[1]
        bp_df = plot_df[plot_df['Senaryo'] == 'Buyuk_Proje']
        if not bp_df.empty:
            bars = ax.bar(bp_df['Model'], bp_df['Tahmin'], color='#E74C3C', edgecolor='black')
            ax.set_ylabel('Efor Tahmini (Adam-Saat)')
            ax.set_title('Büyük Proje Senaryosu - Model Tahminleri')
            ax.grid(True, alpha=0.3, axis='y')
            for bar, rule, w in zip(bars, bp_df['Baskin_Kural'], bp_df['Agirlik']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (bp_df['Tahmin'].max()*0.02),
                        f"{rule} ({w:.0%})", ha='center', fontsize=9, fontweight='bold')

        plt.suptitle(f'Yorumlanabilir Senaryo Analizleri ({config["title"]} Seti)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{dname}_interpretation_analysis.png", dpi=100, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] {dname}_interpretation_analysis.png kaydedildi.")

# Geriye donuk uyumluluk dosyalari
shutil.copyfile(OUTPUT_DIR / "desharnais_interpretation_analysis.png", OUTPUT_DIR / "interpretation_analysis.png")
print("\n[OK] Geriye donuk uyumluluk dosyalari olusturuldu (interpretation_analysis.png)")
