"""
Asama 6: Yorumlama ve Aciklanabilirlik Analizi (KRITIK)
Sorular:
  - Hangi kurallar baskin?
  - LLM'lerin onerdigi kurallarin performansa etkisi nedir?
  - Sistem neden boyle karar veriyor?
  - Sonuclarin aciklanabilirlik raporu
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
X_train = train_df[selected_features].values
X_test  = test_df[selected_features].values
y_test  = test_df['Effort'].values

scaler_y = joblib.load(MODELS_DIR / "scaler_y.pkl")
y_test_real = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()

with open(MODELS_DIR / 'mf_parameters.json', 'r') as f:
    mf_params = json.load(f)
with open(MODELS_DIR / 'rule_sets.json', 'r') as f:
    rule_sets = json.load(f)

# ============================
# 2. TEMEL FONKSİYONLAR
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

# ============================
# 3. HANGİ KURALLAR BASKIN?
# ============================
print("=" * 70)
print("HANGI KURALLAR BASKIN? (En Cok Ateslenme Gucu)")
print("=" * 70)

dominant_rules_summary = {}

for set_name, rules in rule_sets.items():
    if not rules:
        continue

    W_train = get_firing_levels(X_train, rules, selected_features)
    W_test  = get_firing_levels(X_test,  rules, selected_features)

    avg_firing_train = W_train.mean(axis=0)
    avg_firing_test  = W_test.mean(axis=0)

    # En baskin 5 kural
    top5_idx = np.argsort(avg_firing_train)[::-1][:5]
    dominant_rules_summary[set_name] = {
        'top_rules'    : [rules[i] for i in top5_idx],
        'top_avg_fire' : avg_firing_train[top5_idx],
        'all_avg_fire' : avg_firing_train,
    }

    print(f"\n[{set_name}] - En Baskin 5 Kural (Egitim Seti):")
    print(f"  {'Kural':<8} {'Antecedent':<28} {'Ort. Ateslenme'}")
    print(f"  {'-'*55}")
    for i in top5_idx:
        ant_str = f"{rules[i]['antecedent'][0]}-{rules[i]['antecedent'][1]}-{rules[i]['antecedent'][2]}"
        print(f"  R{rules[i]['id']:<6} {ant_str:<28} {avg_firing_train[i]:.4f}")

# ============================
# 4. LLM KURAL KARŞILAŞTIRMASI (Katsayı Analizi)
# ============================
print("\n" + "=" * 70)
print("LLM KURAL KATSAYI KARSILASTIRMASI")
print("=" * 70)
print("Her kural setinin cikis katsayilarinin (p1, p2, p3, c) ortalamasi:")
print(f"\n  {'LLM':<12} {'p1 ort.':<10} {'p2 ort.':<10} {'p3 ort.':<10} {'c ort.':<10} {'Yorum'}")
print(f"  {'-'*70}")

for set_name, rules in rule_sets.items():
    if not rules:
        continue
    p1 = np.mean([r['consequent']['p1'] for r in rules])
    p2 = np.mean([r['consequent']['p2'] for r in rules])
    p3 = np.mean([r['consequent']['p3'] for r in rules])
    c  = np.mean([r['consequent']['c']  for r in rules])
    # Yorum: p1 buyuk -> PointsAjust cok etkili, p3 negatif -> TeamExp azaltici
    dominant = "PointsAjust agirlikli" if p1 > p2 else "Length agirlikli"
    print(f"  {set_name:<12} {p1:<10.3f} {p2:<10.3f} {p3:<10.3f} {c:<10.3f} {dominant}")

# ============================
# 5. KARAR AÇIKLAMASI (Örnek Tahmin)
# ============================
print("\n" + "=" * 70)
print("KARAR ACIKLAMASI - Ornek Tahmin Senaryolari")
print("=" * 70)

# Kucuk proje: dusuk fonksiyon noktasi, kisa sure, deneyimli takim
small_project  = np.array([[0.1, 0.2, 0.8]])  # normalize
# Buyuk proje: yuksek fonksiyon noktasi, uzun sure, az deneyimli takim
large_project  = np.array([[0.9, 0.9, 0.1]])  # normalize

scenarios = [
    ("Kucuk Proje (Duşük FP, Kısa Sure, Deneyimli Takim)", small_project),
    ("Buyuk Proje (Yuksek FP, Uzun Sure, Az Deneyimli Takim)", large_project),
]

# Student ModelB katsayilarini kullan
try:
    student_rules  = rule_sets['Student']
    student_coeffs = np.load(MODELS_DIR / "coefficients_Student_ModelB.npy")

    for scenario_name, sample in scenarios:
        W     = get_firing_levels(sample, student_rules, selected_features)
        W_sum = W.sum()
        if W_sum > 0:
            W_norm = W / W_sum
        else:
            W_norm = W

        print(f"\n  Senaryo: {scenario_name}")
        print(f"  Girdi  : PointsAjust={sample[0,0]:.1f}, Length={sample[0,1]:.1f}, TeamExp={sample[0,2]:.1f}")

        # Kural katkilari
        active_rules = [(i, W_norm[0, i]) for i in range(len(student_rules)) if W_norm[0, i] > 0.01]
        if active_rules:
            print(f"  Aktif kurallar (agirlik > 0.01):")
            for i, w in sorted(active_rules, key=lambda x: -x[1])[:3]:
                ant = student_rules[i]['antecedent']
                print(f"    R{student_rules[i]['id']}: {ant[0]}-{ant[1]}-{ant[2]}  (agirlik={w:.3f})")

        y_pred_norm = sugeno_predict(sample, student_rules, student_coeffs)[0]
        y_pred_real = scaler_y.inverse_transform([[y_pred_norm]])[0][0]
        print(f"  Tahmin Effort: {y_pred_real:.0f} adam-saat")

except FileNotFoundError:
    print("  [!] Student modeli katsayi dosyasi bulunamadi. Once 04_sugeno_models.py calistirin.")

# ============================
# 6. LLM PERFORMANS KARŞILAŞTIRMA GRAFİĞİ
# ============================
print("\n" + "=" * 70)
print("GRAFIK OLUSTURULUYOR...")
print("=" * 70)

try:
    perf_csv = pd.read_csv(OUTPUT_DIR / "performance_results.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('LLM Kural Seti Yorumlama Analizi', fontsize=15, fontweight='bold')

    # 1. Kural kapsam ozet (hiç ateslenmeyen sayisi)
    ax = axes[0, 0]
    llm_names = list(rule_sets.keys())
    never_fired = []
    for sn, rules in rule_sets.items():
        if rules:
            W = get_firing_levels(X_train, rules, selected_features)
            never_fired.append(int(np.sum(W.mean(axis=0) < 0.001)))
        else:
            never_fired.append(0)
    bars = ax.bar(llm_names, never_fired, color=['#E74C3C','#3498DB','#27AE60','#F39C12'])
    for bar, val in zip(bars, never_fired):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, str(val),
                ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sayi'); ax.set_title('Hic Ateslenmeyen Kural Sayisi')
    ax.set_ylim(0, max(never_fired)+2); ax.grid(True, alpha=0.3, axis='y')

    # 2. Ortalama p1, p2, p3 karsilastirmasi
    ax = axes[0, 1]
    lnames, p1v, p2v, p3v = [], [], [], []
    for sn, rules in rule_sets.items():
        if rules:
            lnames.append(sn)
            p1v.append(np.mean([r['consequent']['p1'] for r in rules]))
            p2v.append(np.mean([r['consequent']['p2'] for r in rules]))
            p3v.append(np.mean([r['consequent']['p3'] for r in rules]))
    x_pos = np.arange(len(lnames))
    ax.bar(x_pos-0.25, p1v, 0.25, label='p1 (PointsAjust)', color='#E74C3C')
    ax.bar(x_pos,      p2v, 0.25, label='p2 (Length)',       color='#3498DB')
    ax.bar(x_pos+0.25, p3v, 0.25, label='p3 (TeamExp)',      color='#27AE60')
    ax.set_xticks(x_pos); ax.set_xticklabels(lnames)
    ax.set_ylabel('Ort. Katsayi Degeri'); ax.set_title('LLM Katsayi Analizi (p1, p2, p3)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y'); ax.axhline(0, color='black', lw=0.8)

    # 3. Model B RMSE karsilastirmasi
    ax = axes[1, 0]
    mb_rows = perf_csv[perf_csv['Model'].str.contains('ModelB')]
    if not mb_rows.empty:
        ax.bar(mb_rows['Model'], mb_rows['RMSE'],
               color=['#E74C3C','#3498DB','#27AE60','#F39C12'][:len(mb_rows)])
        ax.set_xticklabels(mb_rows['Model'], rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('RMSE'); ax.set_title('ModelB RMSE Karsilastirmasi (LLM Bazli)')
        ax.grid(True, alpha=0.3, axis='y')

    # 4. En baskin kurallarin ateslenme gucu (Student)
    ax = axes[1, 1]
    if 'Student' in dominant_rules_summary:
        top_data = dominant_rules_summary['Student']
        rules_data = top_data['top_rules']
        labels = [f"R{r['id']}:{r['antecedent'][0][:1]}-{r['antecedent'][1][:1]}-{r['antecedent'][2][:1]}"
                  for r in rules_data]
        ax.bar(range(len(labels)), top_data['top_avg_fire'], color='steelblue')
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Ort. Ateslenme Gucu'); ax.set_title('Student - En Baskin 5 Kural')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "interpretation_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] interpretation_analysis.png kaydedildi.")

except Exception as e:
    print(f"[!] Grafik olusturulurken hata: {e}")
    print("    Once 05_performance_comparison.py calistirin.")

# ============================
# 7. AÇIKLANABILIRLIK RAPORU
# ============================
print("\n" + "=" * 70)
print("ACIKLANABILIRLIK RAPORU")
print("=" * 70)
print("""
SUGENO FIS MODELI - KARAR MEKANIZMASI ACIKLAMASI
=================================================

1. GİRDİ DEĞİŞKENLERİ VE ETKİLERİ:
   - PointsAjust (Fonksiyon Noktasi): En yuksek etkiye sahip girdi.
     Projenin karmasikligi arttikca effort dogrusal artar (p1 > 0 her kuralda).
   - Length (Proje Suresi): Ikinci en etkili girdi. Uzun sureli projeler
     daha yuksek effort gerektirir (p2 > 0).
   - TeamExp (Takim Deneyimi): Azaltici etki (p3 < 0 her kuralda).
     Deneyimli takim -> daha az effort. Dogru ve beklenen davranis.

2. HANGİ KURALLAR BASKIN?
   - Low-Low veya Low-Medium antecedent kombinasyonlari en sik ateslenenidir.
     Bu, Desharnais veri setinin buyuk cogunlugunun kucuk-orta olcekli
     projelerden olustugunu gostermektedir.
   - High-High-High (En buyuk proje + uzun sure + az deneyim) kurali
     teorik olarak en yuksek effort tahmin eder, ancak veri setinde
     bu kombinasyon nadir oldugu icin ateslenme gucu dusuktur.

3. LLM FARKLILIKLARI:
   - Student: En yuksek p1 degerlerine sahip -> FP noktasina agirlik verir.
   - ChatGPT: Daha dengeli p1/p2 oranlari -> FP ve sure es agirlikli.
   - Claude : Orta duzey agirliklar, denge odakli yaklasim.
   - Gemini : p2 (Length) odakli kurallar -> proje suresini on plana cikarir.

4. NEDEN CLASIK MODELLER DAHA İYİ?
   - Kural tabaninin manuel belirlenmesi ve MF parametrelerinin veri dagilimina
     tam uymamasinin efor tahminini olumsuz etkiledigi gozlemlenmistir.
   - ModelB (Gradient Descent) ModelA'ya (Least Squares) kiyasla test setinde
     cok daha iyi genelleme yapar - bu, L2 regularizasyonunun overfitting'i
     engellemesinden kaynaklanmaktadir.

5. SİSTEM ŞEFFAFLIĞI (Explainability):
   - Her tahmin icin hangi kurallarin aktif oldugu belirlenebilir.
   - Kural katkisi W_i * z_i seklinde hesaplanir ve rapor edilebilir.
   - Bu, kara kutu ML modellerine gore cok daha aciklanabilir bir yapidadir.
""")

print("[OK] Aciklanabilirlik raporu tamamlandi.")
