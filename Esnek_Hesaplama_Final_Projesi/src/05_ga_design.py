# =============================================================================
# 05_ga_design.py
# Genetik Algoritma (GA) Tasarımı — Chromosome Encoding
# Klasik ve GA Tabanlı Sugeno FIS ile Yazılım Efor Tahmini
# =============================================================================
"""
Bu modül Genetik Algoritma'nın chromosome (birey) yapısını tanımlar.

Chromosome Yapısı:
  [ MF_parametreleri | Rule_selection_bitleri | Rule_katsayilari ]
  
  - MF parametreleri  : Her girdi için 3 MF × 2 parametre = 18 float (3 girdi)
  - Rule selection    : 20 kural için 0/1 bit  = 20 bit (int 0 veya 1)
  - Rule katsayıları  : Her kural için 4 katsayı (w0,w1,w2,w3) = 80 float

Toplam chromosome uzunluğu = 18 + 20 + 80 = 118 eleman
"""

import numpy as np
import json
import os
import sys

# Proje kökü
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ─── Sabitler ─────────────────────────────────────────────────────────────────
N_INPUTS      = 3          # PointsAjust, Length, TeamExp
N_MF_PER_VAR  = 3          # Low, Medium, High
MF_PARAMS     = 2          # [center, sigma] — Gaussian için
N_RULES       = 20         # Kural sayısı
COEF_PER_RULE = 4          # w0 + w1*x1 + w2*x2 + w3*x3

# Chromosome dilim indeksleri
MF_SLICE_END    = N_INPUTS * N_MF_PER_VAR * MF_PARAMS   # 18
RULE_SLICE_END  = MF_SLICE_END + N_RULES                  # 38
COEF_SLICE_END  = RULE_SLICE_END + N_RULES * COEF_PER_RULE  # 118

CHROMOSOME_LEN  = COEF_SLICE_END   # 118


# ─── Varsayılan MF parametreleri (Gaussian: center, sigma) ───────────────────
# Normalize edilmiş [0,1] uzayında
DEFAULT_MF_PARAMS = np.array([
    # PointsAjust — Low, Medium, High
    0.15, 0.10,
    0.50, 0.15,
    0.85, 0.10,
    # Length — Low, Medium, High
    0.20, 0.12,
    0.50, 0.15,
    0.80, 0.12,
    # TeamExp — Low, Medium, High
    0.25, 0.12,
    0.50, 0.15,
    0.75, 0.12,
], dtype=float)

# Varsayılan rule selection (tüm kurallar aktif)
DEFAULT_RULE_BITS = np.ones(N_RULES, dtype=float)

# Varsayılan katsayılar (sıfır başlangıç)
DEFAULT_COEFS = np.zeros(N_RULES * COEF_PER_RULE, dtype=float)


# ─── Chromosome Kodlama / Çözme Fonksiyonları ────────────────────────────────

def encode_chromosome(mf_params, rule_bits, rule_coefs):
    """
    MF parametreleri, rule bitleri ve katsayıları tek chromosome vektörüne kodlar.

    Parameters
    ----------
    mf_params  : ndarray (18,)  MF center/sigma değerleri [0,1]
    rule_bits  : ndarray (20,)  Aktif kural seçimi (0 veya 1)
    rule_coefs : ndarray (80,)  Sugeno katsayıları

    Returns
    -------
    chromosome : ndarray (118,)
    """
    return np.concatenate([mf_params, rule_bits.astype(float), rule_coefs])


def decode_chromosome(chromosome):
    """
    Chromosome vektörünü bileşenlerine ayrıştırır.

    Returns
    -------
    mf_params  : ndarray (18,)
    rule_bits  : ndarray (20,)  — 0.5 eşiği ile binary
    rule_coefs : ndarray (80,)
    """
    mf_params  = chromosome[:MF_SLICE_END]
    rule_bits  = (chromosome[MF_SLICE_END:RULE_SLICE_END] >= 0.5).astype(int)
    rule_coefs = chromosome[RULE_SLICE_END:COEF_SLICE_END]
    return mf_params, rule_bits, rule_coefs


def mf_params_to_dict(mf_params):
    """
    Düz MF parametre dizisini yapılandırılmış sözlüğe dönüştürür.

    Returns
    -------
    dict: {değişken_adı: [(center, sigma), ...]}
    """
    var_names = ["PointsAjust", "Length", "TeamExp"]
    labels    = ["Low", "Medium", "High"]
    result    = {}
    idx       = 0
    for var in var_names:
        result[var] = {}
        for lbl in labels:
            center = float(mf_params[idx])
            sigma  = float(max(mf_params[idx + 1], 0.01))  # sigma > 0
            result[var][lbl] = {"center": center, "sigma": sigma}
            idx += 2
    return result


def gaussian_mf(x, center, sigma):
    """Gaussian üyelik fonksiyonu: exp(-(x-c)^2 / (2*sigma^2))"""
    return np.exp(-0.5 * ((x - center) / max(sigma, 1e-6)) ** 2)


def fuzzify(x_normalized, mf_params):
    """
    Normalize girdi değerini fuzzify eder.

    Parameters
    ----------
    x_normalized : ndarray (N_INPUTS,)  — normalize edilmiş girdi
    mf_params    : ndarray (18,)        — chromosome'dan çıkan MF parametreleri

    Returns
    -------
    mu : ndarray (N_INPUTS, N_MF_PER_VAR)  — üyelik dereceleri
    """
    mf_dict = mf_params_to_dict(mf_params)
    var_names = ["PointsAjust", "Length", "TeamExp"]
    labels    = ["Low", "Medium", "High"]
    mu        = np.zeros((N_INPUTS, N_MF_PER_VAR))

    for i, var in enumerate(var_names):
        for j, lbl in enumerate(labels):
            c = mf_dict[var][lbl]["center"]
            s = mf_dict[var][lbl]["sigma"]
            mu[i, j] = gaussian_mf(x_normalized[i], c, s)

    return mu


# ─── Varsayılan Kural Seti ────────────────────────────────────────────────────

def load_default_rules():
    """
    models/rule_sets.json dosyasından 'Student' kural setini yükler.
    Dosya yoksa sabit bir kural seti döner.
    """
    rule_path = os.path.join(ROOT, "models", "rule_sets.json")
    if os.path.exists(rule_path):
        with open(rule_path, "r", encoding="utf-8") as f:
            all_rules = json.load(f)
        # İlk kural setini al (Student veya mevcut olan)
        first_key = list(all_rules.keys())[0]
        return all_rules[first_key]
    else:
        # Sabit fallback: 20 kural (antecedent encoding: var_idx × mf_idx çiftleri)
        rules = []
        combos = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
        for a in combos[:3]:
            for b in combos[:3]:
                if len(rules) >= 20:
                    break
                rules.append({"antecedents": [a, b]})
        while len(rules) < 20:
            rules.append({"antecedents": [(0,0),(1,0)]})
        return rules


# ─── Başlangıç Populasyonu ────────────────────────────────────────────────────

def create_initial_population(pop_size=50, seed=42):
    """
    Başlangıç populasyonunu oluşturur.

    İlk birey: varsayılan parametrelerle (iyi başlangıç noktası).
    Diğerleri: ±%20 Gaussian gürültü ile çeşitlendirilmiş.

    Parameters
    ----------
    pop_size : int   Populasyon büyüklüğü
    seed     : int   Tekrarlanabilirlik için seed

    Returns
    -------
    population : ndarray (pop_size, CHROMOSOME_LEN)
    """
    rng = np.random.default_rng(seed)
    population = np.zeros((pop_size, CHROMOSOME_LEN))

    # İlk birey: varsayılan
    default_chr = encode_chromosome(
        DEFAULT_MF_PARAMS.copy(),
        DEFAULT_RULE_BITS.copy(),
        DEFAULT_COEFS.copy()
    )
    population[0] = default_chr

    for i in range(1, pop_size):
        chr_ = default_chr.copy()

        # MF parametreleri: Gaussian gürültü + clip [0.01, 0.99]
        noise_mf = rng.normal(0, 0.08, size=MF_SLICE_END)
        chr_[:MF_SLICE_END] = np.clip(chr_[:MF_SLICE_END] + noise_mf, 0.01, 0.99)

        # Rule selection bitleri: %20 rassal flip
        for j in range(MF_SLICE_END, RULE_SLICE_END):
            if rng.random() < 0.2:
                chr_[j] = 1.0 - chr_[j]

        # Katsayılar: Gaussian gürültü
        noise_coef = rng.normal(0, 0.15, size=N_RULES * COEF_PER_RULE)
        chr_[RULE_SLICE_END:] = chr_[RULE_SLICE_END:] + noise_coef

        population[i] = chr_

    return population


# ─── Kısıt Düzeltme ──────────────────────────────────────────────────────────

def repair_chromosome(chromosome):
    """
    MF parametre sınırlarını ve minimum aktif kural sayısını zorlar.
    En az 5 kural aktif olmalı (çok kısıtlı model önlemi).
    """
    chrom = chromosome.copy()

    # MF parametreleri: [0.01, 0.99]
    chrom[:MF_SLICE_END] = np.clip(chrom[:MF_SLICE_END], 0.01, 0.99)

    # Rule bitleri: [0, 1]
    rule_bits = chrom[MF_SLICE_END:RULE_SLICE_END]
    rule_bits = np.clip(rule_bits, 0.0, 1.0)

    # Minimum 5 aktif kural garantisi
    binary_bits = (rule_bits >= 0.5).astype(float)
    if binary_bits.sum() < 5:
        inactive = np.where(binary_bits < 0.5)[0]
        activate = np.random.choice(inactive, size=5 - int(binary_bits.sum()), replace=False)
        binary_bits[activate] = 1.0
    chrom[MF_SLICE_END:RULE_SLICE_END] = binary_bits

    return chrom


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GA Chromosome Encoding Testi")
    print("=" * 60)

    # Populasyon oluştur
    pop = create_initial_population(pop_size=10)
    print(f"\nPopulasyon boyutu  : {pop.shape}")
    print(f"Chromosome uzunluğu: {CHROMOSOME_LEN}")

    # İlk bireyi çöz
    mf_p, rb, rc = decode_chromosome(pop[0])
    print(f"\nBirey 0:")
    print(f"  MF parametreleri  [{len(mf_p)} eleman]: {np.round(mf_p[:6], 3)} ...")
    print(f"  Aktif kural sayısı: {int(rb.sum())} / {N_RULES}")
    print(f"  Rule katsayıları  [{len(rc)} eleman]: {np.round(rc[:8], 3)} ...")

    # MF yapısı
    mf_dict = mf_params_to_dict(mf_p)
    print("\nMF Yapısı (Birey 0):")
    for var, mfs in mf_dict.items():
        parts = [f"{lbl}: c={v['center']:.2f} σ={v['sigma']:.2f}" for lbl, v in mfs.items()]
        print(f"  {var}: {' | '.join(parts)}")

    # Fuzzification testi
    x_test = np.array([0.3, 0.6, 0.5])
    mu = fuzzify(x_test, mf_p)
    print(f"\nFuzzification testi (x={x_test}):")
    labels = ["Low", "Medium", "High"]
    vars_  = ["PointsAjust", "Length", "TeamExp"]
    for i, var in enumerate(vars_):
        vals = {labels[j]: round(float(mu[i,j]), 3) for j in range(3)}
        print(f"  {var}: {vals}")

    # Kısıt testi
    repaired = repair_chromosome(pop[5])
    _, rb2, _ = decode_chromosome(repaired)
    print(f"\nKısıt düzeltme testi: Birey 5 aktif kural = {int(rb2.sum())}")

    print("\n✓ 05_ga_design.py testi başarılı.")
    print(f"  Chromosome uzunluğu: {CHROMOSOME_LEN} (MF:{MF_SLICE_END} + Rules:{N_RULES} + Coef:{N_RULES*COEF_PER_RULE})")
