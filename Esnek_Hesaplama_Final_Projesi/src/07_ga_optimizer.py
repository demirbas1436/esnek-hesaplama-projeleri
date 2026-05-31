# =============================================================================
# 07_ga_optimizer.py
# Ana GA Döngüsü — RMSE Fitness Fonksiyonu ile Sugeno Optimizasyonu
# Klasik ve GA Tabanlı Sugeno FIS ile Yazılım Efor Tahmini
# =============================================================================
"""
Bu modül Genetik Algoritma'nın ana optimizasyon döngüsünü çalıştırır.

Fitness Fonksiyonu:
    fitness = -RMSE(y_pred, y_true)
    → RMSE minimize = fitness maximize

GA Parametreleri:
    Populasyon : 50 birey
    Nesil      : 100
    Crossover  : 0.80
    Mutation   : 0.05
    Seçim      : Tournament (size=3)
    Elitizm    : 2 birey

Çıktılar:
    models/ga_best_chromosome.npy     — En iyi chromosome
    models/ga_convergence.npy         — Nesil başı en iyi RMSE
    models/ga_stats.json              — Özet istatistikler
    output/ga_convergence.png         — Konverjans grafiği
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import json
import os
import sys
import time

# Proje kökü
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ga_design   import (
    create_initial_population, decode_chromosome, sugeno_predict,
    CHROMOSOME_LEN, N_RULES, encode_chromosome,
    DEFAULT_MF_PARAMS, DEFAULT_RULE_BITS, DEFAULT_COEFS
)
from ga_operators import create_next_generation

# ─── Yollar ───────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(ROOT, "models")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Fitness Fonksiyonu ───────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def fitness_function(chromosome, X_train, y_train, X_val=None, y_val=None, lambda_reg=0.001):
    """
    RMSE tabanli fitness + L2 regularizasyon.
    Eger validation seti verilmisse, fitness validation RMSE'ye gore hesaplanir
    (overfitting onlemi).
    """
    from ga_design import RULE_SLICE_END, COEF_SLICE_END
    y_pred = sugeno_predict(X_train, chromosome)
    err_train = rmse(y_train, y_pred)

    # Validation varsa kullan
    if X_val is not None and y_val is not None:
        y_pred_val = sugeno_predict(X_val, chromosome)
        err_val = rmse(y_val, y_pred_val)
        err = 0.5 * err_train + 0.5 * err_val   # Train+Val ortala
    else:
        err = err_train

    # L2 regularizasyon (katsayilarin buyumesini engelle)
    coefs = chromosome[RULE_SLICE_END:COEF_SLICE_END]
    reg   = lambda_reg * np.sqrt(np.mean(coefs ** 2))

    return -(err + reg)

def evaluate_population(population, X_train, y_train, X_val=None, y_val=None):
    """Tum populasyonu degerlendir -> fitness array"""
    return np.array([
        fitness_function(chrom, X_train, y_train, X_val, y_val)
        for chrom in population
    ])


# ─── Katsayı Başlatma (Least Squares ile) ────────────────────────────────────

def initialize_coefficients_ls(population, X_train, y_train, n_init=5):
    """
    Sadece ilk n_init birey icin katsayilari Least Squares ile baslat.
    Geri kalanlari rasgele kalir -> cesitlilik korunur.
    Ridge (L2) regularizasyon eklenir -> overfitting onlenir.
    """
    from ga_design import (
        decode_chromosome, fuzzify_sample, RULE_ANTECEDENTS,
        COEF_PER_RULE, RULE_SLICE_END, MF_SLICE_END, N_RULES
    )

    initialized = population.copy()

    for idx in range(min(n_init, len(population))):
        chrom = population[idx].copy()
        mf_params, rule_bits, _ = decode_chromosome(chrom)

        active_rules = [r for r in range(N_RULES) if rule_bits[r] == 1]
        if len(active_rules) == 0:
            continue

        n_samples = len(X_train)
        n_active  = len(active_rules)
        A = np.zeros((n_samples, n_active * COEF_PER_RULE))
        w_sum = np.zeros(n_samples)

        for col_i, r_idx in enumerate(active_rules):
            a0, a1, a2 = RULE_ANTECEDENTS[r_idx]
            for s in range(n_samples):
                mu = fuzzify_sample(X_train[s], mf_params)
                strength = mu[0, a0] * mu[1, a1] * mu[2, a2]
                base = col_i * COEF_PER_RULE
                A[s, base]   = strength
                A[s, base+1] = strength * X_train[s, 0]
                A[s, base+2] = strength * X_train[s, 1]
                A[s, base+3] = strength * X_train[s, 2]
                w_sum[s]    += strength

        w_sum_safe = np.where(w_sum > 1e-9, w_sum, 1.0)
        A_norm = A / w_sum_safe[:, None]

        # Ridge (L2) regularizasyonlu OLS
        try:
            n_cols = A_norm.shape[1]
            alpha  = 0.1   # regularizasyon gucu
            ATA    = A_norm.T @ A_norm + alpha * np.eye(n_cols)
            ATy    = A_norm.T @ y_train
            coefs_flat = np.linalg.solve(ATA, ATy)
        except Exception:
            continue

        new_coefs = np.zeros(N_RULES * COEF_PER_RULE)
        for col_i, r_idx in enumerate(active_rules):
            new_coefs[r_idx * COEF_PER_RULE:(r_idx+1) * COEF_PER_RULE] = \
                coefs_flat[col_i * COEF_PER_RULE:(col_i+1) * COEF_PER_RULE]

        initialized[idx, RULE_SLICE_END:] = new_coefs

    return initialized


# ─── Ana GA Döngüsü ───────────────────────────────────────────────────────────

def run_ga(X_train, y_train, X_test, y_test,
           pop_size=50, n_generations=100,
           crossover_rate=0.80, mutation_rate=0.05,
           elite_count=2, seed=42, verbose=True):
    """
    Genetik Algoritma optimizasyon döngüsü.

    Returns
    -------
    best_chromosome : ndarray (118,)
    history         : dict   {best_fitness, avg_fitness, best_rmse_train, best_rmse_test}
    all_stats       : list   [{nesil istatistikleri}]
    """
    rng = np.random.default_rng(seed)

    # ── Başlangıç populasyonu ─────────────────────────────────────────────────
    if verbose:
        print("Baslangic populasyonu olusturuluyor...")
    population = create_initial_population(pop_size=pop_size, seed=seed)
    population = initialize_coefficients_ls(population, X_train, y_train, n_init=5)

    # Train'den validation ayir (overfitting onlemi)
    n_val = max(1, len(X_train) // 5)
    X_val, y_val = X_train[:n_val], y_train[:n_val]

    # Tarihçe
    history = {
        "best_fitness"   : [],
        "avg_fitness"    : [],
        "best_rmse_train": [],
        "best_rmse_test" : [],
        "active_rules_avg": [],
    }
    all_stats = []
    best_chromosome = None
    best_fitness_ever = -np.inf
    t0 = time.time()

    for gen in range(n_generations):
        # ── Fitness hesapla ───────────────────────────────────────────────────
        fitness_scores = evaluate_population(population, X_train, y_train, X_val, y_val)

        # ── İstatistikler ─────────────────────────────────────────────────────
        best_idx    = np.argmax(fitness_scores)
        best_fit    = fitness_scores[best_idx]
        avg_fit     = fitness_scores.mean()

        # Test RMSE (en iyi bireyle)
        best_pred_tr = sugeno_predict(X_train, population[best_idx])
        best_pred_te = sugeno_predict(X_test,  population[best_idx])
        rmse_train   = rmse(y_train, best_pred_tr)
        rmse_test    = rmse(y_test,  best_pred_te)

        # Aktif kural sayısı ortalaması
        active_list = []
        for chrom in population:
            _, rb, _ = decode_chromosome(chrom)
            active_list.append(rb.sum())
        avg_active = np.mean(active_list)

        history["best_fitness"].append(float(best_fit))
        history["avg_fitness"].append(float(avg_fit))
        history["best_rmse_train"].append(float(rmse_train))
        history["best_rmse_test"].append(float(rmse_test))
        history["active_rules_avg"].append(float(avg_active))

        # En iyi chromosome güncelle
        if best_fit > best_fitness_ever:
            best_fitness_ever = best_fit
            best_chromosome   = population[best_idx].copy()

        # Nesil istatistiği
        gen_stat = {
            "generation"    : gen + 1,
            "best_fitness"  : round(float(best_fit), 6),
            "avg_fitness"   : round(float(avg_fit), 6),
            "rmse_train"    : round(float(rmse_train), 4),
            "rmse_test"     : round(float(rmse_test), 4),
            "avg_active_rules": round(float(avg_active), 2),
        }
        all_stats.append(gen_stat)

        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            elapsed = time.time() - t0
            print(f"  Nesil {gen+1:3d}/{n_generations} | "
                  f"RMSE_train={rmse_train:.4f} | "
                  f"RMSE_test={rmse_test:.4f} | "
                  f"Aktif kural={avg_active:.1f} | "
                  f"Sure={elapsed:.1f}s")

        # ── Yeni nesil ────────────────────────────────────────────────────────
        if gen < n_generations - 1:
            population, _ = create_next_generation(
                population, fitness_scores,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                selection_method="tournament",
                crossover_method="single_point",
                elite_count=elite_count,
                rng=rng
            )

    total_time = time.time() - t0
    if verbose:
        print(f"\n[OK] GA tamamlandi. Toplam sure: {total_time:.1f}s")
        print(f"  En iyi RMSE_train : {min(history['best_rmse_train']):.4f}")
        print(f"  En iyi RMSE_test  : {min(history['best_rmse_test']):.4f}")

    return best_chromosome, history, all_stats


# ─── Konverjans Grafiği ───────────────────────────────────────────────────────

def plot_convergence(history, output_dir):
    """GA konverjans grafiği — RMSE vs Nesil"""
    generations = list(range(1, len(history["best_rmse_train"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GA Optimizasyonu — Konverjans Analizi", fontsize=14, fontweight="bold")

    # ── Sol: RMSE eğrisi ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(generations, history["best_rmse_train"], "b-", lw=2, label="Eğitim RMSE")
    ax.plot(generations, history["best_rmse_test"],  "r--", lw=2, label="Test RMSE")
    ax.set_xlabel("Nesil", fontsize=11)
    ax.set_ylabel("RMSE (normalize)", fontsize=11)
    ax.set_title("RMSE Konverjans Eğrisi", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    # Minimum noktaları işaretle
    best_gen_tr = int(np.argmin(history["best_rmse_train"])) + 1
    best_gen_te = int(np.argmin(history["best_rmse_test"])) + 1
    ax.axvline(best_gen_tr, color="blue",  alpha=0.3, linestyle=":")
    ax.axvline(best_gen_te, color="red",   alpha=0.3, linestyle=":")

    # ── Sağ: Aktif kural sayısı ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(generations, history["active_rules_avg"], "g-", lw=2, label="Ortalama Aktif Kural")
    ax2.set_xlabel("Nesil", fontsize=11)
    ax2.set_ylabel("Aktif Kural Sayısı", fontsize=11)
    ax2.set_title("GA Tarafından Seçilen Kural Sayısı", fontsize=12)
    ax2.set_ylim(0, 21)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "ga_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Konverjans grafiği kaydedildi: {path}")
    return path


# ─── Ana Akış ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Asama 7: GA Optimizer - Sugeno FIS Optimizasyonu")
    print("=" * 60)

    # ── Veri yükle ────────────────────────────────────────────────────────────
    train_path = os.path.join(MODELS_DIR, "desharnais_train_clean.csv")
    test_path  = os.path.join(MODELS_DIR, "desharnais_test_clean.csv")

    if not os.path.exists(train_path):
        print("HATA: Eğitim verisi bulunamadı. Önce 01_eda_desharnais.py çalıştırın.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    feature_cols = ["PointsAjust_norm", "Length_norm", "TeamExp_norm"]
    target_col   = "Effort_norm"

    # Sütun adlarını kontrol et
    for col in feature_cols + [target_col]:
        if col not in train_df.columns:
            # _norm eki olmayabilir — ham normalize kolonlar dene
            alt = col.replace("_norm", "")
            if alt in train_df.columns:
                feature_cols = [c.replace("_norm","") for c in feature_cols]
                target_col   = target_col.replace("_norm","")
            break

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df[target_col].values.astype(float)
    X_test  = test_df[feature_cols].values.astype(float)
    y_test  = test_df[target_col].values.astype(float)

    print(f"\nVeri yuklendi:")
    print(f"  Egitim : {X_train.shape[0]} ornek")
    print(f"  Test   : {X_test.shape[0]} ornek")
    print(f"  Girdi  : {feature_cols}")
    print(f"  Hedef  : {target_col}")

    # ── GA çalıştır ───────────────────────────────────────────────────────────
    print("\nGA baslatiliyor (50 birey x 100 nesil)...")
    best_chrom, history, all_stats = run_ga(
        X_train, y_train, X_test, y_test,
        pop_size=50, n_generations=100,
        crossover_rate=0.80, mutation_rate=0.05,
        elite_count=2, seed=42, verbose=True
    )

    # ── Sonuçları kaydet ──────────────────────────────────────────────────────
    np.save(os.path.join(MODELS_DIR, "ga_best_chromosome.npy"), best_chrom)
    np.save(os.path.join(MODELS_DIR, "ga_convergence.npy"),
            np.array(history["best_rmse_test"]))

    ga_stats = {
        "best_rmse_train"  : round(min(history["best_rmse_train"]), 4),
        "best_rmse_test"   : round(min(history["best_rmse_test"]), 4),
        "final_rmse_train" : round(history["best_rmse_train"][-1], 4),
        "final_rmse_test"  : round(history["best_rmse_test"][-1], 4),
        "generations"      : 100,
        "pop_size"         : 50,
        "crossover_rate"   : 0.80,
        "mutation_rate"    : 0.05,
    }
    _, rb_best, _ = decode_chromosome(best_chrom)
    ga_stats["active_rules_best"] = int(rb_best.sum())

    with open(os.path.join(MODELS_DIR, "ga_stats.json"), "w", encoding="utf-8") as f:
        json.dump(ga_stats, f, indent=2, ensure_ascii=False)

    # Konverjans grafiği
    plot_convergence(history, OUTPUT_DIR)

    # ── Özet ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GA Sonuçları")
    print("=" * 60)
    print(f"  En iyi RMSE_train  : {ga_stats['best_rmse_train']}")
    print(f"  En iyi RMSE_test   : {ga_stats['best_rmse_test']}")
    print(f"  Aktif kural (best) : {ga_stats['active_rules_best']} / 20")
    print(f"\n  Dosyalar:")
    print(f"    models/ga_best_chromosome.npy")
    print(f"    models/ga_convergence.npy")
    print(f"    models/ga_stats.json")
    print(f"    output/ga_convergence.png")
    print("\n[OK] Asama 7 tamamlandi.")


if __name__ == "__main__":
    main()
