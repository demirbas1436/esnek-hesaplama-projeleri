# =============================================================================
# 08_karsilastirma.py
# Klasik Sugeno vs GA-Sugeno — Karşılaştırmalı Performans Analizi
# Klasik ve GA Tabanlı Sugeno FIS ile Yazılım Efor Tahmini
# =============================================================================
"""
Bu modül Klasik Sugeno ve GA-optimize Sugeno modellerini karşılaştırır.

Metrikler:
    RMSE (Root Mean Squared Error)
    MAE  (Mean Absolute Error)
    MAPE (Mean Absolute Percentage Error)
    R²   (Coefficient of Determination)

Çıktılar:
    output/karsilastirma_metrikler.png  — Bar chart karşılaştırma
    output/karsilastirma_scatter.png    — Tahmin vs gerçek scatter
    output/karsilastirma_ga_vs_klasik.png — Birleşik rapor figürü
    models/karsilastirma_sonuclari.json — Sayısal sonuçlar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use("Agg")
import json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ga_design    import sugeno_predict, decode_chromosome, create_initial_population
from ga_design    import encode_chromosome, DEFAULT_MF_PARAMS, DEFAULT_RULE_BITS

MODELS_DIR = os.path.join(ROOT, "models")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Metrik Hesaplama ─────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, label="Model"):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mask = y_true > 1e-6
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
    return {
        "model": label,
        "RMSE" : round(rmse, 4),
        "MAE"  : round(mae,  4),
        "MAPE" : round(mape, 2),
        "R2"   : round(r2,   4),
    }


# ─── Klasik Sugeno Modeli ─────────────────────────────────────────────────────

def load_classic_sugeno(X_train, y_train):
    """
    Klasik Sugeno: Varsayilan Gaussian MF parametreleri + Ridge LS katsayilari.
    Her zaman kendi Gaussian MF sistemimizle Ridge LS hesaplar (tutarlilik icin).
    """
    from ga_design import (
        RULE_ANTECEDENTS, COEF_PER_RULE, N_RULES,
        fuzzify_sample, RULE_SLICE_END, MF_SLICE_END
    )

    print("  Klasik katsayilar hesaplaniyor (Ridge LS + Gaussian MF)...")
    mf_params = DEFAULT_MF_PARAMS.copy()
    rule_bits = DEFAULT_RULE_BITS.copy()

    active = [r for r in range(N_RULES) if int(rule_bits[r]) == 1]
    n_s    = len(X_train)
    A      = np.zeros((n_s, len(active) * COEF_PER_RULE))
    w_sum  = np.zeros(n_s)

    for col_i, r_idx in enumerate(active):
        a0, a1, a2 = RULE_ANTECEDENTS[r_idx]
        for s in range(n_s):
            mu = fuzzify_sample(X_train[s], mf_params)
            st = mu[0, a0] * mu[1, a1] * mu[2, a2]
            base = col_i * COEF_PER_RULE
            A[s, base:base+4] = [st, st*X_train[s,0], st*X_train[s,1], st*X_train[s,2]]
            w_sum[s] += st

    w_safe  = np.where(w_sum > 1e-9, w_sum, 1.0)
    A_norm  = A / w_safe[:, None]

    # Ridge regularizasyon
    n_cols = A_norm.shape[1]
    alpha  = 0.1
    ATA    = A_norm.T @ A_norm + alpha * np.eye(n_cols)
    ATy    = A_norm.T @ y_train
    coefs_f = np.linalg.solve(ATA, ATy)

    full_coefs = np.zeros(N_RULES * COEF_PER_RULE)
    for col_i, r_idx in enumerate(active):
        full_coefs[r_idx*COEF_PER_RULE:(r_idx+1)*COEF_PER_RULE] = \
            coefs_f[col_i*COEF_PER_RULE:(col_i+1)*COEF_PER_RULE]

    return encode_chromosome(mf_params, rule_bits, full_coefs)


# ─── Karşılaştırma Grafikleri ─────────────────────────────────────────────────

def plot_metric_comparison(results, output_dir):
    """Bar chart — her metrik için Klasik vs GA karşılaştırması."""
    metrics  = ["RMSE", "MAE", "MAPE", "R2"]
    models   = [r["model"] for r in results]
    colors   = ["#4C72B0", "#DD8452"]
    x        = np.arange(len(metrics))
    width    = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Klasik Sugeno vs GA-Sugeno — Performans Karşılaştırması",
                 fontsize=13, fontweight="bold")

    # ── Sol: RMSE, MAE, MAPE ──────────────────────────────────────────────────
    ax = axes[0]
    for i, (res, color) in enumerate(zip(results, colors)):
        vals = [res["RMSE"], res["MAE"], res["MAPE"]]
        bars = ax.bar(np.arange(3) + i*width, vals, width, label=res["model"],
                      color=color, alpha=0.85, edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(np.arange(3) + width/2)
    ax.set_xticklabels(["RMSE", "MAE", "MAPE (%)"], fontsize=11)
    ax.set_ylabel("Hata Değeri (normalize)", fontsize=11)
    ax.set_title("Hata Metrikleri (düşük = iyi)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # ── Sağ: R² karşılaştırması ───────────────────────────────────────────────
    ax2 = axes[1]
    r2_vals = [r["R2"] for r in results]
    bars2   = ax2.bar(models, r2_vals, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars2, r2_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("R² Skoru", fontsize=11)
    ax2.set_title("R² Katsayısı (yüksek = iyi)", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, color="green", alpha=0.3, linestyle="--", label="Mükemmel=1.0")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "karsilastirma_metrikler.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Metrik karşılaştırma grafiği: {path}")
    return path


def plot_scatter_comparison(results_raw, y_test, output_dir):
    """Tahmin vs Gerçek scatter — her model için ayrı."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Tahmin vs Gerçek Değer Karşılaştırması",
                 fontsize=13, fontweight="bold")

    colors = ["#4C72B0", "#DD8452"]
    for ax, (label, y_pred), color in zip(axes, results_raw, colors):
        ax.scatter(y_test, y_pred, alpha=0.65, s=40, color=color,
                   edgecolors="white", linewidths=0.5)
        # Mükemmel tahmin çizgisi
        lo = min(y_test.min(), y_pred.min()) * 0.95
        hi = max(y_test.max(), y_pred.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Mükemmel tahmin")
        # RMSE annotasyon
        err = np.sqrt(np.mean((y_test - y_pred)**2))
        ax.text(0.05, 0.93, f"RMSE={err:.4f}", transform=ax.transAxes,
                fontsize=10, color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
        ax.set_xlabel("Gerçek Effort (normalize)", fontsize=11)
        ax.set_ylabel("Tahmin Effort (normalize)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "karsilastirma_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scatter karşılaştırma grafiği: {path}")
    return path


def plot_combined_report(results, history_test_rmse, output_dir):
    """Konverjans + Metrikler + Scatter — tek figürde."""
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle("Klasik Sugeno vs GA-Sugeno — Tam Karşılaştırma Raporu",
                 fontsize=14, fontweight="bold")

    colors = ["#4C72B0", "#DD8452"]

    # ── (0,0): Konverjans ─────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    if history_test_rmse is not None:
        ax0.plot(range(1, len(history_test_rmse)+1), history_test_rmse,
                 "r-", lw=2, label="GA Test RMSE")
        ax0.axhline(results[0]["RMSE"], color="blue", linestyle="--", lw=1.5,
                    label=f"Klasik RMSE={results[0]['RMSE']:.4f}")
    ax0.set_xlabel("Nesil", fontsize=10)
    ax0.set_ylabel("Test RMSE", fontsize=10)
    ax0.set_title("GA Konverjans", fontsize=11)
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)

    # ── (0,1)-(0,2): RMSE & R² bar ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    bar_w = 0.35
    for i, (res, col) in enumerate(zip(results, colors)):
        b = ax1.bar(i, res["RMSE"], bar_w, label=res["model"], color=col, alpha=0.85,
                    edgecolor="black")
        ax1.text(b[0].get_x() + bar_w/2, b[0].get_height()+0.003,
                 f"{res['RMSE']:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax1.set_xticks([0,1]); ax1.set_xticklabels([r["model"] for r in results], fontsize=10)
    ax1.set_title("RMSE Karşılaştırması", fontsize=11)
    ax1.set_ylabel("RMSE"); ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    for i, (res, col) in enumerate(zip(results, colors)):
        b = ax2.bar(i, res["R2"], bar_w, color=col, alpha=0.85, edgecolor="black")
        ax2.text(b[0].get_x() + bar_w/2, b[0].get_height()+0.005,
                 f"{res['R2']:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xticks([0,1]); ax2.set_xticklabels([r["model"] for r in results], fontsize=10)
    ax2.set_title("R² Karşılaştırması", fontsize=11)
    ax2.set_ylabel("R²"); ax2.set_ylim(0, 1.1); ax2.grid(axis="y", alpha=0.3)

    # ── (1,0)-(1,1): Tablo ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.axis("off")
    table_data  = [[r["model"], r["RMSE"], r["MAE"], f"{r['MAPE']:.2f}%", r["R2"]]
                   for r in results]
    # İyileşme satırı
    if len(results) == 2:
        imp_rmse = (results[0]["RMSE"] - results[1]["RMSE"]) / results[0]["RMSE"] * 100
        imp_mae  = (results[0]["MAE"]  - results[1]["MAE"])  / results[0]["MAE"]  * 100
        imp_r2   = results[1]["R2"]   - results[0]["R2"]
        table_data.append([
            "GA İyileşmesi",
            f"{imp_rmse:+.1f}%",
            f"{imp_mae:+.1f}%",
            "—",
            f"{imp_r2:+.4f}"
        ])
    table = ax3.table(
        cellText   = table_data,
        colLabels  = ["Model", "RMSE", "MAE", "MAPE", "R²"],
        cellLoc    = "center",
        loc        = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    # Başlık satırı rengi
    for col in range(5):
        table[0, col].set_facecolor("#2c3e50")
        table[0, col].set_text_props(color="white", fontweight="bold")
    ax3.set_title("Performans Özet Tablosu", fontsize=11, pad=12)

    # ── (1,2): MAE & MAPE bar ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    x   = np.arange(2)
    mae_vals  = [r["MAE"]  for r in results]
    mape_vals = [r["MAPE"] for r in results]
    bars_mae  = ax4.bar(x - 0.2, mae_vals,  0.35, label="MAE",      color="#2ca02c", alpha=0.8)
    bars_mape = ax4.bar(x + 0.2, mape_vals, 0.35, label="MAPE (%)", color="#9467bd", alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels([r["model"] for r in results], fontsize=9)
    ax4.set_title("MAE & MAPE", fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(axis="y", alpha=0.3)

    path = os.path.join(output_dir, "karsilastirma_ga_vs_klasik.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Birleşik rapor grafiği: {path}")
    return path


# ─── Ana Akış ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Asama 8: Klasik Sugeno vs GA-Sugeno Karsilastirmasi")
    print("=" * 60)

    # ── Veri yükle ────────────────────────────────────────────────────────────
    train_path = os.path.join(MODELS_DIR, "desharnais_train_clean.csv")
    test_path  = os.path.join(MODELS_DIR, "desharnais_test_clean.csv")
    if not os.path.exists(train_path):
        print("HATA: 01_eda_desharnais.py önce çalıştırılmalı.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Sütun tespiti
    possible_features = [
        ["PointsAjust_norm","Length_norm","TeamExp_norm"],
        ["PointsAjust","Length","TeamExp"],
    ]
    feature_cols, target_col = None, None
    for fc in possible_features:
        if fc[0] in train_df.columns:
            feature_cols = fc
            break
    for tc in ["Effort_norm", "Effort"]:
        if tc in train_df.columns:
            target_col = tc
            break

    if feature_cols is None or target_col is None:
        print(f"HATA: Beklenen sütunlar bulunamadı. Mevcut: {list(train_df.columns)}")
        sys.exit(1)

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df[target_col].values.astype(float)
    X_test  = test_df[feature_cols].values.astype(float)
    y_test  = test_df[target_col].values.astype(float)

    print(f"\nVeri: Egitim={len(X_train)}, Test={len(X_test)}")
    print(f"Girdi sutunlar: {feature_cols}")

    # ── Klasik Sugeno ─────────────────────────────────────────────────────────
    print("\n[1] Klasik Sugeno yukleniyor...")
    klasik_chrom = load_classic_sugeno(X_train, y_train)
    klasik_pred_test = sugeno_predict(X_test, klasik_chrom)
    klasik_metrics   = compute_metrics(y_test, klasik_pred_test, "Klasik Sugeno")
    print(f"    RMSE={klasik_metrics['RMSE']}, MAE={klasik_metrics['MAE']}, "
          f"MAPE={klasik_metrics['MAPE']}%, R2={klasik_metrics['R2']}")

    # ── GA-Sugeno ─────────────────────────────────────────────────────────────
    print("\n[2] GA-Sugeno chromosome yukleniyor...")
    ga_chrom_path = os.path.join(MODELS_DIR, "ga_best_chromosome.npy")
    if not os.path.exists(ga_chrom_path):
        print("    UYARI: GA chromosome bulunamadi. Once 07_ga_optimizer.py calistirin.")
        sys.exit(1)

    ga_chrom      = np.load(ga_chrom_path)
    ga_pred_test  = sugeno_predict(X_test, ga_chrom)
    ga_metrics    = compute_metrics(y_test, ga_pred_test, "GA-Sugeno")
    print(f"    RMSE={ga_metrics['RMSE']}, MAE={ga_metrics['MAE']}, "
          f"MAPE={ga_metrics['MAPE']}%, R2={ga_metrics['R2']}")

    # ── GA konverjans ─────────────────────────────────────────────────────────
    conv_path = os.path.join(MODELS_DIR, "ga_convergence.npy")
    history_rmse = np.load(conv_path).tolist() if os.path.exists(conv_path) else None

    # ── Sonuçlar ──────────────────────────────────────────────────────────────
    results = [klasik_metrics, ga_metrics]

    # GA iyileşmesi
    imp_rmse = (klasik_metrics["RMSE"] - ga_metrics["RMSE"]) / max(klasik_metrics["RMSE"], 1e-9) * 100
    print("\n" + "=" * 60)
    print("SONUC TABLOSU")
    print("=" * 60)
    print(f"{'Metrik':<12} {'Klasik Sugeno':>16} {'GA-Sugeno':>14} {'Iyilesme':>12}")
    print("-" * 56)
    for m in ["RMSE", "MAE", "MAPE", "R2"]:
        v1, v2 = klasik_metrics[m], ga_metrics[m]
        diff   = v2 - v1
        sign   = "<- daha iyi" if (m != "R2" and diff < 0) else ("<- daha iyi" if m == "R2" and diff > 0 else "->")
        print(f"{m:<12} {v1:>16.4f} {v2:>14.4f} {sign:>12}")
    print(f"\n  RMSE Iyilesmesi: %{imp_rmse:.1f}")

    # ── Kaydet ────────────────────────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, "karsilastirma_sonuclari.json"), "w", encoding="utf-8") as f:
        json.dump({
            "klasik_sugeno": klasik_metrics,
            "ga_sugeno"    : ga_metrics,
            "rmse_iyilesmesi_pct": round(float(imp_rmse), 2),
        }, f, indent=2, ensure_ascii=False)

    # ── Grafikler ─────────────────────────────────────────────────────────────
    print("\nGrafikler oluşturuluyor...")
    plot_metric_comparison(results, OUTPUT_DIR)
    plot_scatter_comparison(
        [("Klasik Sugeno", klasik_pred_test), ("GA-Sugeno", ga_pred_test)],
        y_test, OUTPUT_DIR
    )
    plot_combined_report(results, history_rmse, OUTPUT_DIR)

    print("\n[OK] Asama 8 tamamlandi.")


if __name__ == "__main__":
    main()
