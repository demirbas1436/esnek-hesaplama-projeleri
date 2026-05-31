# =============================================================================
# 06_ga_operators.py
# Genetik Algoritma Operatörleri — Selection / Crossover / Mutation
# Klasik ve GA Tabanlı Sugeno FIS ile Yazılım Efor Tahmini
# =============================================================================
"""
Bu modül üç temel GA operatörünü uygular:

1. Selection   : Tournament Selection (Turnuva Seçimi)
2. Crossover   : Single-Point + Uniform Crossover
3. Mutation    :
     - Float genler (MF params, katsayılar): Gaussian perturbation
     - Binary genler (rule bits)           : Bit-flip

Operatör kararlarının detayları 09_yorum_analizi.py'de incelenecek.
"""

import numpy as np
import os
import sys

# 05_ga_design modülüne erişim
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ga_design_utils import (
    CHROMOSOME_LEN, MF_SLICE_END, RULE_SLICE_END, COEF_SLICE_END,
    repair_chromosome, N_RULES
)


# ─── SELECTION — Tournament Selection ─────────────────────────────────────────

def tournament_selection(population, fitness_scores, tournament_size=3, rng=None):
    """
    Turnuva Seçimi (Tournament Selection).

    Rastgele `tournament_size` birey seçilir, en yüksek fitness'a sahip
    olanı ebeveyn olarak döner. Düşük fitness varyansında bile seçim baskısı sağlar.

    Parameters
    ----------
    population      : ndarray (pop_size, chrom_len)
    fitness_scores  : ndarray (pop_size,)   yüksek = iyi
    tournament_size : int                   default=3
    rng             : numpy.random.Generator

    Returns
    -------
    winner : ndarray (chrom_len,)
    """
    if rng is None:
        rng = np.random.default_rng()

    pop_size  = len(population)
    indices   = rng.choice(pop_size, size=tournament_size, replace=False)
    best_idx  = indices[np.argmax(fitness_scores[indices])]
    return population[best_idx].copy()


def roulette_selection(population, fitness_scores, rng=None):
    """
    Rulet Tekerleği Seçimi (Roulette Wheel / Fitness-Proportionate Selection).

    Fitness orantılı olasılıkla seçim yapar. Fitness değerleri negatif
    olabileceğinden min-shift uygulanır.

    Parameters
    ----------
    population     : ndarray (pop_size, chrom_len)
    fitness_scores : ndarray (pop_size,)

    Returns
    -------
    winner : ndarray (chrom_len,)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Negatif fitness'ı pozitife kaydır
    shifted = fitness_scores - fitness_scores.min() + 1e-9
    probs   = shifted / shifted.sum()
    idx     = rng.choice(len(population), p=probs)
    return population[idx].copy()


def select_parents(population, fitness_scores, method="tournament",
                   tournament_size=3, rng=None):
    """
    İki ebeveyn seçer. Aynı birey seçilmesini önler.

    Parameters
    ----------
    method : 'tournament' | 'roulette'

    Returns
    -------
    parent1, parent2 : ndarray (chrom_len,)
    """
    if rng is None:
        rng = np.random.default_rng()

    if method == "tournament":
        p1 = tournament_selection(population, fitness_scores, tournament_size, rng)
        p2 = tournament_selection(population, fitness_scores, tournament_size, rng)
    else:
        p1 = roulette_selection(population, fitness_scores, rng)
        p2 = roulette_selection(population, fitness_scores, rng)

    return p1, p2


# ─── CROSSOVER ────────────────────────────────────────────────────────────────

def single_point_crossover(parent1, parent2, rng=None):
    """
    Tek Noktalı Çaprazlama (Single-Point Crossover).

    Rastgele bir kesim noktası seçilir.
    child1 = p1[:k] + p2[k:]
    child2 = p2[:k] + p1[k:]

    Returns
    -------
    child1, child2 : ndarray (chrom_len,)
    """
    if rng is None:
        rng = np.random.default_rng()

    k      = rng.integers(1, CHROMOSOME_LEN - 1)
    child1 = np.concatenate([parent1[:k], parent2[k:]])
    child2 = np.concatenate([parent2[:k], parent1[k:]])
    return child1, child2


def uniform_crossover(parent1, parent2, mix_prob=0.5, rng=None):
    """
    Uniform Çaprazlama (Uniform Crossover).

    Her gen pozisyonu için `mix_prob` olasılıkla ebeveynler takas edilir.
    MF parametreleri için daha ince granülerlik sağlar.

    Returns
    -------
    child1, child2 : ndarray (chrom_len,)
    """
    if rng is None:
        rng = np.random.default_rng()

    mask   = rng.random(CHROMOSOME_LEN) < mix_prob
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2


def crossover(parent1, parent2, crossover_rate=0.8, method="single_point", rng=None):
    """
    Çaprazlama operatörü (oran kontrolü ile).

    Parameters
    ----------
    crossover_rate : float  [0,1]  çaprazlama uygulama olasılığı
    method         : 'single_point' | 'uniform'

    Returns
    -------
    child1, child2 : ndarray (chrom_len,)
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() < crossover_rate:
        if method == "uniform":
            c1, c2 = uniform_crossover(parent1, parent2, rng=rng)
        else:
            c1, c2 = single_point_crossover(parent1, parent2, rng=rng)
    else:
        # Çaprazlama uygulanmaz — ebeveynler kopyalanır
        c1, c2 = parent1.copy(), parent2.copy()

    return c1, c2


# ─── MUTATION ─────────────────────────────────────────────────────────────────

def mutate(chromosome, mutation_rate=0.05, sigma_mf=0.05, sigma_coef=0.10, rng=None):
    """
    Mutasyon operatörü — üç bölgeye özelleşmiş:

    1. MF parametreleri (float) : Gaussian perturbation, σ=sigma_mf
    2. Rule selection bitleri   : Bit-flip
    3. Rule katsayıları (float) : Gaussian perturbation, σ=sigma_coef

    Parameters
    ----------
    mutation_rate : float  Her genin mutasyona uğrama olasılığı
    sigma_mf      : float  MF parametreleri için Gaussian σ
    sigma_coef    : float  Katsayılar için Gaussian σ

    Returns
    -------
    mutated : ndarray (chrom_len,)
    """
    if rng is None:
        rng = np.random.default_rng()

    mutated = chromosome.copy()

    # ── Bölge 1: MF parametreleri (float) ─────────────────────────────────────
    for i in range(MF_SLICE_END):
        if rng.random() < mutation_rate:
            mutated[i] += rng.normal(0, sigma_mf)

    # ── Bölge 2: Rule selection bitleri ───────────────────────────────────────
    for i in range(MF_SLICE_END, RULE_SLICE_END):
        if rng.random() < mutation_rate:
            mutated[i] = 1.0 - mutated[i]   # bit-flip

    # ── Bölge 3: Rule katsayıları (float) ─────────────────────────────────────
    for i in range(RULE_SLICE_END, COEF_SLICE_END):
        if rng.random() < mutation_rate:
            mutated[i] += rng.normal(0, sigma_coef)

    # Kısıt düzeltme
    mutated = repair_chromosome(mutated)
    return mutated


# ─── Yeni Nesil Üretimi ───────────────────────────────────────────────────────

def create_next_generation(population, fitness_scores,
                           crossover_rate=0.8,
                           mutation_rate=0.05,
                           selection_method="tournament",
                           crossover_method="single_point",
                           elite_count=2,
                           rng=None):
    """
    Elitizm + GA operatörleri ile yeni nesil üretir.

    Elitizm: En iyi `elite_count` birey direkt aktarılır (bozulmaz).
    Kalan bireyler: Selection → Crossover → Mutation.

    Parameters
    ----------
    elite_count : int  Elitizm ile korunan birey sayısı

    Returns
    -------
    new_population : ndarray (pop_size, chrom_len)
    stats          : dict   {selected_pairs, crossover_applied, mutations_applied}
    """
    if rng is None:
        rng = np.random.default_rng()

    pop_size       = len(population)
    new_population = np.zeros_like(population)
    stats          = {"selected_pairs": 0, "crossover_applied": 0, "mutations_applied": 0}

    # ── Elitizm ───────────────────────────────────────────────────────────────
    elite_indices = np.argsort(fitness_scores)[::-1][:elite_count]
    for i, idx in enumerate(elite_indices):
        new_population[i] = population[idx].copy()

    # ── Yeni bireyler üret ────────────────────────────────────────────────────
    i = elite_count
    while i < pop_size:
        # Ebeveyn seçimi
        p1, p2 = select_parents(population, fitness_scores,
                                 method=selection_method, rng=rng)
        stats["selected_pairs"] += 1

        # Çaprazlama
        before_cross = rng.random()
        c1, c2 = crossover(p1, p2, crossover_rate, crossover_method, rng)
        if not np.array_equal(c1, p1):
            stats["crossover_applied"] += 1

        # Mutasyon
        m1 = mutate(c1, mutation_rate, rng=rng)
        m2 = mutate(c2, mutation_rate, rng=rng)
        if not np.array_equal(m1, c1):
            stats["mutations_applied"] += 1
        if not np.array_equal(m2, c2):
            stats["mutations_applied"] += 1

        new_population[i] = m1
        if i + 1 < pop_size:
            new_population[i + 1] = m2
        i += 2

    return new_population, stats


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GA Operatörleri Testi")
    print("=" * 60)

    # Utilities import
    from ga_design_utils import create_initial_population, decode_chromosome

    rng = np.random.default_rng(42)
    pop = create_initial_population(pop_size=10, seed=42)

    # Sahte fitness: uzunlukları baz alan test
    fitness = rng.uniform(-500, -100, size=10)
    fitness[0] = -80   # En iyi birey

    print(f"\nPopulasyon: {pop.shape}")
    print(f"Fitness değerleri: {np.round(fitness, 1)}")
    print(f"En iyi fitness: {fitness.max():.1f} (birey {np.argmax(fitness)})")

    # Tournament Selection
    winner = tournament_selection(pop, fitness, tournament_size=3, rng=rng)
    _, rb, _ = decode_chromosome(winner)
    print(f"\n[Tournament] Seçilen birey aktif kural sayısı: {int(rb.sum())}")

    # Roulette Selection
    winner2 = roulette_selection(pop, fitness, rng=rng)
    _, rb2, _ = decode_chromosome(winner2)
    print(f"[Roulette  ] Seçilen birey aktif kural sayısı: {int(rb2.sum())}")

    # Single-Point Crossover
    p1, p2 = pop[0], pop[1]
    c1, c2 = single_point_crossover(p1, p2, rng=rng)
    print(f"\n[Single-Point Crossover]")
    print(f"  Ebeveyn 1 MF[0:4]: {np.round(p1[:4], 3)}")
    print(f"  Ebeveyn 2 MF[0:4]: {np.round(p2[:4], 3)}")
    print(f"  Çocuk 1   MF[0:4]: {np.round(c1[:4], 3)}")
    print(f"  Çocuk 2   MF[0:4]: {np.round(c2[:4], 3)}")

    # Mutasyon
    m1 = mutate(p1, mutation_rate=0.3, rng=rng)
    diff = np.abs(m1 - p1)
    print(f"\n[Mutasyon] (rate=0.30)")
    print(f"  Değişen gen sayısı   : {(diff > 1e-9).sum()}")
    print(f"  Ortalama değişim mag.: {diff[diff > 1e-9].mean():.4f}" if (diff > 1e-9).any() else "  Değişim yok")

    # Nesil üretimi
    new_pop, stats = create_next_generation(pop, fitness, rng=rng)
    print(f"\n[Yeni Nesil]")
    print(f"  Seçilen çift sayısı  : {stats['selected_pairs']}")
    print(f"  Çaprazlama uygulanan : {stats['crossover_applied']}")
    print(f"  Mutasyon uygulanan   : {stats['mutations_applied']}")

    print("\n✓ 06_ga_operators.py testi başarılı.")
