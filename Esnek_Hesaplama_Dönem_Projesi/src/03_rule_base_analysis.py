"""
Asama 3: Kural Tabani (Rule Base) Analizi - Hem Desharnais Hem Albrecht
- 4 farkli kural seti: Student, ChatGPT, Claude, Gemini
- Celiksi kural kontrolu
- Kural kapsam analizi (ateslenme gucu)
- LLM karsilastirma ozeti
Calistirma: python src/03_rule_base_analysis.py
"""
import pathlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

# --- Dizin Yapisi ---
BASE_DIR   = pathlib.Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# 1. DESHARNAIS KURAL SETLERİ
# ==========================================
RULE_SETS_DESHARNAIS = {
    "Student": [
        {"id": 1,  "antecedent": ["Low",    "Low",    "Low"],    "consequent": {"p1": 0.8,  "p2": 0.6,  "p3": -0.4,  "c": 0.05}},
        {"id": 2,  "antecedent": ["Low",    "Low",    "Medium"], "consequent": {"p1": 0.7,  "p2": 0.5,  "p3": -0.5,  "c": 0.08}},
        {"id": 3,  "antecedent": ["Low",    "Low",    "High"],   "consequent": {"p1": 0.6,  "p2": 0.4,  "p3": -0.6,  "c": 0.10}},
        {"id": 4,  "antecedent": ["Low",    "Medium", "Low"],    "consequent": {"p1": 0.9,  "p2": 0.8,  "p3": -0.3,  "c": 0.06}},
        {"id": 5,  "antecedent": ["Low",    "Medium", "Medium"], "consequent": {"p1": 0.8,  "p2": 0.7,  "p3": -0.4,  "c": 0.09}},
        {"id": 6,  "antecedent": ["Low",    "Medium", "High"],   "consequent": {"p1": 0.7,  "p2": 0.6,  "p3": -0.5,  "c": 0.12}},
        {"id": 7,  "antecedent": ["Low",    "High",   "Low"],    "consequent": {"p1": 1.0,  "p2": 1.0,  "p3": -0.2,  "c": 0.08}},
        {"id": 8,  "antecedent": ["Low",    "High",   "Medium"], "consequent": {"p1": 0.9,  "p2": 0.9,  "p3": -0.3,  "c": 0.11}},
        {"id": 9,  "antecedent": ["Medium", "Low",    "Low"],    "consequent": {"p1": 1.1,  "p2": 0.5,  "p3": -0.3,  "c": 0.10}},
        {"id": 10, "antecedent": ["Medium", "Low",    "Medium"], "consequent": {"p1": 1.0,  "p2": 0.4,  "p3": -0.4,  "c": 0.13}},
        {"id": 11, "antecedent": ["Medium", "Medium", "Low"],    "consequent": {"p1": 1.2,  "p2": 0.8,  "p3": -0.2,  "c": 0.12}},
        {"id": 12, "antecedent": ["Medium", "Medium", "Medium"], "consequent": {"p1": 1.1,  "p2": 0.7,  "p3": -0.3,  "c": 0.15}},
        {"id": 13, "antecedent": ["Medium", "Medium", "High"],   "consequent": {"p1": 1.0,  "p2": 0.6,  "p3": -0.4,  "c": 0.18}},
        {"id": 14, "antecedent": ["Medium", "High",   "Low"],    "consequent": {"p1": 1.3,  "p2": 1.1,  "p3": -0.1,  "c": 0.14}},
        {"id": 15, "antecedent": ["Medium", "High",   "Medium"], "consequent": {"p1": 1.2,  "p2": 1.0,  "p3": -0.2,  "c": 0.17}},
        {"id": 16, "antecedent": ["High",   "Low",    "Low"],    "consequent": {"p1": 1.4,  "p2": 0.4,  "p3": -0.2,  "c": 0.15}},
        {"id": 17, "antecedent": ["High",   "Low",    "Medium"], "consequent": {"p1": 1.3,  "p2": 0.3,  "p3": -0.3,  "c": 0.18}},
        {"id": 18, "antecedent": ["High",   "Medium", "Low"],    "consequent": {"p1": 1.5,  "p2": 0.7,  "p3": -0.1,  "c": 0.17}},
        {"id": 19, "antecedent": ["High",   "Medium", "Medium"], "consequent": {"p1": 1.4,  "p2": 0.6,  "p3": -0.2,  "c": 0.20}},
        {"id": 20, "antecedent": ["High",   "High",   "High"],   "consequent": {"p1": 1.6,  "p2": 1.2,  "p3": -0.5,  "c": 0.25}},
    ],
    "ChatGPT": [
        {"id": 1,  "antecedent": ["Low",    "Low",    "Low"],    "consequent": {"p1": 0.55, "p2": 0.30, "p3": -0.18, "c": 0.08}},
        {"id": 2,  "antecedent": ["Low",    "Low",    "Medium"], "consequent": {"p1": 0.50, "p2": 0.28, "p3": -0.24, "c": 0.06}},
        {"id": 3,  "antecedent": ["Low",    "Low",    "High"],   "consequent": {"p1": 0.45, "p2": 0.25, "p3": -0.32, "c": 0.05}},
        {"id": 4,  "antecedent": ["Low",    "Medium", "Low"],    "consequent": {"p1": 0.58, "p2": 0.42, "p3": -0.16, "c": 0.10}},
        {"id": 5,  "antecedent": ["Low",    "Medium", "High"],   "consequent": {"p1": 0.50, "p2": 0.38, "p3": -0.30, "c": 0.07}},
        {"id": 6,  "antecedent": ["Low",    "High",   "Medium"], "consequent": {"p1": 0.54, "p2": 0.55, "p3": -0.25, "c": 0.12}},
        {"id": 7,  "antecedent": ["Medium", "Low",    "Low"],    "consequent": {"p1": 0.72, "p2": 0.30, "p3": -0.15, "c": 0.14}},
        {"id": 8,  "antecedent": ["Medium", "Low",    "Medium"], "consequent": {"p1": 0.68, "p2": 0.28, "p3": -0.22, "c": 0.11}},
        {"id": 9,  "antecedent": ["Medium", "Low",    "High"],   "consequent": {"p1": 0.64, "p2": 0.26, "p3": -0.34, "c": 0.09}},
        {"id": 10, "antecedent": ["Medium", "Medium", "Low"],    "consequent": {"p1": 0.76, "p2": 0.46, "p3": -0.16, "c": 0.16}},
        {"id": 11, "antecedent": ["Medium", "Medium", "Medium"], "consequent": {"p1": 0.72, "p2": 0.42, "p3": -0.24, "c": 0.13}},
        {"id": 12, "antecedent": ["Medium", "Medium", "High"],   "consequent": {"p1": 0.68, "p2": 0.40, "p3": -0.36, "c": 0.10}},
        {"id": 13, "antecedent": ["Medium", "High",   "Low"],    "consequent": {"p1": 0.78, "p2": 0.62, "p3": -0.14, "c": 0.18}},
        {"id": 14, "antecedent": ["Medium", "High",   "High"],   "consequent": {"p1": 0.70, "p2": 0.58, "p3": -0.34, "c": 0.13}},
        {"id": 15, "antecedent": ["High",   "Low",    "Medium"], "consequent": {"p1": 0.92, "p2": 0.32, "p3": -0.22, "c": 0.18}},
        {"id": 16, "antecedent": ["High",   "Low",    "High"],   "consequent": {"p1": 0.88, "p2": 0.30, "p3": -0.38, "c": 0.15}},
        {"id": 17, "antecedent": ["High",   "Medium", "Low"],    "consequent": {"p1": 1.00, "p2": 0.50, "p3": -0.14, "c": 0.22}},
        {"id": 18, "antecedent": ["High",   "Medium", "High"],   "consequent": {"p1": 0.92, "p2": 0.46, "p3": -0.36, "c": 0.17}},
        {"id": 19, "antecedent": ["High",   "High",   "Medium"], "consequent": {"p1": 1.05, "p2": 0.70, "p3": -0.24, "c": 0.25}},
        {"id": 20, "antecedent": ["High",   "High",   "High"],   "consequent": {"p1": 0.98, "p2": 0.66, "p3": -0.40, "c": 0.20}},
    ],
    "Claude": [
        {"id": 1,  "antecedent": ["Low",    "Low",    "High"],   "consequent": {"p1": 0.40, "p2": 0.25, "p3": -0.35, "c": 0.08}},
        {"id": 2,  "antecedent": ["Low",    "Low",    "Medium"], "consequent": {"p1": 0.45, "p2": 0.28, "p3": -0.28, "c": 0.10}},
        {"id": 3,  "antecedent": ["Low",    "Low",    "Low"],    "consequent": {"p1": 0.52, "p2": 0.32, "p3": -0.20, "c": 0.12}},
        {"id": 4,  "antecedent": ["Low",    "Medium", "High"],   "consequent": {"p1": 0.48, "p2": 0.42, "p3": -0.32, "c": 0.09}},
        {"id": 5,  "antecedent": ["Low",    "Medium", "Medium"], "consequent": {"p1": 0.55, "p2": 0.48, "p3": -0.25, "c": 0.11}},
        {"id": 6,  "antecedent": ["Low",    "Medium", "Low"],    "consequent": {"p1": 0.62, "p2": 0.55, "p3": -0.18, "c": 0.14}},
        {"id": 7,  "antecedent": ["Low",    "High",   "Medium"], "consequent": {"p1": 0.58, "p2": 0.65, "p3": -0.22, "c": 0.13}},
        {"id": 8,  "antecedent": ["Low",    "High",   "Low"],    "consequent": {"p1": 0.65, "p2": 0.72, "p3": -0.15, "c": 0.16}},
        {"id": 9,  "antecedent": ["Medium", "Low",    "High"],   "consequent": {"p1": 0.70, "p2": 0.30, "p3": -0.30, "c": 0.12}},
        {"id": 10, "antecedent": ["Medium", "Low",    "Medium"], "consequent": {"p1": 0.76, "p2": 0.35, "p3": -0.22, "c": 0.15}},
        {"id": 11, "antecedent": ["Medium", "Medium", "High"],   "consequent": {"p1": 0.78, "p2": 0.52, "p3": -0.28, "c": 0.18}},
        {"id": 12, "antecedent": ["Medium", "Medium", "Medium"], "consequent": {"p1": 0.85, "p2": 0.58, "p3": -0.20, "c": 0.22}},
        {"id": 13, "antecedent": ["Medium", "Medium", "Low"],    "consequent": {"p1": 0.92, "p2": 0.65, "p3": -0.12, "c": 0.26}},
        {"id": 14, "antecedent": ["Medium", "High",   "High"],   "consequent": {"p1": 0.82, "p2": 0.78, "p3": -0.32, "c": 0.20}},
        {"id": 15, "antecedent": ["Medium", "High",   "Low"],    "consequent": {"p1": 0.98, "p2": 0.88, "p3": -0.10, "c": 0.30}},
        {"id": 16, "antecedent": ["High",   "Low",    "High"],   "consequent": {"p1": 0.95, "p2": 0.38, "p3": -0.25, "c": 0.20}},
        {"id": 17, "antecedent": ["High",   "Low",    "Medium"], "consequent": {"p1": 1.05, "p2": 0.42, "p3": -0.18, "c": 0.25}},
        {"id": 18, "antecedent": ["High",   "Medium", "High"],   "consequent": {"p1": 1.02, "p2": 0.62, "p3": -0.28, "c": 0.24}},
        {"id": 19, "antecedent": ["High",   "Medium", "Low"],    "consequent": {"p1": 1.15, "p2": 0.72, "p3": -0.12, "c": 0.32}},
        {"id": 20, "antecedent": ["High",   "High",   "Medium"], "consequent": {"p1": 1.10, "p2": 0.95, "p3": -0.22, "c": 0.35}},
    ],
    "Gemini": [
        {"id": 1,  "antecedent": ["Low",    "Low",    "High"],   "consequent": {"p1": 0.4,  "p2": 0.3,  "p3": -0.20, "c": 0.10}},
        {"id": 2,  "antecedent": ["Low",    "Low",    "Medium"], "consequent": {"p1": 0.4,  "p2": 0.3,  "p3": -0.10, "c": 0.15}},
        {"id": 3,  "antecedent": ["Low",    "Low",    "Low"],    "consequent": {"p1": 0.5,  "p2": 0.4,  "p3": -0.05, "c": 0.20}},
        {"id": 4,  "antecedent": ["Medium", "Medium", "High"],   "consequent": {"p1": 0.6,  "p2": 0.5,  "p3": -0.25, "c": 0.20}},
        {"id": 5,  "antecedent": ["Medium", "Medium", "Medium"], "consequent": {"p1": 0.7,  "p2": 0.6,  "p3": -0.15, "c": 0.25}},
        {"id": 6,  "antecedent": ["Medium", "Medium", "Low"],    "consequent": {"p1": 0.8,  "p2": 0.7,  "p3": -0.05, "c": 0.35}},
        {"id": 7,  "antecedent": ["High",   "High",   "High"],   "consequent": {"p1": 0.9,  "p2": 0.8,  "p3": -0.30, "c": 0.30}},
        {"id": 8,  "antecedent": ["High",   "High",   "Medium"], "consequent": {"p1": 1.0,  "p2": 0.9,  "p3": -0.20, "c": 0.40}},
        {"id": 9,  "antecedent": ["High",   "High",   "Low"],    "consequent": {"p1": 1.2,  "p2": 1.1,  "p3": -0.10, "c": 0.50}},
        {"id": 10, "antecedent": ["High",   "Low",    "High"],   "consequent": {"p1": 0.8,  "p2": 0.6,  "p3": -0.20, "c": 0.25}},
        {"id": 11, "antecedent": ["Low",    "High",   "Medium"], "consequent": {"p1": 0.4,  "p2": 0.5,  "p3": -0.15, "c": 0.20}},
        {"id": 12, "antecedent": ["Medium", "Low",    "High"],   "consequent": {"p1": 0.6,  "p2": 0.4,  "p3": -0.20, "c": 0.15}},
        {"id": 13, "antecedent": ["Medium", "High",   "Low"],    "consequent": {"p1": 0.7,  "p2": 0.9,  "p3": -0.05, "c": 0.40}},
        {"id": 14, "antecedent": ["High",   "Medium", "Medium"], "consequent": {"p1": 0.9,  "p2": 0.7,  "p3": -0.20, "c": 0.35}},
        {"id": 15, "antecedent": ["Low",    "Medium", "Low"],    "consequent": {"p1": 0.5,  "p2": 0.6,  "p3": -0.10, "c": 0.25}},
        {"id": 16, "antecedent": ["Medium", "High",   "High"],   "consequent": {"p1": 0.6,  "p2": 0.7,  "p3": -0.30, "c": 0.20}},
        {"id": 17, "antecedent": ["Low",    "High",   "High"],   "consequent": {"p1": 0.3,  "p2": 0.4,  "p3": -0.20, "c": 0.10}},
        {"id": 18, "antecedent": ["High",   "Low",    "Low"],    "consequent": {"p1": 1.1,  "p2": 0.8,  "p3": -0.05, "c": 0.45}},
        {"id": 19, "antecedent": ["Medium", "Low",    "Medium"], "consequent": {"p1": 0.7,  "p2": 0.5,  "p3": -0.15, "c": 0.20}},
        {"id": 20, "antecedent": ["High",   "Medium", "High"],   "consequent": {"p1": 0.8,  "p2": 0.6,  "p3": -0.30, "c": 0.25}},
    ]
}

# ==========================================
# 2. ALBRECHT KURAL SETLERİ
# ==========================================
RULE_SETS_ALBRECHT = {
    "Student": [
        {"id": 1,  "antecedent": ["Low",    "Low",    "Low"],    "consequent": {"p1": 0.25, "p2": 0.20, "p3": 0.15, "c": 0.05}},
        {"id": 2,  "antecedent": ["Low",    "Low",    "Medium"], "consequent": {"p1": 0.25, "p2": 0.20, "p3": 0.30, "c": 0.07}},
        {"id": 3,  "antecedent": ["Low",    "Low",    "High"],   "consequent": {"p1": 0.25, "p2": 0.20, "p3": 0.45, "c": 0.10}},
        {"id": 4,  "antecedent": ["Low",    "Medium", "Low"],    "consequent": {"p1": 0.25, "p2": 0.45, "p3": 0.15, "c": 0.08}},
        {"id": 5,  "antecedent": ["Low",    "Medium", "Medium"], "consequent": {"p1": 0.25, "p2": 0.45, "p3": 0.30, "c": 0.10}},
        {"id": 6,  "antecedent": ["Low",    "Medium", "High"],   "consequent": {"p1": 0.25, "p2": 0.45, "p3": 0.45, "c": 0.12}},
        {"id": 7,  "antecedent": ["Low",    "High",   "Low"],    "consequent": {"p1": 0.25, "p2": 0.70, "p3": 0.15, "c": 0.12}},
        {"id": 8,  "antecedent": ["Low",    "High",   "Medium"], "consequent": {"p1": 0.25, "p2": 0.70, "p3": 0.30, "c": 0.15}},
        {"id": 9,  "antecedent": ["Medium", "Low",    "Low"],    "consequent": {"p1": 0.50, "p2": 0.20, "p3": 0.15, "c": 0.10}},
        {"id": 10, "antecedent": ["Medium", "Low",    "Medium"], "consequent": {"p1": 0.50, "p2": 0.20, "p3": 0.30, "c": 0.12}},
        {"id": 11, "antecedent": ["Medium", "Medium", "Low"],    "consequent": {"p1": 0.50, "p2": 0.45, "p3": 0.15, "c": 0.12}},
        {"id": 12, "antecedent": ["Medium", "Medium", "Medium"], "consequent": {"p1": 0.50, "p2": 0.45, "p3": 0.30, "c": 0.15}},
        {"id": 13, "antecedent": ["Medium", "Medium", "High"],   "consequent": {"p1": 0.50, "p2": 0.45, "p3": 0.45, "c": 0.18}},
        {"id": 14, "antecedent": ["Medium", "High",   "Low"],    "consequent": {"p1": 0.50, "p2": 0.70, "p3": 0.15, "c": 0.16}},
        {"id": 15, "antecedent": ["Medium", "High",   "Medium"], "consequent": {"p1": 0.50, "p2": 0.70, "p3": 0.30, "c": 0.18}},
        {"id": 16, "antecedent": ["High",   "Low",    "Low"],    "consequent": {"p1": 0.80, "p2": 0.20, "p3": 0.15, "c": 0.15}},
        {"id": 17, "antecedent": ["High",   "Low",    "Medium"], "consequent": {"p1": 0.80, "p2": 0.20, "p3": 0.30, "c": 0.18}},
        {"id": 18, "antecedent": ["High",   "Medium", "Low"],    "consequent": {"p1": 0.80, "p2": 0.45, "p3": 0.15, "c": 0.18}},
        {"id": 19, "antecedent": ["High",   "Medium", "Medium"], "consequent": {"p1": 0.80, "p2": 0.45, "p3": 0.30, "c": 0.20}},
        {"id": 20, "antecedent": ["High",   "High",   "High"],   "consequent": {"p1": 0.90, "p2": 0.85, "p3": 0.60, "c": 0.25}}
    ],
    "ChatGPT": [
        {"id": 1,  "antecedent": ["Low",    "Low",    "Low"],    "consequent": {"p1": 0.22, "p2": 0.18, "p3": 0.12, "c": 0.04}},
        {"id": 2,  "antecedent": ["Low",    "Low",    "Medium"], "consequent": {"p1": 0.22, "p2": 0.18, "p3": 0.26, "c": 0.06}},
        {"id": 3,  "antecedent": ["Low",    "Low",    "High"],   "consequent": {"p1": 0.22, "p2": 0.18, "p3": 0.40, "c": 0.08}},
        {"id": 4,  "antecedent": ["Low",    "Medium", "Low"],    "consequent": {"p1": 0.22, "p2": 0.40, "p3": 0.12, "c": 0.07}},
        {"id": 5,  "antecedent": ["Low",    "Medium", "High"],   "consequent": {"p1": 0.22, "p2": 0.40, "p3": 0.40, "c": 0.10}},
        {"id": 6,  "antecedent": ["Low",    "High",   "Medium"], "consequent": {"p1": 0.22, "p2": 0.65, "p3": 0.26, "c": 0.12}},
        {"id": 7,  "antecedent": ["Medium", "Low",    "Low"],    "consequent": {"p1": 0.45, "p2": 0.18, "p3": 0.12, "c": 0.08}},
        {"id": 8,  "antecedent": ["Medium", "Low",    "Medium"], "consequent": {"p1": 0.45, "p2": 0.18, "p3": 0.26, "c": 0.10}},
        {"id": 9,  "antecedent": ["Medium", "Low",    "High"],   "consequent": {"p1": 0.45, "p2": 0.18, "p3": 0.40, "c": 0.12}},
        {"id": 10, "antecedent": ["Medium", "Medium", "Low"],    "consequent": {"p1": 0.45, "p2": 0.40, "p3": 0.12, "c": 0.10}},
        {"id": 11, "antecedent": ["Medium", "Medium", "Medium"], "consequent": {"p1": 0.45, "p2": 0.40, "p3": 0.26, "c": 0.12}},
        {"id": 12, "antecedent": ["Medium", "Medium", "High"],   "consequent": {"p1": 0.45, "p2": 0.40, "p3": 0.40, "c": 0.14}},
        {"id": 13, "antecedent": ["Medium", "High",   "Low"],    "consequent": {"p1": 0.45, "p2": 0.65, "p3": 0.12, "c": 0.14}},
        {"id": 14, "antecedent": ["Medium", "High",   "High"],   "consequent": {"p1": 0.45, "p2": 0.65, "p3": 0.40, "c": 0.16}},
        {"id": 15, "antecedent": ["High",   "Low",    "Medium"], "consequent": {"p1": 0.75, "p2": 0.18, "p3": 0.26, "c": 0.15}},
        {"id": 16, "antecedent": ["High",   "Low",    "High"],   "consequent": {"p1": 0.75, "p2": 0.18, "p3": 0.40, "c": 0.18}},
        {"id": 17, "antecedent": ["High",   "Medium", "Low"],    "consequent": {"p1": 0.75, "p2": 0.40, "p3": 0.12, "c": 0.16}},
        {"id": 18, "antecedent": ["High",   "Medium", "High"],   "consequent": {"p1": 0.75, "p2": 0.40, "p3": 0.40, "c": 0.20}},
        {"id": 19, "antecedent": ["High",   "High",   "Medium"], "consequent": {"p1": 0.75, "p2": 0.65, "p3": 0.26, "c": 0.22}},
        {"id": 20, "antecedent": ["High",   "High",   "High"],   "consequent": {"p1": 0.85, "p2": 0.80, "p3": 0.55, "c": 0.24}}
    ],
    "Claude": [
        {"id": 1,  "antecedent": ["Low",    "Low",    "High"],   "consequent": {"p1": 0.20, "p2": 0.15, "p3": 0.35, "c": 0.05}},
        {"id": 2,  "antecedent": ["Low",    "Low",    "Medium"], "consequent": {"p1": 0.20, "p2": 0.15, "p3": 0.22, "c": 0.04}},
        {"id": 3,  "antecedent": ["Low",    "Low",    "Low"],    "consequent": {"p1": 0.20, "p2": 0.15, "p3": 0.10, "c": 0.03}},
        {"id": 4,  "antecedent": ["Low",    "Medium", "High"],   "consequent": {"p1": 0.20, "p2": 0.35, "p3": 0.35, "c": 0.08}},
        {"id": 5,  "antecedent": ["Low",    "Medium", "Medium"], "consequent": {"p1": 0.20, "p2": 0.35, "p3": 0.22, "c": 0.07}},
        {"id": 6,  "antecedent": ["Low",    "Medium", "Low"],    "consequent": {"p1": 0.20, "p2": 0.35, "p3": 0.10, "c": 0.06}},
        {"id": 7,  "antecedent": ["Low",    "High",   "Medium"], "consequent": {"p1": 0.20, "p2": 0.60, "p3": 0.22, "c": 0.10}},
        {"id": 8,  "antecedent": ["Low",    "High",   "Low"],    "consequent": {"p1": 0.20, "p2": 0.60, "p3": 0.10, "c": 0.08}},
        {"id": 9,  "antecedent": ["Medium", "Low",    "High"],   "consequent": {"p1": 0.40, "p2": 0.15, "p3": 0.35, "c": 0.08}},
        {"id": 10, "antecedent": ["Medium", "Low",    "Medium"], "consequent": {"p1": 0.40, "p2": 0.15, "p3": 0.22, "c": 0.07}},
        {"id": 11, "antecedent": ["Medium", "Medium", "High"],   "consequent": {"p1": 0.40, "p2": 0.35, "p3": 0.35, "c": 0.12}},
        {"id": 12, "antecedent": ["Medium", "Medium", "Medium"], "consequent": {"p1": 0.40, "p2": 0.35, "p3": 0.22, "c": 0.10}},
        {"id": 13, "antecedent": ["Medium", "Medium", "Low"],    "consequent": {"p1": 0.40, "p2": 0.35, "p3": 0.10, "c": 0.08}},
        {"id": 14, "antecedent": ["Medium", "High",   "High"],   "consequent": {"p1": 0.40, "p2": 0.60, "p3": 0.35, "c": 0.15}},
        {"id": 15, "antecedent": ["Medium", "High",   "Low"],    "consequent": {"p1": 0.40, "p2": 0.60, "p3": 0.10, "c": 0.12}},
        {"id": 16, "antecedent": ["High",   "Low",    "High"],   "consequent": {"p1": 0.70, "p2": 0.15, "p3": 0.35, "c": 0.15}},
        {"id": 17, "antecedent": ["High",   "Low",    "Medium"], "consequent": {"p1": 0.70, "p2": 0.15, "p3": 0.22, "c": 0.12}},
        {"id": 18, "antecedent": ["High",   "Medium", "High"],   "consequent": {"p1": 0.70, "p2": 0.35, "p3": 0.35, "c": 0.18}},
        {"id": 19, "antecedent": ["High",   "Medium", "Low"],    "consequent": {"p1": 0.70, "p2": 0.35, "p3": 0.10, "c": 0.14}},
        {"id": 20, "antecedent": ["High",   "High",   "Medium"], "consequent": {"p1": 0.70, "p2": 0.60, "p3": 0.22, "c": 0.20}}
    ],
    "Gemini": [
        {"id": 1,  "antecedent": ["Low",    "Low",    "High"],   "consequent": {"p1": 0.18, "p2": 0.12, "p3": 0.30, "c": 0.04}},
        {"id": 2,  "antecedent": ["Low",    "Low",    "Medium"], "consequent": {"p1": 0.18, "p2": 0.12, "p3": 0.20, "c": 0.03}},
        {"id": 3,  "antecedent": ["Low",    "Low",    "Low"],    "consequent": {"p1": 0.18, "p2": 0.12, "p3": 0.08, "c": 0.02}},
        {"id": 4,  "antecedent": ["Medium", "Medium", "High"],   "consequent": {"p1": 0.38, "p2": 0.30, "p3": 0.30, "c": 0.10}},
        {"id": 5,  "antecedent": ["Medium", "Medium", "Medium"], "consequent": {"p1": 0.38, "p2": 0.30, "p3": 0.20, "c": 0.08}},
        {"id": 6,  "antecedent": ["Medium", "Medium", "Low"],    "consequent": {"p1": 0.38, "p2": 0.30, "p3": 0.08, "c": 0.06}},
        {"id": 7,  "antecedent": ["High",   "High",   "High"],   "consequent": {"p1": 0.75, "p2": 0.70, "p3": 0.50, "c": 0.22}},
        {"id": 8,  "antecedent": ["High",   "High",   "Medium"], "consequent": {"p1": 0.75, "p2": 0.70, "p3": 0.35, "c": 0.18}},
        {"id": 9,  "antecedent": ["High",   "High",   "Low"],    "consequent": {"p1": 0.75, "p2": 0.70, "p3": 0.15, "c": 0.15}},
        {"id": 10, "antecedent": ["High",   "Low",    "High"],   "consequent": {"p1": 0.75, "p2": 0.12, "p3": 0.30, "c": 0.12}},
        {"id": 11, "antecedent": ["Low",    "High",   "Medium"], "consequent": {"p1": 0.18, "p2": 0.70, "p3": 0.20, "c": 0.10}},
        {"id": 12, "antecedent": ["Medium", "Low",    "High"],   "consequent": {"p1": 0.38, "p2": 0.12, "p3": 0.30, "c": 0.08}},
        {"id": 13, "antecedent": ["Medium", "High",   "Low"],    "consequent": {"p1": 0.38, "p2": 0.70, "p3": 0.08, "c": 0.14}},
        {"id": 14, "antecedent": ["High",   "Medium", "Medium"], "consequent": {"p1": 0.75, "p2": 0.30, "p3": 0.20, "c": 0.15}},
        {"id": 15, "antecedent": ["Low",    "Medium", "Low"],    "consequent": {"p1": 0.18, "p2": 0.30, "p3": 0.08, "c": 0.06}},
        {"id": 16, "antecedent": ["Medium", "High",   "High"],   "consequent": {"p1": 0.38, "p2": 0.70, "p3": 0.30, "c": 0.16}},
        {"id": 17, "antecedent": ["Low",    "High",   "High"],   "consequent": {"p1": 0.18, "p2": 0.70, "p3": 0.30, "c": 0.12}},
        {"id": 18, "antecedent": ["High",   "Low",    "Low"],    "consequent": {"p1": 0.75, "p2": 0.12, "p3": 0.08, "c": 0.10}},
        {"id": 19, "antecedent": ["Medium", "Low",    "Medium"], "consequent": {"p1": 0.38, "p2": 0.12, "p3": 0.20, "c": 0.07}},
        {"id": 20, "antecedent": ["High",   "Medium", "High"],   "consequent": {"p1": 0.75, "p2": 0.30, "p3": 0.30, "c": 0.18}}
    ]
}

DATASETS = {
    'desharnais': {
        'train_file': 'desharnais_train_clean.csv',
        'selected_features': ['PointsAjust', 'Length', 'TeamExp'],
        'rule_sets': RULE_SETS_DESHARNAIS,
        'title': 'Desharnais'
    },
    'albrecht': {
        'train_file': 'albrecht_train_clean.csv',
        'selected_features': ['Input', 'Output', 'File'],
        'rule_sets': RULE_SETS_ALBRECHT,
        'title': 'Albrecht'
    }
}

# ============================
# KURAL ÇELİŞKİ KONTROLÜ
# ============================
def check_conflicts(rule_set, set_name):
    seen = {}
    conflicts = []
    for rule in rule_set:
        ant_key = tuple(rule['antecedent'])
        if ant_key in seen:
            conflicts.append({'rule_1': seen[ant_key]['id'], 'rule_2': rule['id'], 'antecedent': ant_key})
        else:
            seen[ant_key] = rule
    return conflicts

# ============================
# KURAL KAPSAM ANALİZİ
# ============================
def trimf_membership(x, params):
    a, b, c = params
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    else:
        return (c - x) / (c - b) if c != b else 1.0

def get_firing_level(x, antecedent, feature_names, mf_params, dataset_name):
    synonyms = {"Short": "Low", "Long": "High"} if dataset_name == "desharnais" else {}
    levels = []
    for i, feat in enumerate(feature_names):
        val    = x[i]
        mf_name = synonyms.get(antecedent[i], antecedent[i])
        if mf_name not in mf_params['triangular'][feat]:
            return 0.0
        params = mf_params['triangular'][feat][mf_name]
        levels.append(trimf_membership(val, params))
    return np.prod(levels)

for dname, config in DATASETS.items():
    print("\n" + "=" * 70)
    print(f"VERI SETI: {dname.upper()} - KURAL CELIKSI ANALIZI")
    print("=" * 70)
    
    rule_sets = config['rule_sets']
    for name, rules in rule_sets.items():
        conflicts = check_conflicts(rules, name)
        print(f"\n{name}: {len(rules)} kural")
        if conflicts:
            print(f"  [X] {len(conflicts)} celiksi bulundu!")
            for c in conflicts:
                print(f"     Kural {c['rule_1']} vs Kural {c['rule_2']}: {c['antecedent']}")
        else:
            print(f"  [OK] Celiksi yok.")

    print("\n" + "=" * 70)
    print(f"VERI SETI: {dname.upper()} - KURAL KAPSAM ANALIZI (Egitim Seti)")
    print("=" * 70)
    
    train_df = pd.read_csv(MODELS_DIR / config['train_file'])
    selected_features = config['selected_features']
    X_train = train_df[selected_features].values
    
    with open(MODELS_DIR / f'{dname}_mf_parameters.json', 'r') as f:
        mf_params = json.load(f)

    coverage_results = {}
    for name, rules in rule_sets.items():
        rule_firings = np.zeros(len(rules))
        max_firings  = np.zeros(len(rules))
        for sample in X_train:
            for i, rule in enumerate(rules):
                fire = get_firing_level(sample, rule['antecedent'], selected_features, mf_params, dname)
                rule_firings[i] += fire
                if fire > max_firings[i]:
                    max_firings[i] = fire
        avg_firings = rule_firings / len(X_train)
        coverage_results[name] = {
            'rules': rules, 'avg_firing': avg_firings,
            'max_firing': max_firings, 'never_fired': int(np.sum(avg_firings < 0.001))
        }
        print(f"\n[{name}]")
        print(f"   Toplam Kural      : {len(rules)}")
        print(f"   Hic ateslenmeyen  : {coverage_results[name]['never_fired']}")
        print(f"   En yuksek ort.    : {avg_firings.max():.4f} (Kural {avg_firings.argmax()+1})")
        print(f"   En dusuk ort.     : {avg_firings.min():.4f} (Kural {avg_firings.argmin()+1})")

    # ============================
    # GÖRSELLEŞTİRME
    # ============================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()

    for idx, (name, data) in enumerate(coverage_results.items()):
        ax = axes[idx]
        rule_labels = [f"R{r['id']}: {'-'.join(r['antecedent'])}" for r in data['rules']]
        norm_vals   = data['avg_firing'] / (data['avg_firing'].max() + 1e-9)
        bar_colors  = plt.cm.RdYlGn(norm_vals)
        ax.barh(range(len(rule_labels)), data['avg_firing'], color=bar_colors)
        ax.set_yticks(range(len(rule_labels)))
        ax.set_yticklabels(rule_labels, fontsize=7)
        ax.set_xlabel('Ortalama Ateslenme Gucu')
        ax.set_title(f'{name} Kural Tabani\n({len(data["rules"])} kural, {data["never_fired"]} hic ateslenmeyen)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle(f'Kural Kapsam Analizi - Tum Setler ({config["title"]})', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{dname}_rule_coverage_all_models.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] {dname}_rule_coverage_all_models.png kaydedildi.")

    # ============================
    # KARŞILAŞTIRMA TABLOSU
    # ============================
    print("\n" + "=" * 70)
    print(f"LLM KARSILASTIRMA OZETI ({config['title']})")
    print("=" * 70)

    comparison = []
    for name, data in coverage_results.items():
        rules = data['rules']
        # Desharnais te p3 negatif olmasini bekleriz (takim tecrubesi), Albrecht te pozitif (dosya sayisi)
        p1_ok = sum(1 for r in rules if r['consequent']['p1'] > 0)
        p2_ok = sum(1 for r in rules if r['consequent']['p2'] > 0)
        if dname == 'desharnais':
            p3_ok = sum(1 for r in rules if r['consequent']['p3'] < 0)
            p3_label = 'p3<0'
        else:
            p3_ok = sum(1 for r in rules if r['consequent']['p3'] > 0)
            p3_label = 'p3>0'
        
        comparison.append({
            'Model'            : name,
            'Kural Sayisi'     : len(rules),
            'Celiksi'          : 'Yok' if not check_conflicts(rules, name) else f"{len(check_conflicts(rules, name))} var",
            'Hic Ateslenmeyen' : data['never_fired'],
            'Max Ort. Ateslenme': f"{data['avg_firing'].max():.4f}",
            'p1>0'             : f"{p1_ok}/{len(rules)}",
            'p2>0'             : f"{p2_ok}/{len(rules)}",
            p3_label           : f"{p3_ok}/{len(rules)}",
        })

    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))

    # ============================
    # KAYDET
    # ============================
    with open(MODELS_DIR / f'{dname}_rule_sets.json', 'w') as f:
        json.dump(rule_sets, f, indent=2)
    print(f"\n[OK] {dname}_rule_sets.json kaydedildi.")

# Geriye donuk uyumluluk (backward compatibility) icin Desharnais dosyalarini kopyala
shutil.copyfile(MODELS_DIR / "desharnais_rule_sets.json", MODELS_DIR / "rule_sets.json")
shutil.copyfile(OUTPUT_DIR / "desharnais_rule_coverage_all_models.png", OUTPUT_DIR / "rule_coverage_all_models.png")
print("\n[OK] Geriye donuk uyumluluk dosyalari olusturuldu (rule_sets.json vb.)")
