# Klasik ve GA Tabanlı Sugeno FIS ile Yazılım Efor Tahmini
**Esnek Hesaplama Final Projesi**

Desharnais yazılım projesi veri seti üzerinde Klasik Sugeno ve Genetik Algoritma
ile optimize edilmiş Sugeno FIS karşılaştırması yapan proje.

---

## Klasör Yapısı

```
Esnek_Hesaplama_Final_Projesi/
├── src/
│   ├── 01_eda_desharnais.py        Aşama 1: Veri Analizi
│   ├── 02_fuzzy_mf_design.py       Aşama 2: Üyelik Fonksiyonu Tasarımı
│   ├── 03_rule_base_analysis.py    Aşama 3: Kural Tabanı (LLM karşılaştırma)
│   ├── 04_sugeno_klasik.py         Aşama 4: Klasik Sugeno Model
│   ├── 05_ga_design.py             Aşama 5: GA Chromosome Encoding (test)
│   ├── 06_ga_operators.py          Aşama 6: GA Operatörleri (test)
│   ├── 07_ga_optimizer.py          Aşama 7: GA Optimizasyon Döngüsü
│   ├── 08_karsilastirma.py         Aşama 8: Klasik vs GA Karşılaştırma
│   ├── 09_yorum_analizi.py         Aşama 9: Operatör Bazlı Yorumlama
│   ├── ga_design.py                GA çekirdek modül (import edilir)
│   └── ga_operators.py             GA operatör modülü (import edilir)
├── dataset/
│   └── desharnais.csv
├── models/                         Eğitilmiş model dosyaları
├── output/                         Üretilen grafikler
└── README.md
```

---

## Kurulum

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-fuzzy scipy joblib
```

---

## Çalıştırma Sırası

```bash
# Aşama 1: Veri analizi + normalizasyon + train/test bölünmesi
python src/01_eda_desharnais.py

# Aşama 2: Üyelik fonksiyonları tasarımı
python src/02_fuzzy_mf_design.py

# Aşama 3: Kural tabanı analizi
python src/03_rule_base_analysis.py

# Aşama 4: Klasik Sugeno model eğitimi
python src/04_sugeno_klasik.py

# Aşama 7: GA optimizasyonu (MF + kural + katsayı optimize)
python src/07_ga_optimizer.py

# Aşama 8: Klasik vs GA-Sugeno karşılaştırması
python src/08_karsilastirma.py

# Aşama 9: Operatör bazlı yorumlama (kritik %25)
python src/09_yorum_analizi.py
```

---

## GA Tasarımı

### Chromosome Yapısı (118 eleman)
| Bölge | Eleman | Açıklama |
|-------|--------|----------|
| MF parametreleri | 18 float | 3 girdi × 3 MF × (center, sigma) |
| Rule selection | 20 bit | Kural aktif/pasif (0/1) |
| Rule katsayıları | 80 float | 20 kural × 4 katsayı (w0,w1,w2,w3) |

### GA Parametreleri
| Parametre | Değer |
|-----------|-------|
| Populasyon | 50 birey |
| Nesil | 100 |
| Crossover | 0.80 (Single-Point) |
| Mutation | 0.05 (Gaussian + Bit-flip) |
| Selection | Tournament (k=3) |
| Elitizm | 2 birey |
| Fitness | -RMSE (minimize) |

---

## Değerlendirme Kriterleri
| Kriter | Ağırlık | Karşılanan Dosya |
|--------|---------|------------------|
| Model doğruluğu | 20% | 07, 08 |
| Fuzzy tasarım kalitesi | 20% | 02, 03 |
| Yorumlama (kritik) | 25% | 09 |
| Karşılaştırmalı analiz | 15% | 08 |
| Sunum | 20% | YouTube |
