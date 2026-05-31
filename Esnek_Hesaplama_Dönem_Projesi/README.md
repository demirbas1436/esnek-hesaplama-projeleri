# Sugeno FIS ile Yazilim Efor Tahmini
**Esnek Hesaplama Donem Projesi**

Desharnais yazilim projesi veri seti uzerinde Sugeno tipi Bulanik Cikarsama Sistemi (FIS) 
kurarak yazilim efor tahmini yapan proje.

---

## Klasor Yapisi

```
Esnek_Hesaplama_Donem_Projesi/
|-- src/                        Python kaynak kodlari
|   |-- 01_eda_desharnais.py    Asama 1: Veri Analizi
|   |-- 02_fuzzy_mf_design.py   Asama 2: Uyelik Fonksiyonu Tasarimi
|   |-- 03_rule_base_analysis.py Asama 3: Kural Tabani (4 LLM)
|   |-- 04_sugeno_models.py     Asama 4: Model Egitimi
|   |-- 05_performance_comparison.py  Asama 5: Performans Analizi
|   |-- 06_interpretation_analysis.py Asama 6: Yorumlama
|-- output/                     Uretilen grafikler ve CSV dosyalari
|   |-- korelasyon_heatmap.png
|   |-- outlier_boxplots.png
|   |-- membership_functions_input.png
|   |-- membership_functions_output.png
|   |-- rule_coverage_all_models.png
|   |-- sugeno_model_analysis.png
|   |-- performance_comparison.png
|   |-- interpretation_analysis.png
|   |-- performance_results.csv
|   |-- cv_results.csv
|-- models/                     Egitilmis model dosyalari
|   |-- scaler_X.pkl / scaler_y.pkl
|   |-- mf_parameters.json
|   |-- rule_sets.json          4 LLM kural seti
|   |-- coefficients_*.npy      Ogrenilen Sugeno katsayilari
|   |-- desharnais_train_clean.csv
|   |-- desharnais_test_clean.csv
|-- dataset/
|   |-- desharnais.csv          Ham veri seti
|-- Esnek Hesaplama Donem Projesi.docx  Odev tanimi
|-- Sugeno_FIS_Yazilim_Efor_Tahmini_Rapor.docx  Proje raporu
|-- README.md
```

---

## Kurulum

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-fuzzy joblib scipy
```

---

## Calistirma Sirasi

Asagilaki komutlari **proje kok dizininden** sirasiyla calistirin:

```bash
# Asama 1: Veri analizi, normalizasyon, egitim/test bolunmesi
python src/01_eda_desharnais.py

# Asama 2: Uyelik fonksiyonlari tasarimi (giris + cikis MF grafikleri)
python src/02_fuzzy_mf_design.py

# Asama 3: 4 LLM kural seti analizi (Student, ChatGPT, Claude, Gemini)
python src/03_rule_base_analysis.py

# Asama 4: Sugeno model egitimi (LeastSquares + GradientDescent)
python src/04_sugeno_models.py

# Asama 5: Performans karsilastirmasi (Sugeno vs LR vs DT)
python src/05_performance_comparison.py

# Asama 6: Yorumlama ve aciklanabilirlik analizi
python src/06_interpretation_analysis.py
```

---

## Proje Ozeti

### Kullanilan Veri Seti
- **Desharnais Dataset**: 77 yazilim projesi, 11 ozellik
- **Giris Degiskenleri**: PointsAjust (Fonksiyon Noktasi), Length (Sure), TeamExp (Deneyim)
- **Hedef Degisken**: Effort (adam-saat)

### Fuzzy Model Yapisi
- **Uyelik Fonksiyonlari**: Gaussian, Triangular, Trapezoidal (3 MF/degisken)
- **Linguistik Degerler**: Low (Dusuk), Medium (Orta), High (Yuksek)
- **Kural Sayisi**: Her LLM icin 20 kural

### Kural Setleri (LLM Karsilastirmasi)
| LLM     | Kural Sayisi | Odak Degiskeni         |
|---------|-------------|------------------------|
| Student | 20          | PointsAjust agirlikli  |
| ChatGPT | 20          | Dengeli p1/p2          |
| Claude  | 20          | Denge odakli           |
| Gemini  | 20          | Length odakli          |

### Ogrenim Yontemleri
- **Model A**: En Kucuk Kareler (Least Squares) - analitik
- **Model B**: Gradient Descent + L2 Regularizasyon - iteratif

### Karsilastirma Modelleri
- Linear Regression
- Decision Tree Regressor (max_depth=5)

---

## Sonuclar

`output/performance_results.csv` dosyasinda tum modellerin RMSE, MAE, MAPE, R2 sonuclari bulunmaktadir.

**Temel Bulgular:**
- Model B (Gradient Descent) her kural seti icin Model A'dan daha iyi test performansi gosterir
- Klasik modeller (LR, DT) Sugeno modellerine gore daha yuksek R2 saglar
- Sugeno modellerinin avantaji: aciklanabilirlik ve kural bazli karar mekanizmasi

---