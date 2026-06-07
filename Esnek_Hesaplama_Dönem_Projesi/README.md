# Sugeno FIS ile Yazılım Efor Tahmini

**Esnek Hesaplama Dönem Projesi**

Bu proje, Desharnais ve Albrecht veri setleri üzerinde **Sugeno tipinde Bulanik Çıkarım Sistemi (FIS)** oluşturup, yazılım efor tahmini yapar. Ayrıca klasik makine öğrenmesi modelleri ve **K‑Means kümeleme temelli regresör** ile karşılaştırmalar sunar.

---

## Klasör Yapısı

```text
Esnek_Hesaplama_Dönem_Projesi/
|-- src/                        # Python kaynak kodları
|   |-- 01_eda_desharnais.py    # Aşama 1: Veri analizi (IQR box‑plot)
|   |-- 01_eda_albrecht.py     # Aşama 1: Veri analizi (IQR box‑plot)
|   |-- 02_fuzzy_mf_design.py   # Aşama 2: Üyelik fonksiyonları tasarımı
|   |-- 03_rule_base_analysis.py # Aşama 3: Kural seti analizi (4 LLM)
|   |-- 04_sugeno_models.py     # Aşama 4: Sugeno model eğitimi
|   |-- 05_performance_comparison.py # Aşama 5: Performans karşılaştırması (LR, DT, KMeansRegressor, Sugeno)
|   |-- 06_interpretation_analysis.py # Aşama 6: Açıklanabilirlik ve senaryo analizi (IQR‑temelli gerçek değerler)
|-- output/                     # Üretilen grafikler ve CSV dosyaları
|-- models/                     # Eğitilmiş model dosyaları
|   |-- scaler_X.pkl, scaler_y.pkl
|   |-- mf_parameters.json
|   |-- rule_sets.json
|   |-- coefficients_*.npy
|-- dataset/                    # Ham veri setleri
|-- README.md                   # Bu dosya
|-- presentation.txt            # Sunum metni (bu dosya)
```

---

## Kurulum

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-fuzzy joblib scipy
```

---

## Çalıştırma Sırası

Projeyi kök dizinden aşağıdaki adımları sırayla çalıştırın:

```bash
# Aşama 1 – Veri analizi ve IQR‑box‑plot
python src/01_eda_desharnais.py
python src/01_eda_albrecht.py

# Aşama 2 – Üyelik fonksiyonları tasarımı
python src/02_fuzzy_mf_design.py

# Aşama 3 – Kural seti analizi (4 LLM)
python src/03_rule_base_analysis.py

# Aşama 4 – Sugeno model eğitimi (Least Squares & Gradient Descent)
python src/04_sugeno_models.py

# Aşama 5 – Performans karşılaştırması
#   - Linear Regression, Decision Tree ve yeni KMeansRegressor (klasik basamak) 
#   - Sugeno modelleri (Model A & Model B)
python src/05_performance_comparison.py

# Aşama 6 – Açıklanabilirlik & Senaryo Analizi
#   • Gerçek (inverse‑scaled) değerler
#   • IQR‑temelli "Düşük / Orta / Yüksek" sınıflandırması
python src/06_interpretation_analysis.py
```

---

## Yeni Eklemeler

### K‑Means Regresör (Baseline)
* `src/05_performance_comparison.py` dosyasına **KMeansRegressor** sınıfı eklendi.
* K‑Means kümeleme, girdileri *n_clusters* (varsayılan 4) gruplar ve her küme için ortalama `Effort` değeri hesaplanır. Tahmin, yeni örnek ilgili kümeye atanıp o kümenin ortalama eforu döndürür.
* Bu basit model, klasik Linear Regression ve Decision Tree modelleriyle aynı tablo ve grafiklerde gösterilir, böylece **“Box Plot veya Kümeleme algoritmaları”** önerisine doğrudan yanıt verilir.

### IQR‑Temelli Kategorilendirme
* `src/06_interpretation_analysis.py` içinde `get_iqr_category(value, series)` fonksiyonu tanımlandı.
* Eğitim verisinin **Q1 (%25)** ve **Q3 (%75)** değerleri kullanılarak her özelliğin gerçek (ters ölçeklenmiş) değeri **Düşük / Orta / Yüksek** olarak etiketlenir.
* Sunumda, bu etiketlerin kutu grafiğindeki çeyrek sınırlarıyla bire bir eşleştiği vurgulanır.

---

## Proje Özeti

### Kullanılan Veri Setleri
1. **Desharnais** –  proje, özellikler: `PointsAjust`, `Length`, `TeamExp`; hedef `Effort`.
2. **Albrecht** – 24 proje, özellikler: `Input`, `Output`, `File`; hedef `Effort`.

### Sugeno Modeli
* Üyelik fonksiyonları (Gaussian, Triangular) üç dilsel değer (*Low, Medium, High*) üretir.
* 4 LLM (Student, ChatGPT, Claude, Gemini) için **her birinde 20 kural** oluşturuldu – toplam 80 kural.
* Model A: En Küçük Kareler (analitik çözüm).
* Model B: Gradient Descent + L2 regularizasyon (iteratif, daha iyi performans).

### Neden Sugeno?
* **Açıklanabilirlik** – Her kural doğal dilde (`IF … THEN …`) ifade edilir.
* **Bulunabilir Üyelik Fonksiyonları** – Fuzzy C‑Means kümeleme ile otomatik olarak merkezler belirlenir, subjektif tercih azaltılır.
* **Parçalı Lineer Çıktı** – Sonuç lineer kombinasyon, analiz ve yorumlama kolaydır.

### Neden IQR (Box Plot)?
* Outlier (uç değer) tespiti için **IQR** en güvenilir istatistiklerden biridir.
* **Winsorization** ile uç değerler sınırlandırılır, modelin aşırı etkilenmesi önlenir.
* Box‑plot görselleştirmesi, veri dağılımını hızlıca göstermek ve **Düşük/Orta/Yüksek** dilsel sınıflandırma temeli sunmak için ideal.

---

## Sonuçlar (output dizinindeki CSV dosyalarından)
* `*_performance_results.csv` – RMSE, MAE, MAPE, R2, %25‑25 hata aralığı (Accuracy) değerleri.
* **K‑MeansRegressor** ortalama RMSE/MAE değerleri, klasik modellerle kıyaslandığında makul bir baseline sağlar.
* **Sugeno Model B** genellikle en yüksek doğruluk ve açıklanabilirlik dengesi sunar.

---

## Sunum İçin Hazırlanan Metin (`presentation.txt`)
* Bu dosya aynı klasörde bulunur ve `cat presentation.txt` ya da bir metin editöründe açılarak doğrudan slayt açıklaması olarak kullanılabilir.

---

## Katkıda Bulunma
* Sorularınız ve önerileriniz için `issues` bölümünü kullanabilirsiniz.

---

*Bu README, projenin tüm adımlarını, yeni eklemeleri ve akademik arka planı kapsar. Sunum sırasında her bölümü sırayla anlatarak hoca ve dinleyicilerin sorularına hazır olabilirsiniz.*