===================================================================
ESNEK HESAPLAMA DÖNEM PROJESİ — TAM SUNUM METNİ
(Doğal dil, insan gibi konuşma tarzında yazılmıştır)
===================================================================

-------------------------------------------------------------------
BÖLÜM 1 — PROJEYİ TANITIRKEN NE SÖYLERSIN?
-------------------------------------------------------------------

"Hocam, bu projede yazılım projelerinin ne kadar adam-saat efor
gerektireceğini tahmin etmeye çalıştık. Bunun için klasik
istatistik yöntemleri yerine esnek hesaplama tekniklerinden olan
Sugeno tipi Bulanık Çıkarım Sistemi'ni (FIS) kullandık.

Elimizde iki farklı veri seti vardı:

- Desharnais: 77 yazılım projesi; her projenin fonksiyon noktası
  (PointsAjust), süresi (Length), takım deneyimi (TeamExp) ve
  gerçek eforu (adam-saat) kayıtlı.

- Albrecht: 24 yazılım projesi; giriş sayısı (Input), çıkış sayısı
  (Output), mantıksal dosya sayısı (File) ve gerçek eforu kayıtlı.

Her iki veri seti için ayrı ayrı modeller kuruldu, eğitildi ve
sonuçlar karşılaştırıldı."

-------------------------------------------------------------------
BÖLÜM 2 — VERİ ÖN İŞLEME (AŞAMA 1) — IQR VE BOX PLOT
-------------------------------------------------------------------

"İlk olarak verileri temizledik. Burada iki önemli kavram var:

BOX PLOT NEDİR?
Kutu grafik, verimizin nasıl dağıldığını gösteren bir görsel
araçtır. Kutunun alt kenarı Q1 (alt çeyrek, %25), üst kenarı Q3
(üst çeyrek, %75), ortasındaki çizgi ise medyan (%50)'dır.
Kutunun dışında kalan noktalar ise aykırı değerlerdir (outlier).

IQR NEDİR?
IQR = Q3 - Q1. Yani verinin ortadaki %50'lik diliminin genişliği.
Aykırı değer sınırları: alt = Q1 - 1.5*IQR, üst = Q3 + 1.5*IQR.

BİZ NASIL KULLANDIK?
Eğitim verisindeki her özellik için box plot çizip aykırı
değerleri tespit ettik. Sonra Winsorization yöntemiyle bu aykırı
değerleri sınırlandırdık; yani silmedik, sınıra çektik.

NEDEN BU ÖNEMLİ?
Çünkü örneğin bir proje 10.000 adam-saat gerektiriyorsa ve
diğerleri 100-500 arasındaysa, o tek proje modelin katsayılarını
tamamen çarpıtırdı. IQR ile bunu önledik.

IQR'NIN DİĞER KULLANIMI — DİLSEL KATEGORİLER:
Box plot'tan elde ettiğimiz Q1 ve Q3 değerleri, bir özelliğin
'Düşük mü, Orta mı, Yüksek mi' olduğunu da belirliyor.
  - value <= Q1  →  'Düşük'
  - Q1 < value < Q3  →  'Orta'
  - value >= Q3  →  'Yüksek'

Bu tam olarak derste anlatılan 'yaşa göre genç-orta-yaşlı' örneğiyle
aynı mantık. Biz de yazılım projelerini boyutuna, süresine ve
deneyimine göre Düşük-Orta-Yüksek olarak etiketledik."

-------------------------------------------------------------------
BÖLÜM 3 — ÜYELİK FONKSİYONLARI (AŞAMA 2) — 3 MF / DEĞİŞKEN
-------------------------------------------------------------------

"Sugeno modelinde her giriş değişkeni için üyelik fonksiyonları
(Membership Function, MF) tanımladık. Ödevde en az 3 MF/değişken
isteniyor ve biz bunu yaptık.

ÜYELİK FONKSİYONU NE DEMEK?
Klasik mantıkta bir değer ya 'Yüksek'tir ya da değildir. Bulanık
mantıkta ise bir değer 0 ile 1 arasında bir 'üyelik derecesi' ile
birden fazla kategoriye aynı anda ait olabilir.

Örnek: TeamExp = 3 yıl diyelim. Bu değer;
  - 'Düşük' kategorisine %20 üye
  - 'Orta' kategorisine %80 üye
  - 'Yüksek' kategorisine %0 üye
olabilir. Bu 'bulanıklık' gerçek hayatı daha iyi yansıtır.

BİZ KAÇAR MF KULLANDIK?
Her değişken için 3 MF: Düşük (Low), Orta (Medium), Yüksek (High).
Toplam 3 değişken × 3 MF = 9 üyelik fonksiyonu / veri seti.

KULLANDIĞIMIZ MF TİPLERİ:
1. Gaussian (Gauss Eğrisi): Merkezde 1, uzaklaştıkça sıfıra
   yaklaşır. Merkezi FCM kümeleme ile otomatik bulduk.
2. Triangular (Üçgen): Q1, medyan, Q3 ile tanımlandı. Box plot'tan
   doğrudan çıkarıldı.
3. Trapezoidal (Yamuk): Uç değerler için düz bir üst bölge sağlar.

ÇIKTI (OUTPUT) İÇİN DE MF ÇİZİLDİ:
Sadece giriş değil, çıkış (Effort) da görselleştirildi. Bu da
ödevde istenen 'hem MF'ler hem output çizilerek gösterilmeli'
maddesini karşılıyor.

FCM KÜMELEME NEDİR?
Fuzzy C-Means, Gaussian MF'lerin merkezlerini bulmak için
kullandığımız kümeleme algoritması. Veriyi 3 kümeye ayırıp
her kümenin merkezini bulur; bu merkez Gaussian'ın zirvesi olur.
Böylece MF parametrelerini elle değil, veriden otomatik öğrendik.
Bu tamamen veri güdümlü, subjektif değil."

-------------------------------------------------------------------
BÖLÜM 4 — KURAL SETLERİ (AŞAMA 3) — 4 LLM KARŞILAŞTIRMASI
-------------------------------------------------------------------

"Sugeno modelinde IF-THEN kuralları var. Bu kuralları biz hem
kendimiz (öğrenci) hem de 3 farklı yapay zeka (ChatGPT, Claude,
Gemini) ile oluşturduk. Her biri için 20'şer kural yazdık.

KURAL NE DEMEK?
Örnek: 'IF PointsAjust=Low AND Length=Low AND TeamExp=High
         THEN Effort = p1*PointsAjust + p2*Length + p3*TeamExp + c'

Yani giriş değişkenleri belirli kategorilerdeyse, çıkış belirli
bir lineer formülle hesaplanır. Bu 'First-Order Sugeno' demek.

NEDEN LLM KULLANDIK?
LLM'ler (büyük dil modelleri), yazılım mühendisliği literatürünü
bilerek mantıklı kurallar önerdi. Biz bu kuralları karşılaştırdık.

KURAL SEÇİMİNDE NE BAKTIK?
- Hangi kurallar çakışıyor (çelişkili kural var mı)?
- Hangi kurallar hiç ateşlenmiyor (veriyle örtüşmüyor mu)?
- Hangi LLM'in kuralları gerçek veriyi daha iyi kapsıyor?

Bu analiz src/03_rule_base_analysis.py'de yapıldı."

-------------------------------------------------------------------
BÖLÜM 5 — SUGENO MODEL EĞİTİMİ (AŞAMA 4) — 2 MODEL
-------------------------------------------------------------------

"Ödevde 2 model kurulacak deniyordu. Biz tam olarak bunu yaptık:

MODEL A — EN KÜÇÜK KARELER (LEAST SQUARES):
Her kural için consequent (sonuç) katsayıları p1, p2, p3, c'yi
analitik olarak (matris çarpımıyla) tek seferde hesaplar.
W^T * W * theta = W^T * y denklemini çözüyoruz.
Hızlı ama veri boyutu küçükse overfitting riski var.

MODEL B — GRADIENT DESCENT + L2 REGULARIZASYON:
Katsayıları iteratif olarak günceller. Her adımda hatayı azaltmak
için katsayıları biraz değiştirir. L2 cezası ise katsayıların çok
büyümesini engeller. Daha yavaş ama genellikle daha iyi sonuç.

SUGENO'DA AĞIRLIKLAR NASIL ÖĞRENİLİR?
(Hoca bu soruyu sorabilir!)

Şu şekilde çalışır:
1. Giriş değerleri üyelik fonksiyonlarına sokulur, her kural için
   bir 'ateşlenme gücü' (firing level) hesaplanır: w_i.
2. Firing level'lar normalize edilir: w_i / toplam(w_i).
3. Her kural kendi lineer formülüyle bir z_i değeri üretir.
4. Çıktı: y = toplam(w_i * z_i) — ağırlıklı ortalama.
5. Model A bu z_i'nin katsayılarını analitik çözer.
   Model B ise gradyan inerek adım adım bulur.

KATSAYILARIN İŞARETİ NE ANLAMA GELİR?
Desharnais için:
  - p1 (PointsAjust) > 0: Proje büyüdükçe efor artar. Mantıklı.
  - p2 (Length) > 0: Süre uzadıkça efor artar. Mantıklı.
  - p3 (TeamExp) < 0: Deneyim arttıkça efor AZALIR. Mantıklı!
    Çünkü deneyimli ekip daha hızlı çalışır.

Albrecht için:
  - p1 (Input) > 0, p2 (Output) > 0, p3 (File) > 0: Hepsi
    yazılım boyutunu temsil eder, hepsi eforu artırır."

-------------------------------------------------------------------
BÖLÜM 6 — KÜMELEMENİN BASELINE MODEL OLARAK KULLANIMI
-------------------------------------------------------------------

"Hoca 'kümeleme algoritmaları da kullanılabilir' demişti. Biz bunu
iki farklı şekilde yaptık:

1. FUZZY C-MEANS (FCM): Üyelik fonksiyonu merkezlerini bulmak için
   veriyi 3 kümeye ayırdık. Bu kümelerin merkezleri Gaussian MF'in
   zirvesi oldu. Dolayısıyla FCM, modelin içinde zaten var.

2. K-MEANS REGRESSOR (Baseline): Bunu ayrıca karşılaştırma modeli
   olarak ekledik. Çalışması şöyle:
   - Eğitim verisini 4 kümeye ayır (K-Means).
   - Her kümenin ortalama Effort değerini kaydet.
   - Test aşamasında: yeni proje en yakın kümeye atanır, o kümenin
     ortalama eforu tahmin olarak döndürülür.
   - Sonuçlar Sugeno, LR ve DT ile karşılaştırıldı."

-------------------------------------------------------------------
BÖLÜM 7 — PERFORMANS ANALİZİ (AŞAMA 5) — KARŞILAŞTIRMA
-------------------------------------------------------------------

"Tüm modelleri aynı test verisi üzerinde değerlendirdik:

KULLANDIĞIMIZ METRİKLER:
- RMSE (Root Mean Square Error): Tahmin hatalarının ortalama büyüklüğü.
- MAE (Mean Absolute Error): Ortalama mutlak hata.
- MAPE (%): Yüzde cinsinden ortalama hata.
- R²: Modelin varyansı ne kadar açıkladığı (1.0 mükemmel, 0 kötü).
- Accuracy (%25 bant): Tahmin gerçekten %25 içinde kalıyor mu?

Ek olarak:
- 5-Fold Cross Validation: Klasik modeller için aşırı öğrenmeyi
  test etmek amacıyla.
- Wilcoxon Signed-Rank Test: Sugeno ile LR arasındaki farkın
  istatistiksel olarak anlamlı olup olmadığını test eder.

NEDEN KLASİK MODELLER DE VAR?
Ödev bir karşılaştırma istiyor. LR ve DT 'baseline' olarak duruyor;
Sugeno modelinin onlara göre ne kadar iyi veya açıklanabilir
olduğunu gösteriyoruz. Klasik modeller bazen daha iyi R² verse de
kural-bazlı açıklama yapamıyorlar."

-------------------------------------------------------------------
BÖLÜM 8 — AÇIKLANABILIRLIK RAPORU (AŞAMA 6) — NEDEN BU KARAR?
-------------------------------------------------------------------

"Ödevde 'sistem neden böyle karar veriyor?' sorusuna cevap vermemiz
gerekiyor. Bunu Aşama 6'da yaptık.

SENARYO ANALİZİ:
'Küçük Proje' ve 'Büyük Proje' olmak üzere iki senaryo oluşturduk.
Her senaryo için:

1. GERÇEKLEŞTİRME (FUZZIFICATION):
   Normalize edilmiş (0-1) girdi değerleri önce gerçek fiziksel
   değerlerine çevrildi (inverse transform). Örneğin:
     - PointsAjust = 150, Length = 12 ay, TeamExp = 4 yıl

2. DİLSEL KATEGORİ (IQR-temelli):
   Her gerçek değer, eğitim verisinin Q1-Q3'ü ile karşılaştırılarak
   Düşük/Orta/Yüksek etiketi aldı. Örnek:
     - PointsAjust=150 → Düşük, Length=12 → Düşük, TeamExp=4 → Yüksek

3. HAZIRLAMA (FIRING LEVELS):
   Her kural için ateşlenme gücü hesaplandı. Hangi kural en çok
   ateşlendi? O baskın kural!

4. TAHMIN:
   Baskın kural ve diğer ateşlenen kuralların ağırlıklı ortalaması
   alınarak nihai efor tahmini üretildi.

5. AÇIKLAMA:
   'Bu proje, Low-Low-High (küçük boyut, kısa süre, deneyimli ekip)
   profiline uyuyor. R7 kuralı %42 ateşlenme ile baskın. Tahmin:
   128 adam-saat.'

Bu tam olarak dersteki 'yaşa göre genç/orta/yaşlı → çıkarım' örneği.
Biz de 'proje boyutuna/süresine/deneyimine göre Low/Medium/High →
efor tahmini' yaptık."

-------------------------------------------------------------------
BÖLÜM 9 — HOCA SORULARINA HAZIR CEVAPLAR
-------------------------------------------------------------------

S: "Neden Sugeno seçtiniz, Mamdani de vardı?"
C: "Sugeno'nun consequent kısmı lineer fonksiyon, bu yüzden
   katsayıları matematiksel olarak öğrenebiliyoruz (LS veya GD ile).
   Mamdani'de çıktı da bulanık olur, defuzzification gerekir, bu
   hesaplama yükünü artırır. Ayrıca Sugeno, regresyon modellerine
   daha yakın olduğu için karşılaştırma yapmak daha kolay."

S: "First-order Sugeno ne demek?"
C: "Sugeno'da consequent kısmı bir sabit ise zeroth-order, lineer
   fonksiyon ise first-order denir. Biz first-order kullandık:
   z_i = p1*x1 + p2*x2 + p3*x3 + c. Bu sayede her kural kendi
   lineer modelini öğreniyor; model daha ifade gücüne sahip oluyor."

S: "MF parametrelerini nasıl belirlediniz?"
C: "Üç yöntem birlikte kullandık:
   1. Triangular MF: Box plot'tan Q1, medyan, Q3 alındı.
   2. Gaussian MF: FCM (Fuzzy C-Means) kümeleme ile merkez bulundu,
      standart sapma veriden hesaplandı.
   3. Trapezoidal MF: Uç bölgeler için veri aralığına göre ayarlandı.
   Hiçbiri subjektif değil, hepsi veriden otomatik türetildi."

S: "Baskın kural nasıl belirleniyor?"
C: "Her kural için, antecedent (öncül) kısımdaki üyelik değerleri
   çarpılır. Bu çarpım o kuralın 'ateşlenme gücü'dür (w_i). En büyük
   w_i değerine sahip kural baskın kuraldır. Sistemin kararını en çok
   o kural etkiliyor demektir."

S: "IQR'yi neden sınıflandırma için de kullandınız?"
C: "Box plot zaten verideki doğal kırılma noktalarını gösteriyor.
   Q1 ve Q3, verinin alt %25 ve üst %25'ini ayırıyor. Bu değerleri
   kullanarak 'Düşük/Orta/Yüksek' sınırını belirlemek hem veri-güdümlü
   hem de görsel olarak açıklanabilir. Hoca da box plot veya kümeleme
   kullanılabileceğini belirtmişti, biz ikisini de kullandık."

S: "K-Means regressor neden eklendi?"
C: "Hocanın 'kümeleme algoritmaları kullanılabilir' önerisine doğrudan
   yanıt vermek için. Ayrıca fuzzy sistemin sadece üyelik fonksiyonu
   tasarımı için değil, tahmin için de kümeleme mantığı kullanılabileceğini
   göstermek istedik. K-Means Regressor, karşılaştırma tablolarında
   yerini alıyor."

S: "Model A mı daha iyi, Model B mi?"
C: "Genellikle Model B (Gradient Descent + L2 regularizasyon) daha iyi
   test performansı veriyor. Bunun sebebi L2 cezasının katsayıların
   aşırı büyümesini engellemesi. Model A analitik ve hızlı, ama küçük
   veri setlerinde overfitting yapabilir."

S: "Neden iki veri seti kullandınız?"
C: "Modelin genellenebilirliğini test etmek için. Desharnais 77 büyük
   boyutlu proje, Albrecht 24 küçük boyutlu proje içeriyor. Farklı
   özellik yapıları (deneyim odaklı vs boyut odaklı) üzerinde modelin
   nasıl davrandığı karşılaştırıldı."

S: "Wilcoxon testi ne için kullandınız?"
C: "Sugeno Model B ile Linear Regression arasındaki RMSE farkının
   gerçekten anlamlı mı yoksa tesadüf mü olduğunu test etmek için.
   p < 0.05 ise fark istatistiksel olarak anlamlı diyoruz."

S: "Kodu kendiniz mi yazdınız?"
C: "Evet hocam. Tüm pipeline adımları 6 Python dosyasında organize
   edildi: EDA, MF tasarımı, kural seti analizi, model eğitimi,
   performans karşılaştırması ve açıklanabilirlik raporu. Her dosya
   bağımsız çalışabilir ve kendi çıktısını üretiyor."

-------------------------------------------------------------------
BÖLÜM 10 — PROJEYİ KAPATIRKEN NE SÖYLERSİN?
-------------------------------------------------------------------

"Sonuç olarak hocam, bu projede:

1. İki farklı veri seti üzerinde Sugeno FIS uyguladık.
2. Üyelik fonksiyonlarını hem Box-Plot hem de FCM kümeleme ile
   veri-güdümlü olarak belirledik.
3. 4 farklı kural seti (öğrenci + 3 LLM) oluşturduk ve bunların
   model performansına etkisini analiz ettik.
4. İki öğrenme yöntemi (Least Squares ve Gradient Descent) ile
   katsayıları öğrendik.
5. Sistemin neden bu kararı verdiğini IQR-tabanlı dilsel kategoriler
   ve baskın kural analizi ile açıkladık.
6. K-Means kümeleme de hem MF tasarımında (FCM) hem de baseline
   regresör olarak kullanıldı.

Model sadece tahmin yapmıyor; hangi kuralın, neden devreye girdiğini
ve giriş değerlerinin dilsel karşılığını da açıklıyor. Bu,
açıklanabilir yapay zeka perspektifinden önemli bir avantaj."

===================================================================
ÇALIŞTIRMA SIRASI (SUNUMDA CANLI GÖSTERİM İÇİN)
===================================================================
python src/01_eda_desharnais.py   → EDA + box plot grafikler
python src/01_eda_albrecht.py     → Albrecht EDA
python src/02_fuzzy_mf_design.py  → MF grafikler üretilir
python src/03_rule_base_analysis.py → Kural kapsamı analizi
python src/04_sugeno_models.py    → Model A ve B eğitimi
python src/05_performance_comparison.py → Tablo + grafikler
python src/06_interpretation_analysis.py → Senaryo açıklamaları
===================================================================
