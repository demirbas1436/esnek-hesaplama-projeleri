"""
Asama 1: Veri Analizi (EDA) - Albrecht Veri Seti
- Outlier analizi (IQR yontemi)
- Winsorize ile outlier temizleme
- Normalizasyon (Min-Max)
- Ozellik secimi
- Egitim/test bolunmesi
Calistirma: python src/01_eda_albrecht.py
"""
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ekransiz kaydet
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- Dizin Yapisi ---
BASE_DIR = pathlib.Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR  = BASE_DIR / "models"
OUTPUT_DIR  = BASE_DIR / "output"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================
# 1. VERİYİ YÜKLE
# ============================
df = pd.read_csv(DATASET_DIR / 'albrecht.csv')

print("=" * 50)
print("ALBRECHT ILK 5 SATIR")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("VERI SETI BILGISI")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("ISTATISTIKSEL OZET")
print("=" * 50)
print(df.describe())

# ============================
# 2. EKSİK VERİ ANALİZİ
# ============================
print("\n" + "=" * 50)
print("EKSIK VERI SAYISI")
print("=" * 50)
print(df.isnull().sum())

if df.isnull().sum().sum() > 0:
    print("\nEksik verili satirlar:")
    print(df[df.isnull().any(axis=1)])
    df_clean = df.dropna()
    print(f"\nEksik veriler cikarildi. Yeni boyut: {df_clean.shape}")
else:
    df_clean = df.copy()
    print("\nEksik veri yok.")

# ============================
# 3. KORELASYON ANALİZİ
# ============================
print("\n" + "=" * 50)
print("EFFORT ILE KORELASYONLAR (SIRALI)")
print("=" * 50)
corr_with_effort = df_clean.corr(numeric_only=True)['Effort'].sort_values(ascending=False)
print(corr_with_effort)

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(df_clean.corr(numeric_only=True), dtype=bool))
sns.heatmap(df_clean.corr(numeric_only=True),
            annot=True, fmt=".2f", cmap='RdYlBu_r',
            mask=mask, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title('Albrecht Veri Seti - Korelasyon Matrisi', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "albrecht_korelasyon_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("[OK] albrecht_korelasyon_heatmap.png kaydedildi.")

# ============================
# 4. OUTLIER ANALİZİ (Boxplot)
# ============================
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if 'id' in numeric_cols:
    numeric_cols.remove('id')

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()
for idx, col in enumerate(numeric_cols):
    if idx < len(axes):
        sns.boxplot(y=df_clean[col], ax=axes[idx], color='skyblue')
        axes[idx].set_title(f'{col}', fontsize=12)
        axes[idx].set_ylabel('')
for idx in range(len(numeric_cols), len(axes)):
    axes[idx].set_visible(False)
plt.suptitle('Albrecht Outlier Analizi - Boxplotlar', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "albrecht_outlier_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()
print("[OK] albrecht_outlier_boxplots.png kaydedildi.")

# ============================
# 5. IQR İLE OUTLIER TESPİTİ
# ============================
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

print("\n" + "=" * 50)
print("IQR OUTLIER TESPITI")
print("=" * 50)
for col in numeric_cols:
    outliers, lower, upper = detect_outliers_iqr(df_clean, col)
    print(f"{col}: {len(outliers)} outlier (Alt: {lower:.2f}, Ust: {upper:.2f})")

# ============================
# 6. WİNSORİZE (Outlier Temizleme)
# ============================
df_processed = df_clean.copy()
for col in numeric_cols:
    _, lower, upper = detect_outliers_iqr(df_processed, col)
    df_processed[col] = np.where(df_processed[col] < lower, lower, df_processed[col])
    df_processed[col] = np.where(df_processed[col] > upper, upper, df_processed[col])

print("\n" + "=" * 50)
print("WINSORIZE SONRASI ISTATISTIKLER")
print("=" * 50)
print(df_processed.describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_clean['Effort'], kde=True, ax=axes[0], color='salmon')
axes[0].set_title('Effort - Orijinal (Outlierlar Dahil)')
sns.histplot(df_processed['Effort'], kde=True, ax=axes[1], color='mediumseagreen')
axes[1].set_title('Effort - Winsorize Sonrasi')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "albrecht_effort_karsilastirma.png", dpi=150, bbox_inches='tight')
plt.close()
print("[OK] albrecht_effort_karsilastirma.png kaydedildi.")

# ============================
# 7. ÖZELLİK SEÇİMİ
# ============================
# Fuzzy model icin 3 girdi degiskeni: Input, Output, File
# (Islev noktasi analizinin temel bilesenleri, yuksek korelasyonlu)
feature_cols = ['Input', 'Output', 'Inquiry', 'File', 'FPAdj', 'RawFPcounts', 'AdjFP']
target_col = 'Effort'

X = df_processed[feature_cols]
y = df_processed[target_col]

print("\n" + "=" * 50)
print("SECILEN OZELLIKLER")
print("=" * 50)
print("Girdiler:", feature_cols)
print("Hedef:", target_col)
print(f"Toplam ornek: {len(X)}")

# ============================
# 8. TRAIN / TEST SPLIT
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nEgitim seti: {len(X_train)} ornek")
print(f"Test seti: {len(X_test)} ornek")

# ============================
# 9. NORMALİZASYON (Min-Max)
# ============================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=feature_cols, index=X_test.index)

print("\n" + "=" * 50)
print("NORMALIZE EDILMIS VERI (ILK 5 SATIR)")
print("=" * 50)
print(X_train_scaled.head())

# ============================
# 10. KAYDET
# ============================
train_df = X_train_scaled.copy()
train_df['Effort'] = y_train_scaled
test_df  = X_test_scaled.copy()
test_df['Effort']  = y_test_scaled

train_df.to_csv(MODELS_DIR / "albrecht_train_clean.csv", index=False)
test_df.to_csv(MODELS_DIR  / "albrecht_test_clean.csv",  index=False)
joblib.dump(scaler_X, MODELS_DIR / "albrecht_scaler_X.pkl")
joblib.dump(scaler_y, MODELS_DIR / "albrecht_scaler_y.pkl")

print("\n" + "=" * 50)
print("KAYIT TAMAMLANDI")
print("=" * 50)
print(f"  -> {MODELS_DIR / 'albrecht_train_clean.csv'}")
print(f"  -> {MODELS_DIR / 'albrecht_test_clean.csv'}")
print(f"  -> {MODELS_DIR / 'albrecht_scaler_X.pkl'} / albrecht_scaler_y.pkl")
print(f"  -> {OUTPUT_DIR / 'albrecht_korelasyon_heatmap.png'}")
