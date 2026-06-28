# Ringkasan Analisis & Panduan Best Practice Implementasi Random Forest — Urban Pulse

Dokumen ini berisi analisis mendalam terhadap codebase **Urban Pulse**, penjelasan teoritis dan praktis pemilihan algoritma **Random Forest (RF)**, serta panduan langkah demi langkah implementasi best practice-nya pada file [randomforest.ipynb](file:///d:/urbanpulse/randomforest.ipynb).

---

## 1. Analisis Struktur Proyek & Dataset saat Ini

Berdasarkan analisis file preprocessing dan modeling XGBoost (`preprocessing.ipynb` dan `modeling.ipynb`), berikut karakteristik dataset Urban Pulse:
*   **Dataset Tabular Spasial**: Terdiri dari 830 baris (unit kelurahan/desa) dari 4 kota di Indonesia (Ambon, DKI Jakarta, Kebumen, Samarinda).
*   **Fitur Model (41 Kolom)**: Terdiri dari data spektral/tekstur Sentinel-2, data SAR Sentinel-1, data bangunan OpenStreetMap (OSM) / Google Earth Engine (GEE), serta 7 fitur rekayasa interaksi.
*   **Target (`slum`)**: Klasifikasi biner (1 untuk permukiman kumuh, 0 untuk non-kumuh).
*   **Class Imbalance Tinggi**: Hanya 14.2% (118 unit) yang berlabel slum, sedangkan 85.8% (712 unit) berlabel non-slum.
*   **Spatial Autocorrelation**: Unit geografis yang dekat cenderung mirip. Oleh karena itu, skema evaluasi utama wajib menggunakan **Leave-One-City-Out (LOCO) Cross-Validation** (4 folds) agar estimasi performa model di kota baru bernilai jujur dan objektif.

---

## 2. Best Practice Pengaturan Model Random Forest

Untuk menghasilkan model Random Forest yang setara atau bahkan lebih stabil dibanding XGBoost pada dataset ini, kita harus merancang konfigurasi algoritma secara hati-hati:

### A. Penanganan Class Imbalance (`class_weight`)
Berbeda dengan XGBoost yang menggunakan `scale_pos_weight`, Random Forest dari Scikit-Learn menangani ketidakseimbangan kelas menggunakan parameter `class_weight`.
*   **Rekomendasi**: Gunakan `class_weight='balanced_subsample'`.
*   **Alasan**: Opsi `'balanced_subsample'` menghitung ulang bobot kelas berdasarkan sampel bootstrap yang diambil secara acak saat membangun setiap pohon (tree), bukan menghitung bobot secara global di awal. Ini memberikan variasi bobot yang lebih dinamis dan mencegah model mendominasi kelas mayoritas pada setiap tree individual.

### B. Mencegah Overfitting pada Dataset Kecil (Regularisasi)
Dengan dataset yang hanya memiliki 830 baris and 41 fitur, Random Forest yang dibiarkan tumbuh tanpa batas (`max_depth=None`) akan mengalami overfitting parah (memorizing noise).
*   `n_estimators=300`: Jumlah pohon yang cukup untuk menghasilkan estimasi probabilitas yang stabil tanpa beban komputasi berlebih.
*   `max_depth=6`: Membatasi kedalaman pohon agar model belajar pola generalisasi tingkat tinggi (high-level features) daripada pola spesifik per baris.
*   `min_samples_leaf=3` atau `min_samples_split=8`: Memastikan setiap leaf node memiliki minimal 3 sampel data. Ini memperhalus batas keputusan (decision boundary) dan mengurangi varians prediksi.
*   `max_features='sqrt'`: Memilih $\sqrt{41} \approx 6$ fitur acak di setiap split. Ini penting untuk memperkecil korelasi antar pohon (tree decorrelation) sehingga ensemble bekerja optimal.

### C. Threshold Tuning (Prioritas Recall)
Sesuai dengan domain mapping permukiman kumuh (screening awal), model harus memprioritaskan **Recall tinggi (target $\ge 0.80$)** pada kelas slum. Melewatkan area kumuh (false negative) jauh lebih merugikan secara anggaran/kebijakan dibanding melakukan survei lapangan tambahan akibat false positive (precision rendah).
*   Oleh karena itu, kita tidak boleh menggunakan threshold default `0.5`. Kita harus mencari threshold optimal terkecil pada random split 80/20 yang masih menghasilkan recall $\ge 0.80$.

### D. Model Interpretability dengan SHAP
Scikit-learn `RandomForestClassifier` didukung penuh oleh `shap.TreeExplainer`. Namun, ada perbedaan output penting dibanding XGBoost:
*   Untuk model XGBoost biner, `TreeExplainer.shap_values(X)` menghasilkan array berdimensi 2 (N_samples, N_features).
*   Untuk Scikit-Learn Random Forest, `shap_values` berupa list berukuran 2, di mana elemen ke-0 berisi SHAP values untuk kelas 0 (non-slum) dan elemen ke-1 untuk kelas 1 (slum).
*   **Best Practice**: Selalu gunakan pengecekan tipe data `isinstance(shap_values, list)` untuk mengekstrak array kelas slum (`shap_values[-1]`) agar kode kompatibel dan terhindar dari crash error index.

---

## 3. Draf Struktur Kode Implementasi (`randomforest.ipynb`)

Berikut draf struktur sel-sel kode yang direkomendasikan untuk dituliskan pada notebook [randomforest.ipynb](file:///d:/urbanpulse/randomforest.ipynb) agar konsisten dengan pipeline evaluasi proyek:

### Cell 1: Import Library
```python
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.ensemble import RandomForestClassifier
import shap

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
RANDOM_STATE = 42
```

### Cell 2: Konfigurasi Path & Target
```python
PROCESSED_DIR = Path("data/processed")
DATA_PATH = PROCESSED_DIR / "features_buildings_clean.csv"
FEATURE_COLS_PATH = PROCESSED_DIR / "feature_columns.csv"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "slum_rf_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_rf_metadata.json"  # Dipisah agar tidak menimpa XGBoost

TARGET_COL = "slum"
CITY_COL = "city"
TEST_SIZE = 0.2
TARGET_RECALL = 0.80  # Target minimal recall slum
```

### Cell 3: Load Dataset
```python
df = pd.read_csv(DATA_PATH)
feature_cols = pd.read_csv(FEATURE_COLS_PATH)["feature_name"].tolist()

print(f"Dataset: {df.shape}")
print(f"Jumlah fitur model: {len(feature_cols)}")
print(f"Distribusi kelas: slum={int(df[TARGET_COL].sum())} "
      f"({df[TARGET_COL].mean()*100:.1f}%) / "
      f"non-slum={int((df[TARGET_COL]==0).sum())}")
```

### Cell 4: Definisi Model Instantiation
```python
def make_rf(random_state=RANDOM_STATE):
    # Menggunakan class_weight='balanced_subsample' untuk imbalance handling secara dinamis
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
```

### Cell 5: Evaluasi LOCO dengan Threshold Default (0.5)
```python
def run_loco_evaluation(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    cities = sorted(df[CITY_COL].unique())
    rows = []

    for city in cities:
        train_mask = df[CITY_COL] != city
        test_mask = df[CITY_COL] == city

        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, TARGET_COL]
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, TARGET_COL]

        model = make_rf()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = np.nan

        rows.append({
            "held_out_city": city,
            "n_test": len(y_test),
            "n_slum_test": int(y_test.sum()),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": auc,
        })

    return pd.DataFrame(rows)

loco_results = run_loco_evaluation(df, feature_cols)
loco_summary = loco_results[["recall", "precision", "f1", "roc_auc"]].agg(["mean", "std"]).round(3)
loco_summary
```

### Cell 6: Random Stratified Split 80/20 & Threshold Tuning
```python
strat_key = df[CITY_COL].astype(str) + "_" + df[TARGET_COL].astype(str)
X = df[feature_cols]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat_key
)

rf_split = make_rf().fit(X_train, y_train)
y_proba_split = rf_split.predict_proba(X_test)[:, 1]

def find_threshold_for_recall(y_true, y_proba, target_recall=TARGET_RECALL):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold, best_precision = 0.0, -1.0
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        r = recall_score(y_true, pred, zero_division=0)
        p = precision_score(y_true, pred, zero_division=0)
        rows.append({"threshold": round(t, 3), "recall": round(r, 3), "precision": round(p, 3)})
        
        if r >= target_recall and p > best_precision:
            best_threshold, best_precision = t, p
    return best_threshold, pd.DataFrame(rows)

best_threshold_split, threshold_scan_split = find_threshold_for_recall(y_test, y_proba_split)
print(f"Threshold terpilih (target recall >= {TARGET_RECALL}): {best_threshold_split:.4f}")
```

### Cell 7: Validasi Ulang Threshold pada Setiap Fold LOCO
```python
def run_loco_with_threshold(df, feature_cols, threshold):
    cities = sorted(df[CITY_COL].unique())
    rows = []
    for city in cities:
        train_mask = df[CITY_COL] != city
        test_mask = df[CITY_COL] == city
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, TARGET_COL]
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, TARGET_COL]

        model = make_rf().fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        rows.append({
            "held_out_city": city,
            "threshold": threshold,
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        })
    return pd.DataFrame(rows)

loco_at_threshold = run_loco_with_threshold(df, feature_cols, best_threshold_split)
print("Rata-rata performa LOCO pada threshold hasil tuning:")
print(loco_at_threshold[["recall", "precision", "f1"]].mean().round(3))
```

### Cell 8: Pelatihan Model Final pada Seluruh Dataset
```python
rf_final = make_rf()
rf_final.fit(X, y)
print("Model final Random Forest berhasil dilatih pada seluruh dataset:", X.shape)
```

### Cell 9: SHAP Explanation (Global & Local)
```python
explainer = shap.TreeExplainer(rf_final)
shap_values = explainer.shap_values(X)

# Slicing shap_values untuk Scikit-Learn Random Forest
if isinstance(shap_values, list):
    shap_values = shap_values[-1]  # ambil kelas 1 (slum)

# Hitung rata-rata kontribusi fitur (Global Importance)
global_importance = pd.Series(
    np.abs(shap_values).mean(axis=0), index=feature_cols
).sort_values(ascending=False)

print("Top 5 fitur paling berkontribusi (global SHAP):")
print(global_importance.head(5))
```

### Cell 10: Ekspor Artefak dan Metadata JSON
```python
joblib.dump(rf_final, MODEL_PATH)
print(f"Model disimpan -> {MODEL_PATH}")

metadata = {
    "model_file": MODEL_PATH.name,
    "model_type": "RandomForestClassifier",
    "threshold": round(float(best_threshold_split), 4),
    "threshold_selection_method": f"max precision subject to recall >= {TARGET_RECALL} (random stratified split)",
    "feature_names": feature_cols,
    "n_features": len(feature_cols),
    "target_column": TARGET_COL,
    "cities": sorted(df[CITY_COL].unique().tolist()),
    "n_training_rows": int(len(df)),
    "class_balance": {
        "slum": int(y.sum()),
        "non_slum": int((y == 0).sum()),
        "pct_slum": round(float(y.mean() * 100), 2),
    },
    "validation_metrics": {
        "random_split": {
            "recall": round(float(recall_score(y_test, (y_proba_split >= best_threshold_split).astype(int), zero_division=0)), 4),
            "precision": round(float(precision_score(y_test, (y_proba_split >= best_threshold_split).astype(int), zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, (y_proba_split >= best_threshold_split).astype(int), zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba_split)), 4),
        },
        "loco_at_threshold": {
            "recall_mean": round(float(loco_at_threshold["recall"].mean()), 4),
            "precision_mean": round(float(loco_at_threshold["precision"].mean()), 4),
            "f1_mean": round(float(loco_at_threshold["f1"].mean()), 4),
        },
        "loco_default_threshold_0_5": {
            "recall_mean": round(float(loco_summary.loc["mean", "recall"]), 4),
            "precision_mean": round(float(loco_summary.loc["mean", "precision"]), 4),
            "roc_auc_mean": round(float(loco_summary.loc["mean", "roc_auc"]), 4),
        },
    },
    "top_global_features": global_importance.head(10).round(4).to_dict(),
    "trained_on": str(date.today()),
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata RF disimpan -> {METADATA_PATH}")
```

---

## 4. Perbandingan Teoretis: XGBoost vs. Random Forest

Berikut adalah tabel perbandingan untuk memperjelas alasan mengapa kita menyertakan alternatif Random Forest dalam eksperimen ini:

| Aspek | XGBoost | Random Forest |
| :--- | :--- | :--- |
| **Metode Dasar** | **Boosting**: Pohon dibangun secara sekuensial untuk mengoreksi kesalahan pohon sebelumnya (residual learning). | **Bagging**: Pohon dibangun secara paralel dan independen menggunakan bootstrap samples (voting/rata-rata). |
| **Kerentanan Overfitting** | Lebih tinggi jika `learning_rate` terlalu besar atau `max_depth` terlalu tinggi pada dataset kecil. | Lebih rendah/sangat stabil karena efek ensemble averaging meredam varians tinggi pohon individual. |
| **Penanganan Imbalance** | Mengalikan loss kelas minoritas dengan `scale_pos_weight` secara global. | Menggunakan `class_weight='balanced_subsample'` yang menyesuaikan bobot secara lokal di tingkat sampel bootstrap. |
| **Kebutuhan Parameter Tuning** | Sangat sensitif terhadap kombinasi hyperparameter (butuh tuning intensif). | Relatif kokoh dengan nilai parameter default atau tuning sederhana. |
| **Kecepatan Training** | Sangat cepat dengan komputasi paralel (`n_jobs=-1`), tetapi komputasi sekuensial per split membutuhkan resource. | Sangat mudah diparalelkan karena pohon-pohon dibangun secara independen satu sama lain. |

---
*Dokumen ini dibuat sebagai referensi teknis yang komprehensif untuk implementasi algoritma Random Forest berikutnya.*
