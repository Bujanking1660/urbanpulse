# Urban Pulse — Dokumentasi Preprocessing Pipeline

> **File**: `preprocessing.ipynb`  
> **Tujuan**: Mengolah 4 CSV hasil ekstraksi fitur satelit (Google Earth Engine) per kota menjadi satu dataset bersih yang siap digunakan untuk training model klasifikasi slum/non-slum (Random Forest / XGBoost).

---

## 📁 Struktur Input & Output

| Jenis | Path | Keterangan |
|---|---|---|
| **Input** | `data/raw_features/features_*.csv` | 4 CSV per kota dari GEE |
| **Output** | `data/processed/features_buildings_clean.csv` | Dataset penuh + flag outlier |
| **Output** | `data/processed/feature_columns.csv` | Daftar 41 kolom fitur model |
| **Output** | `data/processed/random_split_train.csv` | Set training (80%) |
| **Output** | `data/processed/random_split_test.csv` | Set test (20%) |
| **Output** | `data/processed/loco_fold_summary.csv` | Ringkasan Leave-One-City-Out folds |

---

## 🔬 Penjelasan Setiap Cell

### Cell 1 — Import Library

```python
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```

**Tujuan**: Mengimpor semua library yang dibutuhkan.

| Library | Kegunaan |
|---|---|
| `pathlib.Path` | Manajemen path file yang portabel (Windows/Linux) |
| `dataclasses.dataclass` | Membuat struktur data `LOCOFold` yang rapi |
| `numpy` | Operasi array untuk indexing pada split data |
| `pandas` | Manipulasi tabular data utama |
| `sklearn.model_selection` | Fungsi `train_test_split` untuk random split |

`pd.set_option` diset agar semua kolom tampil saat preview dataframe.

---

### Cell 2 — Konfigurasi (Bagian 0)

```python
RAW_DIR = Path("data/raw_features")
OUT_DIR = Path("data/processed")
CITY_FILES = {"ambon": ..., "dki": ..., "kebumen": ..., "samarinda": ...}
ID_COLS = ["unit_id", "unit_name", "city"]
LEAKAGE_COLS = ["slum_frac", "kumuh_lvl"]
TARGET_COL = "slum"
RANDOM_STATE = 42
TEST_SIZE = 0.2
```

**Tujuan**: Mendefinisikan semua konstanta konfigurasi pipeline di satu tempat.

**Poin penting — Data Leakage**:
- `slum_frac` adalah proporsi area kumuh yang dipakai untuk **membuat** label `slum`. Jika dimasukkan sebagai fitur, model akan "mencontek" dari labelnya sendiri → **dikeluarkan**.
- `kumuh_lvl` adalah kategori severity turunan dari label → **dikeluarkan** juga.
- `ID_COLS` (`unit_id`, `unit_name`, `city`) adalah metadata identitas, bukan sinyal prediktif.

---

### Cell 3 — Fungsi `load_and_combine()` (Bagian 1)

```python
def load_and_combine(raw_dir) -> pd.DataFrame:
    ...
```

**Tujuan**: Mendefinisikan fungsi yang membaca 4 CSV per kota dan menggabungkannya menjadi satu DataFrame.

**Proses**:
1. Iterasi 4 kota (ambon, dki, kebumen, samarinda).
2. Setiap file dibaca dengan `pd.read_csv`.
3. Dilakukan validasi: kolom `city` di dalam file harus sesuai dengan nama kota yang diharapkan (peringatan jika tidak cocok).
4. Semua frame digabung dengan `pd.concat(ignore_index=True)`.

---

### Cell 4 — Eksekusi Load & Tampil Sampel (Bagian 1)

```python
df = load_and_combine()
print(f"Loaded: {df.shape}")
df.head()
```

**Tujuan**: Menjalankan `load_and_combine()` dan menampilkan 5 baris pertama.

**Output yang diharapkan**: `Loaded: (830, 40)` — 830 unit kelurahan/desa, 40 kolom (termasuk metadata dan fitur mentah dari GEE).

**Kolom utama** yang terlihat:
| Kelompok | Kolom |
|---|---|
| Identitas | `unit_id`, `unit_name`, `city`, `area_m2` |
| Label | `slum` (0/1), `slum_frac`, `kumuh_lvl` |
| Spektral (Sentinel-2) | `ndvi_*`, `ndbi_*`, `bsi_*`, `brightness_*` |
| Tekstur GLCM | `ndbi_contrast_*`, `ndbi_ent_*`, `ndbi_var_*` |
| SAR (Sentinel-1) | `VV_*`, `VH_*`, `vv_vh_*` |
| Bangunan (OSM/GEE) | `building_*`, `b_count`, `b_area_*`, `b_coverage`, `b_density_km2` |

---

### Cell 5 — Fungsi `validate()` (Bagian 2)

```python
def validate(df) -> None:
    ...
```

**Tujuan**: Mendefinisikan fungsi validasi kualitas data.

**Pemeriksaan yang dilakukan**:
1. **Duplikat `unit_id`** — Jika ada, pipeline langsung berhenti (`raise ValueError`).
2. **Missing values** — Mengecek kolom fitur (bukan ID/leakage/target). Hanya peringatan, tidak menghentikan pipeline.
3. **Distribusi kelas per kota** — Membuat crosstab `city × slum` beserta persentase slum. Penting karena dataset ini *imbalanced* dan distribusinya bervariasi antar kota.

---

### Cell 6 — Eksekusi Validasi (Bagian 2)

```python
validate(df)
```

**Output aktual**:
```
OK: tidak ada missing values di kolom fitur numerik.

Distribusi kelas per kota:
slum         0   1  total  pct_slum
city
ambon       35  15     50      30.0
dki        199  62    261      23.8
kebumen    433  27    460       5.9
samarinda   45  14     59      23.7

Total: 830 unit | slum=118 (14.2%)
```

**Insight penting**: Dataset sangat imbalanced (14.2% slum). Kebumen memiliki proporsi slum terendah (5.9%), yang bisa jadi tantangan dalam evaluasi LOCO.

---

### Cell 7 — Statistik Deskriptif Fitur Numerik (Bagian 3)

```python
num_cols = df.select_dtypes(include="number").columns.tolist()
df[num_cols].describe().T
```

**Tujuan**: Inspeksi cepat distribusi semua fitur numerik (count, mean, std, min, max, quartiles).

**Catatan**: Tree-based model (RF/XGBoost) tidak memerlukan fitur ter-*scale*, tetapi statistik deskriptif ini berguna untuk:
- Mendeteksi anomali kasar (nilai yang tidak masuk akal)
- Memahami rentang nilai setiap fitur
- Menjadi referensi saat ada outlier atau data baru yang masuk

---

### Cell 8 — Fungsi `add_engineered_features()` (Bagian 4)

```python
def add_engineered_features(df) -> pd.DataFrame:
    ...
```

**Tujuan**: Mendefinisikan fungsi rekayasa fitur (feature engineering) berbasis domain knowledge slum mapping.

**7 fitur interaksi yang dibuat**:

| Fitur | Formula | Interpretasi Domain |
|---|---|---|
| `built_minus_green` | `ndbi_mean - ndvi_mean` | Makin tinggi → kawasan terbangun dominan, sedikit vegetasi (karakteristik slum padat) |
| `texture_to_built_ratio` | `ndbi_contrast_mean / (|ndbi_mean| + ε)` | Tekstur GLCM kasar relatif terhadap intensitas built-up; slum sering menunjukkan tekstur tidak teratur |
| `small_building_dominance` | `b_area_median / (b_area_mean + ε)` | Rasio mendekati 1 → bangunan kecil mendominasi; mendekati 0 → ada bangunan besar yang menarik rata-rata |
| `dense_lowrise` | `b_density_km2 / (building_height_mean + ε)` | Kepadatan bangunan tinggi tapi tinggi rendah → pola *low-rise dense* khas permukiman kumuh |
| `packing_tightness` | `b_count / (b_area_mean + ε)` | Banyak bangunan kecil dalam satu unit area → kepadatan tinggi |
| `b_size_cv` | `b_area_std / (b_area_mean + ε)` | Koefisien variasi ukuran bangunan; CV tinggi → ukuran bangunan sangat bervariasi (pola organik) |
| `sar_built_interaction` | `vv_vh_mean × ndbi_mean` | Interaksi sinyal SAR dengan indeks built-up; menggabungkan informasi struktur 3D (SAR) dengan indeks 2D |

> **Referensi**: Fitur-fitur ini terinspirasi dari studi Leonita et al. (2018) dan Urban Science (2024) tentang slum mapping di Indonesia menggunakan karakteristik fisik permukiman.

---

### Cell 9 — Eksekusi Feature Engineering (Bagian 4)

```python
df = add_engineered_features(df)
df[new_cols].describe().T
```

**Tujuan**: Menjalankan `add_engineered_features()` dan menampilkan statistik 7 fitur baru.

**Output**: Konfirmasi "Ditambahkan 7 fitur interaksi" beserta statistik deskriptifnya. Setelah langkah ini, dataframe memiliki 47 kolom (40 asli + 7 baru).

---

### Cell 10 — Fungsi `get_feature_columns()` (Bagian 5)

```python
def get_feature_columns(df) -> list[str]:
    exclude = set(ID_COLS + LEAKAGE_COLS + [TARGET_COL])
    cols = [c for c in df.columns if c not in exclude]
    ...
    return cols
```

**Tujuan**: Mendefinisikan fungsi yang mengembalikan daftar kolom yang **aman** dipakai sebagai input model.

**Logika pemisahan kolom**:
- **Dibuang**: `unit_id`, `unit_name`, `city` (identitas) + `slum_frac`, `kumuh_lvl` (leakage) + `slum` (target)
- **Divalidasi**: Semua kolom yang tersisa harus numerik; jika ada kolom non-numerik akan raise error (perlu encoding manual)
- **Dikembalikan**: Semua kolom fitur yang bersih

---

### Cell 11 — Eksekusi Pemilihan Fitur (Bagian 5)

```python
feature_cols = get_feature_columns(df)
print(f"Total kolom fitur model: {len(feature_cols)}")
feature_cols
```

**Output**:
```
Total kolom fitur model: 41
Kolom dibuang dari fitur (leakage/identitas): ['unit_id', 'unit_name', 'city', 'slum_frac', 'kumuh_lvl']
```

**41 fitur final** terdiri dari: `area_m2` + 14 fitur spektral/tekstur Sentinel-2 + 6 fitur SAR Sentinel-1 + 6 fitur building presence (GEE) + 7 fitur statistik bangunan OSM + 7 fitur rekayasa interaksi.

---

### Cell 12 — Fungsi `flag_outliers()` (Bagian 6)

```python
def flag_outliers(df, feature_cols, z_thresh=4.0) -> pd.DataFrame:
    z = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()
    df["_n_outlier_features"] = (z.abs() > z_thresh).sum(axis=1)
    df["_is_outlier"] = df["_n_outlier_features"] > 0
    return df
```

**Tujuan**: Mendefinisikan fungsi deteksi outlier menggunakan Z-score.

**Filosofi desain**:
- Threshold yang digunakan adalah **|z-score| > 4** (lebih konservatif dari threshold umum 3) agar tidak terlalu agresif memflag data.
- Baris **TIDAK dihapus** otomatis — hanya ditambahkan 2 kolom flag:
  - `_n_outlier_features`: jumlah fitur yang melebihi threshold di baris tersebut
  - `_is_outlier`: boolean flag (True jika ≥ 1 fitur jadi outlier)
- Keputusan untuk menyimpan outlier karena **tree-based model tahan terhadap outlier**; flag ini untuk keperluan inspeksi manual atau eksperimen exclude secara opsional saat training.

---

### Cell 13 — Eksekusi Deteksi Outlier (Bagian 6)

```python
df = flag_outliers(df, feature_cols)
df.loc[df["_is_outlier"], ["unit_id", "city", "slum", "_n_outlier_features"]] \
    .sort_values("_n_outlier_features", ascending=False).head(10)
```

**Output**: `73 unit terflag punya >=1 fitur dengan |z-score| > 4` (~8.8% dari total data).

Menampilkan 10 unit dengan jumlah fitur outlier terbanyak — mayoritas dari DKI Jakarta dan Samarinda, semua bernilai 3 fitur yang melampaui threshold.

---

### Cell 14 — Fungsi `LOCOFold` & `make_loco_folds()` (Bagian 7)

```python
@dataclass
class LOCOFold:
    held_out_city: str
    train_idx: np.ndarray
    test_idx: np.ndarray

def make_loco_folds(df) -> list[LOCOFold]:
    ...

def summarize_loco_folds(df, folds) -> pd.DataFrame:
    ...
```

**Tujuan**: Mendefinisikan struktur dan fungsi untuk **Leave-One-City-Out (LOCO)** cross-validation.

**Mengapa LOCO, bukan random split biasa?**

> Data spasial memiliki *spatial autocorrelation* — unit yang berdekatan geografis cenderung berkorelasi. Random split akan "mengontaminasi" train dan test dengan data dari kota yang sama, sehingga estimasi performa **terlalu optimis**. LOCO memastikan test set benar-benar dari distribusi yang belum pernah dilihat model.

**Cara kerja**:
- Untuk setiap kota (4 fold): train pakai 3 kota lain, test pakai kota yang di-hold-out.
- Mensimulasikan deployment nyata: model dipakai untuk prediksi di kota baru.

---

### Cell 15 — Eksekusi LOCO Folds (Bagian 7)

```python
loco_folds = make_loco_folds(df)
loco_summary = summarize_loco_folds(df, loco_folds)
loco_summary
```

**Output**:
| held_out_city | n_train | train_slum_pct | n_test | test_slum_pct |
|---|---|---|---|---|
| ambon | 780 | 13.2% | 50 | 30.0% |
| dki | 569 | 9.8% | 261 | 23.8% |
| kebumen | 370 | 24.6% | 460 | 5.9% |
| samarinda | 771 | 13.5% | 59 | 23.7% |

**Insight**: Fold kebumen adalah yang paling menantang — test set sangat besar (460 unit) dengan proporsi slum sangat rendah (5.9%), sementara training set justru memiliki proporsi slum lebih tinggi (24.6%).

---

### Cell 16 — Fungsi `make_random_split()` (Bagian 8)

```python
def make_random_split(df, feature_cols):
    strat_key = df["city"].astype(str) + "_" + df[TARGET_COL].astype(str)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index,
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=strat_key,
    )
```

**Tujuan**: Mendefinisikan fungsi random split 80/20 yang **terstratifikasi** pada kombinasi kota × label.

**Mengapa stratifikasi pada `city_label`?**
- Tanpa stratifikasi, bisa saja kota kecil (ambon: 50 unit) kurang terwakili di test set.
- Stratifikasi ganda memastikan proporsi kelas (slum/non-slum) **dan** representasi kota terjaga di keduanya.

**Catatan penting**: Split ini digunakan untuk **eksplorasi cepat dan hyperparameter tuning**, **bukan** sebagai metrik final yang dilaporkan. Metrik final harus dari LOCO.

---

### Cell 17 — Eksekusi Random Split (Bagian 8)

```python
X_train, X_test, y_train, y_test, idx_train, idx_test = make_random_split(df, feature_cols)
print(f"Train: {X_train.shape} | slum={y_train.mean()*100:.1f}%")
print(f"Test:  {X_test.shape} | slum={y_test.mean()*100:.1f}%")
```

**Output**:
```
Train: (664, 41) | slum=14.3%
Test:  (166, 41) | slum=13.9%
```

Proporsi slum di train (14.3%) dan test (13.9%) hampir identik — bukti stratifikasi berhasil.

---

### Cell 18 — Simpan Output (Bagian 9)

```python
df.to_csv(OUT_DIR / "features_buildings_clean.csv", index=False)
pd.Series(feature_cols).to_csv(OUT_DIR / "feature_columns.csv", index=False)
train_df.to_csv(OUT_DIR / "random_split_train.csv", index=False)
test_df.to_csv(OUT_DIR / "random_split_test.csv", index=False)
loco_summary.to_csv(OUT_DIR / "loco_fold_summary.csv", index=False)
```

**Tujuan**: Menyimpan semua artifact output ke `data/processed/`.

| File Output | Isi | Kegunaan |
|---|---|---|
| `features_buildings_clean.csv` | Seluruh 830 baris + semua kolom + flag outlier | Dataset utama; dipakai modul training |
| `feature_columns.csv` | Daftar 41 nama fitur | Training script bisa load dan langsung pakai tanpa hardcode |
| `random_split_train.csv` | 664 baris training | Eksplorasi / tuning cepat |
| `random_split_test.csv` | 166 baris test | Evaluasi sanity check |
| `loco_fold_summary.csv` | 4 baris ringkasan LOCO | Referensi training script untuk merekonstruksi folds |

---

## 🗺️ Alur Pipeline Keseluruhan

```
4 CSV Raw (GEE)
       │
       ▼
[Cell 3-4] load_and_combine()
  830 baris, 40 kolom
       │
       ▼
[Cell 5-6] validate()
  Cek duplikat, missing values, distribusi kelas
       │
       ▼
[Cell 7] Statistik deskriptif
  Inspeksi visual rentang nilai
       │
       ▼
[Cell 8-9] add_engineered_features()
  +7 fitur interaksi → 47 kolom
       │
       ▼
[Cell 10-11] get_feature_columns()
  Pisahkan 41 fitur dari ID/leakage/target
       │
       ▼
[Cell 12-13] flag_outliers()
  Flag 73 unit outlier (z>4), tidak dihapus
       │
       ▼
[Cell 14-15] make_loco_folds()       [Cell 16-17] make_random_split()
  4 fold LOCO untuk evaluasi utama     Split 80/20 untuk eksplorasi
       │                                      │
       └──────────────┬───────────────────────┘
                      ▼
              [Cell 18] Simpan Output
              5 file CSV ke data/processed/
```

---

## 📚 Referensi Akademik

| Fitur / Metode | Referensi |
|---|---|
| **NDBI** | Zha et al. (2003), *Int. Journal of Remote Sensing*, 24(3) |
| **Tekstur GLCM** | Haralick et al. (1973), *IEEE Trans. Systems, Man, Cybernetics* |
| **GLCM untuk slum mapping** | Kuffer et al. (2016), *IEEE JSTARS*, 9(5) |
| **RF & SVM untuk slum di Indonesia** | Leonita et al. (2018), *Remote Sensing*, 10(10) |
| **Karakteristik fisik slum** | Urban Science (2024), 8(4), doi:10.3390/urbansci8040189 |
| **SAR Sentinel-1 untuk built-up** | Koppel et al. (2017), *Int. Journal of Remote Sensing*, 38(22) |
| **Spatial cross-validation** | Roberts et al. (2017), *Ecography*, 40(8); Le Rest et al. (2014), *Global Ecology and Biogeography* |

---

# Urban Pulse — Dokumentasi Modeling Pipeline

> **Files**: [modeling_combine.ipynb](file:///d:/urbanpulse/modeling_combine.ipynb) (Combined Comparison), [modeling.ipynb](file:///d:/urbanpulse/modeling.ipynb) (XGBoost), [randomforest.ipynb](file:///d:/urbanpulse/randomforest.ipynb) (Random Forest)  
> **Tujuan**: Membangun model machine learning terbaik untuk klasifikasi biner slum/non-slum dengan skema evaluasi yang robust terhadap autokorelasi spasial, serta menghasilkan penjelasan prediktor penting berbasis SHAP.

---

## 📁 Struktur Input & Output Modeling

| Jenis | Path | Keterangan |
|---|---|---|
| **Input** | `data/processed/features_buildings_clean.csv` | Dataset bersih hasil preprocessing |
| **Input** | `data/processed/feature_columns.csv` | Daftar 41 fitur terpilih |
| **Output** | `models/slum_xgb_pipeline.joblib` | Model final XGBoost terlatih |
| **Output** | `models/model_metadata.json` | Kredensial & metrik evaluasi model XGBoost |
| **Output** | `models/slum_rf_pipeline.joblib` | Model final Random Forest terlatih |
| **Output** | `models/model_rf_metadata.json` | Kredensial & metrik evaluasi model Random Forest |
| **Output** | `models/model_comparison.png` | Grafik komparasi performa model (XGBoost, RF, Ensemble) |
| **Output** | `modeling_combine_report.md` | Laporan lengkap komparasi performa model gabungan |

---

## 🤖 Ringkasan Metode Pemodelan

Proyek ini mengeksplorasi dua algoritma berbasis pohon (*tree-based models*) serta hasil ensemblenya yang tangguh terhadap data tabular:
1. **XGBoost (Extreme Gradient Boosting)**: Boosting sekuensial yang melatih pohon baru untuk mengoreksi error dari pohon sebelumnya. Sangat baik dalam mendeteksi batas kelas yang rumit spasi (*spatial transferability*).
2. **Random Forest**: Bagging paralel yang merata-ratakan prediksi ratusan pohon keputusan independen. Sangat stabil terhadap overfitting spuria pada dataset ukuran kecil.
3. **Ensemble (Gabungan)**: Model kombinasi berbasis soft-voting yang merata-ratakan probabilitas prediksi dari model XGBoost dan Random Forest untuk menstabilkan prediksi dan mengurangi variabilitas kesalahan spuria.

---

## 🔬 Skema Evaluasi & Parameter Best Practice

### 1. Leave-One-City-Out (LOCO) Cross-Validation (Metrik Utama)
Karena data spasial memiliki korelasi lokal, model dievaluasi secara silang dengan menahan satu kota penuh sebagai data testing, dan melatih model pada tiga kota lainnya. Ini memastikan evaluasi model mencerminkan kinerja saat di-deploy ke kota baru (*transferability*).

### 2. Penanganan Class Imbalance
* **XGBoost**: Menggunakan parameter `scale_pos_weight` yang dihitung dinamis pada data training di setiap fold LOCO.
* **Random Forest**: Menggunakan parameter `class_weight='balanced_subsample'` yang menghitung ulang bobot kelas secara dinamis untuk setiap sampel bootstrap pohon.

### 3. Cost-Sensitive Threshold Tuning
Model menggeser threshold probabilitas (default 0.5) untuk memprioritaskan **Recall kelas slum $\ge 0.80$** (karena melompati area kumuh jauh lebih merugikan secara kebijakan daripada kesalahan deteksi positif palsu).
* Threshold tuned **XGBoost**: `0.0900`
* Threshold tuned **Random Forest**: `0.4100`
* Threshold tuned **Ensemble (Gabungan)**: `0.2500`

---

## 📊 Hasil Komparasi Model Spasial (LOCO Tuned)

Laporan detail dan visualisasi lengkap disimpan pada berkas **[modeling_combine_report.md](file:///d:/urbanpulse/modeling_combine_report.md)** dan `models/model_comparison.png`.

| Metrik Evaluasi | XGBoost (Threshold: 0.09) | Random Forest (Threshold: 0.41) | Ensemble (Threshold: 0.25) |
|---|---|---|---|
| **Recall (Random Split)** | **0.8261** | **0.8261** | **0.8261** |
| **Precision (Random Split)** | `0.3393` | **`0.3725`** | `0.3393` |
| **ROC-AUC (Random Split)** | `0.8045` | **`0.8310`** | `0.8227` |
| **Recall Rata-rata (LOCO)** | **`0.7052`** | `0.6482` | `0.6562` |
| **Precision Rata-rata (LOCO)** | **`0.4098`** | `0.2810` | `0.2797` |

---

## 📚 Referensi Akademik Modeling

| Algoritma / Metode | Referensi |
|---|---|
| **XGBoost** | Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. |
| **Random Forest** | Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32. |
| **SHAP** | Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. |
| **Class Imbalance** | King, G., & Zeng, L. (2001). Logistic regression in rare events data. |
