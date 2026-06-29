# Urban Pulse — Laporan Evaluasi Pemodelan Gabungan (XGBoost & Random Forest)

> **File**: `modeling_combine_report.md`  
> **Tujuan**: Menyajikan hasil analisis, perbandingan performa, pencarian decision threshold, dan analisis interpretabilitas (SHAP) secara gabungan antara model **XGBoost** dan **Random Forest** untuk klasifikasi kawasan kumuh (*slum mapping*).

---

## 📊 1. Ringkasan Model & Dataset

Model dilatih menggunakan dataset yang telah melalui proses preprocessing dan pembersihan dari kebocoran data (*data leakage*). Berikut adalah informasi umum terkait model dan dataset:

| Parameter | Keterangan / Nilai |
|---|---|
| **Jumlah Baris Data** | 830 kelurahan/desa |
| **Jumlah Fitur Klasifikasi** | 41 kolom (fitur fisik & spektral satelit) |
| **Keseimbangan Kelas** | Slum: 118 (14.22%) \| Non-Slum: 712 (85.78%) |
| **Target Utama Optimasi** | **Recall Kelas Slum $\ge 0.80$** (Tuning Threshold) |
| **Metode Pemilihan Threshold** | *Max Precision subject to Recall $\ge$ 0.80* (pada Stratified Split) |
| **Algoritma Model 1** | `XGBClassifier` (Gradient Boosting) |
| **Algoritma Model 2** | `RandomForestClassifier` (Bagging) |
| **Algoritma Model 3** | `Ensemble` (Soft-Voting: XGBoost + RF) |

---

## 🎯 2. Hasil Tuning Decision Threshold

Tuning threshold dilakukan untuk mengatasi dampak ketidakseimbangan kelas (*class imbalance*) dan memprioritaskan tangkapan kelas slum (Recall tinggi) dengan tetap meminimalkan kesalahan prediksi (Precision).

- **XGBoost**: Decision threshold diturunkan dari 0.50 menjadi **`0.0900`**.
- **Random Forest**: Decision threshold diturunkan dari 0.50 menjadi **`0.4100`**.
- **Ensemble (Gabungan)**: Decision threshold hasil tuning adalah **`0.2500`**.

---

## 📈 3. Perbandingan Kinerja pada Stratified Random Split (80% Train / 20% Test)

Metrik evaluasi pada bagian test split (20% data yang di-holdout) setelah menerapkan threshold hasil tuning:

| Model | Decision Threshold | Recall (Slum) | Precision (Slum) | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | `0.0900` | **`0.8261`** | `0.3393` | `0.4810` | `0.8045` |
| **Random Forest** | `0.4100` | **`0.8261`** | **`0.3725`** | **`0.5135`** | **`0.8310`** |
| **Ensemble (Gabungan)** | `0.2500` | **`0.8261`** | `0.3393` | `0.4810` | `0.8227` |

> 📌 **Analisis**: Pada skema *Random Split*, Random Forest menghasilkan performa terunggul secara keseluruhan (F1-Score `0.5135` dan ROC-AUC `0.8310`). Model Ensemble memiliki kinerja metrik biner yang identik dengan XGBoost (Recall `0.8261`, Precision `0.3393`, F1 `0.4810`) namun dengan nilai ROC-AUC yang lebih stabil (`0.8227` vs `0.8045`).

---

## 🗺️ 4. Evaluasi Generalisasi Spasial (Leave-One-City-Out / LOCO)

Skema LOCO dilakukan dengan melatih model pada 3 kota dan mengujinya pada 1 kota yang belum pernah dilihat sama sekali oleh model. Ini adalah simulasi paling realistis jika model dideploy pada kota baru.

### Performa Folds per Kota pada Threshold Terpilih
Berikut adalah metrik detail per kota yang menjadi test fold (held-out city) setelah tuning threshold:

| Kota Uji (Held-Out Fold) | XGBoost Recall | XGBoost Precision | Random Forest Recall | Random Forest Precision | Ensemble Recall | Ensemble Precision |
|---|---|---|---|---|---|---|
| **Ambon** | `0.800` | `0.571` | `0.800` | `0.571` | `0.800` | `0.571` |
| **DKI Jakarta** | **`0.984`** | `0.235` | `0.935` | `0.228` | `0.968` | `0.232` |
| **Kebumen** | `0.037` | **`0.500`** | `0.000` | `0.000` | `0.000` | `0.000` |
| **Samarinda** | `1.000` | `0.333` | `0.857` | `0.324` | `0.857` | `0.316` |
| **Rata-rata (Mean)** | **`0.7052`** | **`0.4098`** | `0.6482` | `0.2810` | `0.6562` | `0.2797` |

### 🔍 Temuan Utama Evaluasi LOCO:
1. **Keunggulan XGBoost dalam Generalisasi**: Rata-rata Recall spasial untuk **XGBoost (`0.7052`)** jauh lebih baik dibandingkan **Random Forest (`0.6482`)** dan **Ensemble (`0.6562`)**. Demikian pula rata-rata Precision spasial untuk **XGBoost (`0.4098`)** mengungguli **Random Forest (`0.2810`)** dan **Ensemble (`0.2797`)**.
2. **Karakteristik Model Ensemble**: Model Ensemble (Gabungan) memiliki performa rata-rata LOCO yang cenderung berada di antara XGBoost dan Random Forest (Recall `0.6562` vs RF `0.6482` vs XGB `0.7052`). Ini menunjukkan bahwa soft-voting meredam variansi prediksi tetapi tidak dapat mengejar ketajaman model XGBoost murni dalam menangkap pola spasial di kota baru.
3. **Masalah Kota Kebumen**: Ketiga model mengalami penurunan performa drastis saat menguji Kebumen (Recall sangat rendah). Hal ini disebabkan oleh:
   - Kebumen memiliki proporsi slum terendah (hanya 5.9%).
   - Karakteristik fisik wilayah kumuh perdesaan/semi-urban di Kebumen sangat berbeda dengan wilayah perkotaan padat seperti DKI Jakarta atau Samarinda, sehingga model kesulitan mengenali polanya.
4. **Optimisme Wilayah Padat (DKI & Samarinda)**: Pada kota-kota urban padat, XGBoost mampu menangkap hampir seluruh wilayah kumuh (Recall `0.984` di DKI dan `1.000` di Samarinda), sedangkan Ensemble juga menunjukkan Recall yang tinggi (`0.968` di DKI dan `0.857` di Samarinda).

---

## 🧠 5. Analisis Interpretabilitas Global (SHAP Feature Importance)

Menggunakan nilai kontribusi absolut SHAP (TreeExplainer) untuk memetakan 10 fitur terpenting yang memengaruhi keputusan prediksi kedua model:

| Peringkat | Fitur XGBoost | Nilai SHAP XGB | Fitur Random Forest | Nilai SHAP RF |
|:---:|---|:---:|---|:---:|
| **1** | `b_coverage` | `0.3753` | `building_presence_mean` | `0.0347` |
| **2** | `building_presence_stdDev` | `0.3674` | `b_coverage` | `0.0332` |
| **3** | `ndbi_var_stdDev` | `0.3041` | `building_presence_stdDev` | `0.0325` |
| **4** | `b_area_mean` | `0.2965` | `b_density_km2` | `0.0251` |
| **5** | `building_fractional_count_stdDev` | `0.2736` | `ndbi_var_stdDev` | `0.0194` |
| **6** | `VV_stdDev` | `0.2659` | `vv_vh_stdDev` | `0.0177` |
| **7** | `dense_lowrise` | `0.2499` | `VV_stdDev` | `0.0163` |
| **8** | `ndbi_contrast_mean` | `0.2381` | `building_height_mean` | `0.0152` |
| **9** | `ndbi_stdDev` | `0.2300` | `building_fractional_count_mean` | `0.0149` |
| **10** | `ndbi_ent_mean` | `0.2297` | `building_fractional_count_stdDev` | `0.0144` |

### 💡 Penjelasan Domain Fitur-Fitur Kunci:
- **Karakteristik Bangunan (`b_coverage`, `building_presence_mean`, `building_presence_stdDev`)**: Menjadi kontributor teratas di kedua model. Wilayah slum ditandai dengan persentase tutupan bangunan tinggi dan ketidakseragaman distribusi bangunan.
- **Indeks Built-up Tekstur (`ndbi_var_stdDev`, `ndbi_contrast_mean`, `ndbi_ent_mean`)**: Karakteristik spektral pemukiman kumuh memiliki tekstur spasial yang tidak beraturan/acak (entropi tinggi, kontras tinggi).
- **Fitur Hasil Rekayasa (`dense_lowrise`)**: Fitur `dense_lowrise` (kepadatan bangunan tinggi tapi tinggi bangunan rendah) terbukti sangat penting bagi model XGBoost dalam mengidentifikasi pola hunian slum padat vertikal rendah khas Indonesia.
- **Karakteristik SAR (`VV_stdDev`, `vv_vh_stdDev`)**: Deviasi standar sinyal radar Sentinel-1 membantu mengidentifikasi kekasaran permukaan struktur 3D dari bangunan di area slum.

---

## 🎨 6. Lampiran Visualisasi
Visualisasi detail kurva ROC-AUC dan grafik perbandingan metrik LOCO telah diekspor oleh pipeline ke direktori models:
- **File Visualisasi**: [model_comparison.png](file:///d:/urbanpulse/models/model_comparison.png)

---
*Laporan dibuat otomatis berdasarkan run final pada berkas `modeling_combine.ipynb` pada tanggal 2026-06-29.*
