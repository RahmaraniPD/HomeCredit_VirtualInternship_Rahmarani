# Home Credit Indonesia – Prediksi Kesulitan Pembayaran Nasabah

Proyek *virtual internship* Data Scientist di PT Home Credit Indonesia. Tujuannya membangun model *machine learning* untuk memprediksi apakah seorang nasabah berpotensi mengalami **kesulitan pembayaran** (`TARGET = 1`) atau **tidak** (`TARGET = 0`), sebagai bahan pertimbangan apakah pengajuan pinjaman layak disetujui.

## Latar Belakang

PT Home Credit Indonesia adalah perusahaan pembiayaan yang melayani konsumen yang berbelanja secara online maupun offline, serta pembiayaan untuk memenuhi kebutuhan finansial nasabah. Model prediksi risiko membantu proses *credit scoring* agar keputusan persetujuan pinjaman lebih terukur.

## Dataset

| File | Keterangan |
|------|------------|
| `application_train.csv` | Data aplikasi pinjaman beserta label `TARGET` (307.511 baris awal) |
| `application_test.csv` | Data aplikasi tanpa label, untuk dibuatkan prediksi (48.744 baris awal) |

- `SK_ID_CURR`: ID unik pinjaman nasabah
- `TARGET`: 1 = mengalami kesulitan pembayaran, 0 = tidak

> File dataset tidak disertakan di repository ini. Letakkan kedua file CSV di folder yang sama dengan notebook sebelum menjalankannya.

## Struktur Repository

```
.
├── HomeCredit_VirtualIntership.ipynb   # Notebook utama (preprocessing → modelling → evaluasi → prediksi)
├── application_train.csv               # (tambahkan sendiri) data latih
├── application_test.csv                # (tambahkan sendiri) data uji
└── README.md
```

Output yang dihasilkan notebook saat dijalankan:

- `application_train_clean.csv` dan `application_test_clean.csv` – data setelah dibersihkan
- `Prediction Result1.csv` – prediksi Random Forest
- `Prediction Result2.csv` – prediksi Gradient Boosting

## Alur Pengerjaan

### 1. Data Pre-processing
- **Duplikat**: tidak ditemukan data duplikat pada data train maupun test.
- **Missing value**: kolom yang tidak diperlukan dan kolom dengan missing value > 50% dibuang; sisanya diisi `0`.
- **Pengecekan typo/kategori**: baris dengan `CODE_GENDER = 'XNA'` dibuang; kolom `DAYS_*` dikonversi menjadi nilai positif.
- **Outlier**: dibuang dengan metode IQR (1,5 × IQR) pada kolom numerik terpilih.
  - Train: 307.507 → 203.358 baris
  - Test: 48.744 → 29.085 baris
- **Encoding fitur kategorikal**
  - *Label encoding*: `NAME_CONTRACT_TYPE`, `CODE_GENDER`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`
  - *One-hot encoding*: `NAME_INCOME_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`
  - *Drop* (terlalu banyak nilai unik): `OCCUPATION_TYPE`, `WEEKDAY_APPR_PROCESS_START`

### 2. Modelling
- Seleksi fitur berdasarkan *correlation matrix heatmap*.
- Data train dibagi **80% training / 20% testing** (`random_state=0`).
- Tiga algoritma dilatih dengan parameter default scikit-learn:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier

### 3. Evaluasi
Evaluasi memakai *confusion matrix* dan *classification report* (precision, recall, F1) pada data training dan testing.

| Model | Akurasi (test) | Precision / Recall kelas 1 (kesulitan bayar) |
|-------|:-:|:-:|
| Logistic Regression | 0,92 | 0,00 / 0,00 |
| Random Forest | 1,00 | 1,00 / 1,00 |
| Gradient Boosting | 1,00 | 1,00 / 1,00 |

### 4. Prediksi
Model Random Forest dan Gradient Boosting dipakai untuk memprediksi `application_test`, hasilnya disimpan ke CSV.

## Keterbatasan yang Perlu Diperhatikan

Sebelum hasil di atas dipakai sebagai acuan, ada beberapa hal yang sebaiknya diperbaiki:

1. **Kebocoran target (data leakage)**. Pada tahap training, `X = newDF` masih memuat kolom `TARGET`, sehingga fitur sama dengan label. Inilah sebab skor sempurna (1,00) pada Random Forest dan Gradient Boosting. Angka tersebut **tidak mencerminkan performa nyata**. Perbaikan: `X = newDF.drop(columns='TARGET')`.
2. **Ketidakseimbangan kelas**. Kelas 1 hanya sekitar 8% data. Logistic Regression memprediksi semua nasabah sebagai kelas 0 (akurasi 0,92 tetapi recall kelas 1 = 0), sehingga akurasi bukan metrik yang tepat. Pertimbangkan `class_weight`, SMOTE/*undersampling*, dan metrik seperti ROC-AUC, PR-AUC, atau recall kelas 1.
3. **Prediksi test set tidak lengkap**. Outlier juga dibuang dari data test, sehingga file prediksi hanya berisi 29.085 dari 48.744 baris dan tidak memuat `SK_ID_CURR`. Sebaiknya data test tidak difilter, dan ID disertakan di file hasil.

## Cara Menjalankan

1. Clone repository dan siapkan dataset (lihat bagian Dataset).
2. Install dependensi:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Jalankan notebook:

   ```bash
   jupyter notebook HomeCredit_VirtualIntership.ipynb
   ```

> Catatan: sel *one-hot encoding* hanya boleh dijalankan **sekali**. Jalankan ulang dari awal (*Restart & Run All*) bila perlu mengulang.

## Teknologi

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn

## Rencana Pengembangan

- [ ] Perbaiki kebocoran target dan evaluasi ulang ketiga model
- [ ] Tangani ketidakseimbangan kelas
- [ ] Hyperparameter tuning dan validasi silang
- [ ] Manfaatkan tabel pendukung lain (bureau, previous application, dll.) untuk fitur tambahan
- [ ] Simpan model terlatih dan pipeline *preprocessing* agar reproducible
