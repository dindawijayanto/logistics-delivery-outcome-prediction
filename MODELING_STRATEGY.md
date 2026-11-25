# Strategi Modeling — Prediksi `status_penanganan` (Datavidia 11.0)

Dokumen ini merangkum temuan dari notebook 01–10 beserta seluruh output
(`eda_outputs/`, `section06–09_outputs/`), lalu menurunkan desain modeling yang
paling sesuai dengan masalah dan temuan tersebut.

---

## 1. Ringkasan masalah

| Aspek | Nilai |
|---|---|
| Tugas | Klasifikasi multikelas, 7 kelas, level shipment |
| Metrik kompetisi | **Macro F1**, sehingga setiap kelas berbobot sama |
| Train / Test | 1.243.617 / 362.245 shipment |
| Distribusi kelas | normal 38,1% · konfirmasi_alamat 21,9% · tertunda_operasional 14,9% · gagal_antar_awal 10,0% · retur_pengirim 7,1% · rusak_perjalanan 5,0% · klaim_hilang 3,0% |
| Baseline saat ini (nb 10, skenario B) | Akurasi 64,2% · **Macro F1 0,494** |
| F1 per kelas (baseline) | retur 0,23 · gagal 0,36 · rusak 0,37 · klaim 0,39 · konfirmasi 0,64 · tertunda 0,68 · normal 0,78 |

Macro F1 tertahan oleh empat kelas minoritas. Karena itu, strategi modeling di
bawah diarahkan untuk **menaikkan recall kelas minoritas** tanpa mengorbankan
kelas besar terlalu banyak.

---

## 2. Temuan kunci dan implikasinya

### F1. Tidak ada pergeseran distribusi train–test
- PSI semua fitur < 0,01 (tertinggi `sc_quality_mean` 0,008).
- Importance adversarial validation datar (tertinggi 0,03, `ship_month`).
- Pola missing train dan test praktis identik (selisih < 0,1 poin).

**Implikasi:** `StratifiedKFold` acak sudah valid. Skor OOF dapat dipercaya
sebagai proksi leaderboard, sehingga validasi waktu atau grup tidak diperlukan.

### F2. Setiap kelas punya "sumber sinyal" yang berbeda (SMD one-vs-rest)

| Kelas | Pembeda terkuat | Sumber |
|---|---|---|
| konfirmasi_alamat | `cust_address_quality` (−1,47), `cust_complaint_index` (+1,25), `jumlah_catatan`, `sc_n_upaya_antar` | **customer** |
| klaim_hilang | `declared_value` (+1,44), `iv_amount_*` (+1,1) | nilai barang, **invoice** |
| tertunda_operasional | `hub_density_index` (+1,05), `hub_capacity_index`, `sc_gap_*` (+0,9) | **hub**, scan |
| rusak_perjalanan | `sc_quality_mean` (−1,04), `cour_courier_score` (−0,85), `distance_km` (+0,74) | scan, **courier**, rute |
| gagal_antar_awal | `cour_courier_score` (−0,84), `cour_experience_years`, `sc_manual_rate` | **courier**, scan |
| retur_pengirim | `cust_complaint_index` (+0,77), **`iv_n_klaim`/`iv_count` (+0,54)** | **customer**, **invoice** |

**Implikasi:** masalah ini didominasi **atribut dan histori entitas** (customer,
courier, hub), ditambah ringkasan scan dan invoice. Model berbasis pohon dengan
fitur level-entitas yang kaya adalah pilihan yang tepat.

### F3. Histori target per entitas sangat prediktif (nb 09)
Korelasi target rate OOF dengan indikator kelas:

| Kelas | customer | courier | hub |
|---|---|---|---|
| tertunda_operasional | 0,10 | **0,59** | **0,44** |
| normal | 0,30 | **0,45** | 0,31 |
| konfirmasi_alamat | **0,52** | 0,09 | 0,07 |
| retur_pengirim | **0,37** | 0,05 | 0,06 |
| gagal_antar_awal | 0,08 | **0,32** | 0,09 |
| rusak_perjalanan | 0,11 | 0,24 | 0,07 |
| klaim_hilang | 0,03 | 0,23 | 0,25 |

- **88,6% baris test memiliki `customer_id` yang ada di train**, dan
  courier/hub praktis 100% ada di train. Angka "70% unseen" di
  `section07_outputs/unseen_category_results.csv` berasal dari *sampel*, sehingga
  menyesatkan.
- Saat ini **notebook 10 membuang `customer_id` dan tidak memakai fitur nb 09
  sama sekali.** Padahal kelas terburuk (**retur_pengirim, F1 0,23**) memiliki
  sinyal terkuatnya justru di histori customer (0,37).
- Fold nb 09 = fold nb 10. Keduanya memakai `StratifiedKFold(5, shuffle=True,
  random_state=42)` pada urutan target yang sama, sehingga fitur OOF nb 09 aman
  dipakai di CV nb 10 tanpa leakage antar-fold.
- Target rate di nb 09 **tidak di-smoothing**. Customer rata-rata hanya punya
  2,4 shipment, jadi rate-nya bernilai 0/0,5/1 yang sangat noisy, dan rate hasil
  OOF (80% data) sedikit berbeda skala dengan rate untuk test (100% data).

**Implikasi:** fitur target encoding (TE) entitas adalah peningkatan terbesar
yang belum dipakai. Smoothing perlu ditambahkan.

### F4. Fitur invoice (skenario C) layak dipakai
- Invoice `klaim` ada di test dengan proporsi yang sama dengan train (18%
  shipment di keduanya). `sc_n_selesai` juga tersedia di test.
- `iv_n_klaim > 0` **bukan leakage deterministik**: P(kelas | ada klaim) adalah
  normal 34%, konfirmasi 20%, retur 15%, dan seterusnya. Fitur ini sinyal
  moderat (retur 37% punya klaim vs 16% baseline; klaim_hilang 32%).
- Skenario B membuang semua `iv_*`, sehingga kehilangan sinyal untuk dua kelas
  terlemah (retur, klaim_hilang).

**Implikasi:** untuk tujuan kompetisi, gunakan **skenario C plus `iv_n_klaim`,
`iv_amount_klaim`, dan `sc_n_selesai`**. Ketiganya hanya dikecualikan karena
nama kolomnya, bukan karena hasil uji. Konfirmasi lewat ablation CV.

### F5. Redundansi fitur (nb 06)
Pasangan identik atau hampir identik: `distance_km`≈`rt_distance_sum`,
`x_address_risk`=`x_address_notes`, `iv_count`≈`iv_n_klaim`,
`iv_penalty_count`=`iv_n_gagal`, `iv_discount_missing`=`iv_n_langganan`,
`iv_amount_sum`≈`iv_tax_sum`, `promo_nama/tipe`↔`promo_id`,
`pkg_speed/handling_class`↔`package_id`.

**Implikasi:** buang salah satu dari tiap pasangan. Tujuannya mempercepat
training dan menstabilkan importance, bukan mengejar akurasi, karena pohon cukup
robust terhadap redundansi.

### F6. Fitur interaksi manual hampir tidak menambah informasi
Eta-squared fitur interaksi lebih rendah daripada fitur induknya, kecuali
`x_distance_route_complexity` (+0,006). Biarkan pohon menemukan interaksi
sendiri. Fitur baru yang lebih berguna adalah **deviasi terhadap profil entitas**
(lihat §3.2).

### F7. Kualitas data baik, missing kecil
- Tidak ada duplikasi, PK valid, dan FK valid kecuali `promo_id`. Orphan
  `promo_id` (83%) hanya satu key, yang artinya "tanpa promo", bukan error.
- Missing ≤ 3% (`volume_l`, `credit_score`, `courier_score`, `durasi_jam`).
  LightGBM menangani NaN secara native, jadi tidak perlu imputasi. Cukup
  tambahkan indikator missing untuk `cust_credit_score` dan `cour_courier_score`.

### F8. Diagnosis baseline
- `courier_id` (23.806 kategori) sebagai kategorikal mentah menjadi fitur gain
  terbesar, dan early stopping berhenti di sekitar iterasi 220. Ini indikasi
  overfit pada split kategori berkardinalitas tinggi.
- Prediksi memakai argmax tanpa kalibrasi threshold. Untuk Macro F1 pada data
  imbalanced, cara ini kurang cocok.
- Bobot `balanced` penuh hanya menaikkan F1 sedikit (fold 1: 0,4959 vs 0,4938)
  dan butuh sekitar 10× iterasi.

---

## 3. Desain modeling yang direkomendasikan

```
Data (skenario C+) ──► fitur entitas TE (OOF, smoothed) + frekuensi + deviasi
                    ──► LightGBM multiclass (5-fold, fold sama dgn nb09)
                    ──► CatBoost multiclass (5-fold)        [ensemble]
                    ──► rata-rata probabilitas OOF/test
                    ──► optimasi multiplier per kelas pada OOF (Macro F1)
                    ──► submission (rata-rata prediksi 5 model fold)
```

### 3.1 Validasi
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, sama dengan
  nb 09 dan nb 10.
- Semua fitur yang memakai target (TE) **wajib OOF** pada fold yang sama.
- Metrik utama: **Macro F1 OOF**. Laporkan juga F1 per kelas di setiap
  eksperimen.

### 3.2 Feature set

| Grup | Isi | Alasan (temuan) |
|---|---|---|
| Basis | Skenario C + `iv_n_klaim`, `iv_amount_klaim`, `sc_n_selesai`, minus fitur redundan (F5) | F4, F5 |
| TE entitas | Dari nb 09: `ent_{customer,courier,hub}_target_rate_*`, `_entropy`, `_max_rate`, `_shipment_count`, `_is_unseen`, `_*_mean` | F3 |
| Smoothing TE | `rate_s = (n·rate + m·prior) / (n + m)`, dengan m ≈ 10 untuk customer, 20 untuk courier, 50 untuk hub | F3 (noise customer) |
| Frekuensi | Jumlah shipment per customer/courier/hub dihitung dari **train+test** (tanpa target, legal) | Mengganti hitungan OOF yang bias skala |
| Deviasi | `declared_value / ent_customer_declared_value_mean`, `distance_km / ent_courier_distance_km_mean`, `cour_courier_score − rata-rata courier di hub` | Shipment "tidak biasa" bagi entitasnya → klaim/rusak |
| ID mentah | `hub_id` dan `package_id` tetap kategorikal. **`courier_id` dan `customer_id` mentah dibuang**, diwakili TE | F8 |
| Missing flag | `cust_credit_score_isna`, `cour_courier_score_isna` | F7 |

### 3.3 Model utama — LightGBM

```python
lgb_params = {
    "objective": "multiclass",
    "learning_rate": 0.03,
    "n_estimators": 5000,          # early stopping 200
    "num_leaves": 127,
    "min_child_samples": 200,
    "colsample_bytree": 0.6,
    "subsample": 0.8, "subsample_freq": 1,
    "reg_alpha": 0.5, "reg_lambda": 2.0,
    "cat_smooth": 50, "min_data_per_group": 200, "max_cat_to_onehot": 8,
    "random_state": 42, "n_jobs": -1, "verbosity": -1,
}
```

- **Bobot kelas "sqrt-balanced"**: `w_c = sqrt(N / (K · n_c))`. Bobot ini
  berada di antara tanpa bobot dan balanced penuh, sehingga mendorong recall
  minoritas tanpa membuat training 10× lebih lama.
- **Prediksi test = rata-rata 5 model fold**, bukan refit di full data. Dengan
  begitu tidak perlu menebak `n_estimators`, varians lebih kecil, dan RAM lebih
  hemat karena tidak perlu satu training penuh lagi.

### 3.4 Model kedua — CatBoost (untuk ensemble)
- Ordered target statistics menangani `customer_id` dan `courier_id` mentah
  secara native dan anti-leakage. Ini melengkapi TE manual di LightGBM.
- `loss_function="MultiClass"`, `depth=8`, `learning_rate=0.08`,
  `iterations≈3000`, early stopping 200, `auto_class_weights="SqrtBalanced"`.
- Ensemble: rata-rata probabilitas (bobot dipilih di OOF, mulai dari 0,5/0,5).

### 3.5 Post-processing untuk Macro F1
- Cari multiplier per kelas di probabilitas OOF ensemble. Fungsi
  `coordinate_search` sudah ada di nb 10; perluas kandidatnya ke 0,5–3,0.
- Terapkan multiplier yang sama ke probabilitas test.
- Ini biasanya langkah dengan lift Macro F1 terbesar per biaya komputasi.
  Hampir gratis karena hanya beroperasi pada OOF.

---

## 4. Urutan eksperimen (ablation)

Kerjakan berurutan, ukur Macro F1 OOF, dan pertahankan hanya langkah yang
menaikkan skor.

| # | Eksperimen | Ekspektasi | Biaya |
|---|---|---|---|
| E0 | Baseline nb 10 (skenario B) | 0,494 | – |
| E1 | E0 + multiplier per kelas | +0,01 s/d +0,03 | sangat murah |
| E2 | Skenario C + `iv_n_klaim`/`iv_amount_klaim`/`sc_n_selesai` | naik di retur & klaim_hilang | 1× CV |
| E3 | + TE entitas nb 09 (smoothed), buang `courier_id` mentah | **lift terbesar** (retur, konfirmasi, tertunda) | 1× CV |
| E4 | + frekuensi train+test & fitur deviasi | moderat | 1× CV |
| E5 | Bobot sqrt-balanced | kecil–moderat | 1× CV |
| E6 | Tuning ringan (num_leaves, min_child_samples, colsample) | kecil | beberapa CV |
| E7 | CatBoost + ensemble + multiplier ulang | +0,005 s/d +0,015 | 1× CV CatBoost |
| E8 | Seed averaging (3 seed) model terbaik | kecil, stabil | 3× CV |

Angka ekspektasi adalah perkiraan, bukan hasil. Validasi setiap langkah dengan CV.

---

## 5. Risiko dan catatan

- **Leakage TE:** jangan menghitung TE dari full train untuk baris train. Pakai
  OOF nb 09 untuk train dan full-train untuk test (sudah begitu di nb 09).
- **Skala TE OOF vs test:** referensi OOF memakai 80% data, sedangkan test 100%.
  Smoothing dan fitur frekuensi train+test meredam selisih ini.
- **RAM 8 GB:** gunakan float32, baca parquet per kolom, `del` + `gc.collect()`
  per fold, dan hindari menggabungkan frame penuh dua kali (lihat perbaikan nb 09).
- **Skenario C:** sah untuk kompetisi karena datanya tersedia di test. Untuk
  penerapan bisnis real-time (prediksi saat shipment dibuat), skenario A/B tetap
  relevan. Sebutkan trade-off ini di laporan.
- **Kelas `retur_pengirim`** paling sulit. Kalau setelah E3 masih < 0,30,
  pertimbangkan model biner khusus retur-vs-rest yang probabilitasnya dimasukkan
  sebagai fitur stacking.
