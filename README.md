# Klasifikasi Hama Pertanian Ulat, Siput, dan Tikus Berbasis Citra Menggunakan KNN, SVM, dan Random Forest

## Nama Anggota
- TEGUH IMAM AZKARI : F1D021069
- VEBY FEBRIAN SAFITRI : F1D02410025
- KANAYA SALSABILA HUMAIRA : F2D02410061
- BAIQ FEBYLIA EKA NINGRUM : F1D02410109

---

## Project Overview

Project ini merupakan implementasi Pengolahan Citra Digital (PCD) untuk melakukan klasifikasi citra hama tanaman berdasarkan dataset gambar. Dataset yang digunakan terdiri dari tiga kelas utama, yaitu:

- `catterpillar`
- `snail`

Tujuan utama dari project ini adalah membandingkan beberapa tahapan preprocessing citra untuk melihat pengaruhnya terhadap hasil ekstraksi fitur dan performa model klasifikasi. Proses klasifikasi dilakukan dengan memanfaatkan fitur tekstur dari citra menggunakan metode **Gray Level Co-occurrence Matrix (GLCM)**, kemudian hasil fiturnya digunakan untuk melatih beberapa model machine learning.

Model klasifikasi yang digunakan dalam project ini adalah:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest

Project ini tidak hanya berfokus pada nilai akurasi akhir, tetapi juga pada pemilihan preprocessing yang sesuai, proses ekstraksi fitur, serta analisis hasil evaluasi model.

---

## Struktur Repository

```text
Project-PCD-Kelompok-16/
│
├── dataset/
│   ├── catterpillar/
│   └── snail/
│
├── percobaan/
│   ├── 01.preprocessing.ipynb
│   ├── 02.ekstraksi_fitur.ipynb
│   └── 03.klasifikasi.ipynb
│
├── templateProyek (1).ipynb
└── README.md
```

Keterangan:

- Folder `dataset/` berisi gambar asli yang dikelompokkan berdasarkan label kelas.
- Folder `percobaan/` berisi notebook utama untuk preprocessing, ekstraksi fitur, dan klasifikasi.
- File `templateProyek (1).ipynb` digunakan sebagai acuan struktur pengerjaan project.
- File `README.md` berisi dokumentasi project.

---

## Import Library

Library yang digunakan pada project ini menyesuaikan kebutuhan setiap tahap, mulai dari pembacaan gambar, pengolahan citra, ekstraksi fitur, hingga klasifikasi.

Beberapa library utama yang digunakan:

```python
import os
import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
```

---

## Load Data

Tahap pertama adalah membaca dataset gambar dari folder `dataset/`. Setiap subfolder pada folder dataset dianggap sebagai label kelas.

Label yang digunakan:

```text
catterpillar
snail
```

Dataset dibaca dengan mengambil seluruh file gambar berekstensi:

```text
.jpg, .jpeg, .png, .bmp
```

Contoh alur load data:

```python
DATASET_DIR = PROJECT_ROOT / "dataset"

data = []
labels = []
file_name = []

valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

for sub_folder in os.listdir(DATASET_DIR):
    sub_folder_path = DATASET_DIR / sub_folder

    if not sub_folder_path.is_dir():
        continue

    for filename in os.listdir(sub_folder_path):
        img_path = sub_folder_path / filename

        if img_path.suffix.lower() not in valid_extensions:
            continue

        img = cv.imread(str(img_path))

        if img is None:
            continue

        data.append(img)
        labels.append(sub_folder)
        file_name.append(filename)
```

Pada project ini, gambar kemudian diseragamkan ukurannya menjadi:

```python
TARGET_SIZE = (384, 256)
```

Resize dilakukan agar seluruh gambar memiliki ukuran yang sama sebelum masuk ke tahap preprocessing dan ekstraksi fitur.

---

## Data Understanding

Dataset terdiri dari tiga kelas citra hama tanaman. Setiap kelas disimpan dalam folder yang berbeda sehingga proses labeling dapat dilakukan secara otomatis berdasarkan nama folder.

Distribusi dataset:

| Label | Jumlah Data |
|---|---:|
| `catterpillar` | 70 gambar |
| `snail` | 70 gambar |
| **Total** | **140 gambar** |

Karakteristik umum dataset:

- Gambar berasal dari tiga jenis objek hama yang berbeda.
- Ukuran dan orientasi gambar dapat bervariasi.
- Background gambar tidak selalu seragam.
- Kondisi pencahayaan dapat berbeda pada setiap gambar.
- Beberapa objek memiliki tekstur yang cukup mirip dengan background, sehingga preprocessing diperlukan untuk memperjelas informasi citra.

Karena dataset memiliki jumlah data yang seimbang untuk setiap kelas, model tidak terlalu terdampak oleh masalah imbalance antar kelas.

---

## Data Preparation

### Data Augmentation

Pada project ini, data augmentation tidak menjadi tahap utama karena jumlah data pada tiap kelas sudah mencapai 70 gambar. Jumlah tersebut masih berada pada batas minimal yang dapat digunakan untuk percobaan klasifikasi sederhana.

Namun, augmentation tetap dapat diterapkan apabila ingin menambah variasi data, misalnya dengan:

- Rotasi gambar
- Flip horizontal
- Perubahan brightness
- Zoom atau cropping ringan

Augmentation dapat membantu model mengenali objek dalam berbagai posisi dan kondisi pencahayaan, tetapi pada project ini fokus utama diarahkan pada perbandingan preprocessing.

---

## Preprocessing

Preprocessing dilakukan untuk menyeragamkan citra, mengurangi noise, memperbaiki kualitas visual, dan menonjolkan karakteristik tertentu sebelum dilakukan ekstraksi fitur GLCM.

Pada notebook `01.preprocessing.ipynb`, terdapat lima skenario preprocessing:

| Preprocessing | Tahapan |
|---|---|
| `prepo1_resize+grayscale` | Resize → Grayscale |
| `prepo2_resize+grayscale+median` | Resize → Grayscale → Median Filter |
| `prepo3_resize+grayscale+median+equ` | Resize → Grayscale → Median Filter → Histogram Equalization |
| `prepo4_resize+grayscale+median+sobel` | Resize → Grayscale → Median Filter → Sobel |
| `prepo5_resize+grayscale+median+sobel+thresholding` | Resize → Grayscale → Median Filter → Sobel → Thresholding |

### Penjelasan Preprocessing

#### 1. Resize

Resize digunakan untuk menyamakan ukuran seluruh citra menjadi `384 x 256`. Hal ini diperlukan agar proses pengolahan citra dan ekstraksi fitur dapat berjalan konsisten.

#### 2. Grayscale

Grayscale digunakan untuk mengubah citra RGB/BGR menjadi citra keabuan. Karena metode GLCM bekerja pada hubungan intensitas piksel, citra grayscale lebih sesuai digunakan untuk ekstraksi fitur tekstur.

#### 3. Median Filter

Median filter digunakan untuk mengurangi noise tanpa terlalu merusak tepi objek. Filter ini cocok untuk citra hama karena dapat membersihkan gangguan kecil pada gambar, tetapi tetap mempertahankan bentuk utama objek.

#### 4. Histogram Equalization

Histogram equalization digunakan untuk meningkatkan kontras citra. Teknik ini membantu memperjelas perbedaan intensitas antara objek dan background, terutama pada gambar dengan pencahayaan yang kurang merata.

#### 5. Sobel

Sobel digunakan untuk mendeteksi tepi objek. Tahap ini membantu menonjolkan bentuk dan kontur objek hama, sehingga informasi struktur visual dapat lebih terlihat.

#### 6. Thresholding

Thresholding digunakan untuk mengubah citra hasil deteksi tepi menjadi citra biner. Tahap ini dapat membantu memisahkan area objek dan background, tetapi juga berpotensi menghilangkan detail tekstur jika nilai ambang tidak sesuai.

---

## Skenario Percobaan

Project ini menggunakan lima jenis preprocessing. Untuk melihat pengaruh penambahan preprocessing terhadap performa model, percobaan dapat dibagi menjadi beberapa skenario:

| Percobaan | Preprocessing yang Digunakan |
|---|---|
| Percobaan 1 | `prepo1`, `prepo2` |
| Percobaan 2 | `prepo1`, `prepo2`, `prepo3`, `prepo4` |
| Percobaan 3 | `prepo1`, `prepo2`, `prepo3`, `prepo4`, `prepo5` |

Dengan skenario ini, setiap hasil klasifikasi dapat dibandingkan untuk mengetahui preprocessing mana yang paling sesuai terhadap dataset hama tanaman.

---

## Feature Extraction

Tahap ekstraksi fitur dilakukan menggunakan metode **Gray Level Co-occurrence Matrix (GLCM)**.

GLCM digunakan untuk mengambil informasi tekstur dari citra berdasarkan hubungan spasial antar piksel. Pada project ini, fitur GLCM dihitung pada empat arah sudut:

- 0°
- 45°
- 90°
- 135°

Implementasi pada notebook menggunakan `graycomatrix` dan `graycoprops` dari `skimage.feature`.

Fitur yang diekstraksi:

| Fitur | Keterangan |
|---|---|
| Contrast | Mengukur perbedaan intensitas antara piksel bertetangga |
| Homogeneity | Mengukur keseragaman tekstur citra |
| Dissimilarity | Mengukur tingkat perbedaan antar piksel |
| Entropy | Mengukur kompleksitas atau ketidakteraturan tekstur |
| ASM | Mengukur keseragaman distribusi nilai GLCM |
| Energy | Mengukur energi atau kekuatan pola tekstur |
| Correlation | Mengukur hubungan linear antar piksel |

Contoh struktur fungsi ekstraksi fitur:

```python
def extract_glcm_features(image):
    angles = [0, 45, 90, 135]
    features = {}

    for angle in angles:
        matriks = glcm(image, angle)

        features[f"Contrast{angle}"] = contrast(matriks)
        features[f"Homogeneity{angle}"] = homogenity(matriks)
        features[f"Dissimilarity{angle}"] = dissimilarity(matriks)
        features[f"Entropy{angle}"] = entropyGlcm(matriks)
        features[f"ASM{angle}"] = ASM(matriks)
        features[f"Energy{angle}"] = energy(matriks)
        features[f"Correlation{angle}"] = correlation(matriks)

    return features
```

Hasil ekstraksi fitur disimpan ke dalam folder:

```text
hasil_ekstraksi/
```

File CSV yang dihasilkan:

```text
hasil_ekstraksi_prepo1.csv
hasil_ekstraksi_prepo2.csv
hasil_ekstraksi_prepo3.csv
hasil_ekstraksi_prepo4.csv
hasil_ekstraksi_prepo5.csv
```

---

## Feature Selection

Feature selection dilakukan untuk memilih fitur yang paling relevan dan mengurangi fitur yang terlalu berkorelasi satu sama lain.

Pada template project, feature selection dapat dilakukan menggunakan correlation. Tahap ini membantu mengurangi redundansi fitur sehingga proses klasifikasi menjadi lebih efisien.

Contoh pendekatan correlation:

```python
correlation = hasilEkstrak.drop(columns=["Label", "Filename"]).corr()
```

Fitur yang memiliki korelasi terlalu tinggi dapat dipertimbangkan untuk dihapus agar model tidak mempelajari informasi yang berulang.

---

## Splitting Data

Dataset hasil ekstraksi fitur dibagi menjadi data training dan data testing.

Perbandingan data yang dapat digunakan:

```text
80% data training
20% data testing
```

Contoh kode:

```python
X = df.drop(columns=["Label", "Filename", "Preprocessing"])
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=90,
    stratify=y
)
```

Penggunaan `stratify=y` bertujuan agar distribusi label pada data training dan testing tetap seimbang.

---

## Normalization

Normalisasi dilakukan agar seluruh fitur memiliki skala nilai yang lebih seragam. Hal ini penting terutama untuk model seperti KNN dan SVM yang sensitif terhadap jarak atau skala fitur.

Metode normalisasi yang dapat digunakan:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Modeling

Model klasifikasi yang digunakan pada project ini terdiri dari tiga algoritma:

### 1. K-Nearest Neighbors (KNN)

KNN mengklasifikasikan data berdasarkan kedekatan jarak dengan data training. Model ini cukup sederhana, tetapi sangat dipengaruhi oleh skala fitur sehingga membutuhkan normalisasi.

### 2. Support Vector Machine (SVM)

SVM bekerja dengan mencari hyperplane terbaik untuk memisahkan kelas. Model ini cocok untuk data berdimensi fitur cukup banyak seperti hasil ekstraksi GLCM.

### 3. Random Forest

Random Forest merupakan model ensemble berbasis decision tree. Model ini cukup stabil terhadap variasi fitur dan dapat menangani hubungan non-linear antar fitur.

Contoh training model:

```python
knn = KNeighborsClassifier()
svm = SVC()
rf = RandomForestClassifier(random_state=90)

knn.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)
```

---

## Evaluation

Evaluasi model dilakukan untuk mengetahui performa klasifikasi pada setiap preprocessing.

Metrik evaluasi yang digunakan:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Contoh kode evaluasi:

```python
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

Format tabel hasil evaluasi:

| Preprocessing | Model | Accuracy | Precision | Recall | F1-Score |
|------------|------------|------------|------------|------------|------------|
| prepo1 | KNN | 83.33% | 84.88% | 83.33% | 82.78% |
| prepo1 | SVM | **90.48%** | **91.84%** | **90.48%** | **90.25%** |
| prepo1 | Random Forest | 80.95% | 80.98% | 80.95% | 80.77% |
| prepo2 | KNN | **95.24%** | **95.60%** | **95.24%** | **95.19%** |
| prepo2 | SVM | 90.48% | 90.73% | 90.48% | 90.39% |
| prepo2 | Random Forest | 73.81% | 74.83% | 73.81% | 73.94% |
| prepo3 | KNN | 83.33% | 83.70% | 83.33% | 83.07% |
| prepo3 | SVM | **88.10%** | **90.15%** | **88.10%** | **87.70%** |
| prepo3 | Random Forest | 73.81% | 74.04% | 73.81% | 73.88% |
| prepo4 | KNN | 64.29% | 64.07% | 64.29% | 64.14% |
| prepo4 | SVM | **78.57%** | **78.49%** | **78.57%** | **78.48%** |
| prepo4 | Random Forest | 66.67% | 66.67% | 66.67% | 66.67% |
| prepo5 | KNN | 64.29% | 64.56% | 64.29% | 64.39% |
| prepo5 | SVM | 66.67% | 66.67% | 66.67% | 66.67% |
| prepo5 | Random Forest | **73.81%** | **74.04%** | **73.81%** | **73.88%** |

Nilai pada tabel dapat diisi setelah notebook `03.klasifikasi.ipynb` dijalankan.

### Analisis Evaluasi
Analisis:

- Berdasarkan hasil pengujian, preprocessing 1 menunjukkan bahwa informasi dasar dari citra grayscale sudah mampu merepresentasikan karakteristik tekstur objek dengan sangat baik. Hal ini terlihat dari tingginya performa model, terutama SVM yang mencapai akurasi 90.48%. Hasil tersebut menunjukkan bahwa perbedaan tekstur antara kelas Caterpillar dan Snail masih dapat ditangkap dengan baik oleh fitur GLCM tanpa memerlukan pengolahan citra tambahan.
- Pada preprocessing 2, performa model mengalami peningkatan dan menghasilkan hasil terbaik pada penelitian ini. Penggunaan median filter membantu mengurangi noise pada citra tanpa menghilangkan struktur utama objek, sehingga fitur tekstur yang diekstraksi menjadi lebih stabil dan lebih representatif. Kombinasi preprocessing ini dengan algoritma KNN menghasilkan akurasi tertinggi sebesar 95.24%.
- Preprocessing 3 tidak memberikan peningkatan yang lebih baik dibanding preprocessing 2. Meskipun histogram equalization mampu meningkatkan kontras citra, perubahan distribusi intensitas piksel dapat memengaruhi pola tekstur asli yang digunakan oleh GLCM. Akibatnya, performa klasifikasi mengalami sedikit penurunan dibandingkan preprocessing terbaik.
- Pada preprocessing 4, penambahan operator Sobel menghasilkan penurunan performa yang cukup signifikan. Informasi tepi yang dihasilkan memang dapat memperjelas bentuk objek, namun fitur tekstur yang menjadi fokus utama ekstraksi GLCM justru menjadi kurang dominan. Hal ini menyebabkan kemampuan model dalam membedakan kedua kelas menjadi menurun.
- Preprocessing 5 menghasilkan performa yang lebih rendah dibandingkan preprocessing sebelumnya. Penggunaan thresholding mengubah citra menjadi lebih sederhana dengan menghilangkan banyak variasi tingkat keabuan yang sebenarnya mengandung informasi tekstur penting. Karena GLCM sangat bergantung pada hubungan antar tingkat intensitas piksel, hilangnya informasi tersebut menyebabkan kualitas fitur yang diekstraksi menjadi berkurang dan berdampak pada penurunan akurasi klasifikasi.

---

## Cara Menjalankan Project

1. Clone repository:

```bash
git clone https://github.com/Cannaset/Project-PCD-Kelompok-16.git
cd Project-PCD-Kelompok-16
```

2. Install library yang dibutuhkan:

```bash
pip install numpy pandas matplotlib opencv-python scikit-image scipy scikit-learn
```

3. Jalankan notebook secara berurutan:

```text
percobaan/01.preprocessing.ipynb
percobaan/02.ekstraksi_fitur.ipynb
percobaan/03.klasifikasi.ipynb
```

Urutan notebook penting karena:

- Notebook preprocessing menghasilkan folder `preprocessing_output/`
- Notebook ekstraksi fitur menghasilkan folder `hasil_ekstraksi/`
- Notebook klasifikasi menggunakan CSV dari hasil ekstraksi fitur

---

## Output Project

Output utama dari project ini adalah:

```text
preprocessing_output/
hasil_ekstraksi/
classification report
confusion matrix
tabel perbandingan akurasi model
```

Folder `preprocessing_output/` berisi hasil citra dari setiap tahapan preprocessing. Folder `hasil_ekstraksi/` berisi file CSV fitur GLCM yang digunakan pada tahap klasifikasi.

## Kesimpulan

Penelitian ini mengevaluasi pengaruh lima teknik preprocessing citra terhadap klasifikasi gambar Caterpillar dan Snail menggunakan ekstraksi fitur GLCM dengan algoritma KNN, SVM, dan Random Forest. Berdasarkan hasil pengujian, tahapan preprocessing terbukti berpengaruh terhadap performa klasifikasi.

Konfigurasi terbaik diperoleh pada Preprocessing 2, yaitu Resize → Grayscale → Median Filter, dengan algoritma KNN. Kombinasi ini menghasilkan accuracy sebesar 95,24%, precision sebesar 95,60%, recall sebesar 95,24%, dan F1-score sebesar 95,19%. Hasil tersebut menunjukkan bahwa median filter mampu mengurangi noise pada citra tanpa menghilangkan informasi tekstur penting yang dibutuhkan oleh GLCM.

Citra grayscale dasar juga sudah mampu merepresentasikan karakteristik tekstur objek dengan baik. Namun, penambahan preprocessing seperti Histogram Equalization, Sobel, dan Thresholding tidak meningkatkan performa secara signifikan, bahkan cenderung menurunkan akurasi. Hal ini menunjukkan bahwa transformasi citra yang terlalu berlebihan dapat mengubah atau menghilangkan detail tekstur penting.

Secara keseluruhan, metode terbaik dalam penelitian ini adalah kombinasi GLCM, Preprocessing 2, dan KNN untuk membedakan citra Caterpillar dan Snail.
