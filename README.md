# Tugas 05 - Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

Eksperimen transfer learning dengan ResNet-18 pada subset 10 kelas Oxford-IIIT Pet
(5 ras kucing + 5 ras anjing, ~2000 citra, split stratified 70/15/15, seed 42).

## Kondisi eksperimen

| Kode | Strategi |
|------|----------|
| B1 | From scratch (bobot acak) |
| B2 | Feature extraction (backbone beku, latih `fc`) |
| B3 | Fine-tune `layer4` + `fc`, differential LR |
| B4 | B3 tanpa augmentasi (pembanding B3) |
| B5 | MobileNetV3-Small (bonus) |

Protokol seragam di semua kondisi: input 224x224, normalisasi ImageNet, batch size 32,
AdamW (weight decay 1e-2), CrossEntropyLoss, early stopping (patience 4), seed 42.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# torch/torchvision build GPU (CUDA 12.8); sesuaikan di https://pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Menjalankan

Buka `Tugas05_Transfer_Learning.ipynb` dan jalankan semua sel dari atas ke bawah.
Dataset Oxford-IIIT Pet terunduh otomatis ke `data/` pada eksekusi pertama.

## Luaran

- `metrics.csv`, `metrics.json` - metrik mentah tiap kondisi
- `outputs/loss_curves.png` - kurva training vs validation loss (B1, B2, B3)
- `outputs/failure_analysis.png` - sampel test set yang salah prediksi
- `laporan_tugas05.docx` - laporan analisis (Bagian C)

## Struktur

```
Tugas05_Transfer_Learning.ipynb   notebook eksperimen
laporan.txt                       teks laporan Bagian C
laporan_tugas05.docx              laporan terformat
metrics.csv / metrics.json        log metrik
outputs/                          gambar hasil
requirements.txt
```
