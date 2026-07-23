# DeepRestore AI – Intelligent Multimedia Noise Removal using Autoencoders

DeepRestore AI is a deep-learning system that **removes noise from images** using
convolutional autoencoders (DnCNN / U-Net), and then goes a step further: it can
take a **noisy scanned document**, restore it, **read the text with OCR**, and
**summarise and clean that text with NLP**. The project therefore spans two areas
end-to-end — **Computer Vision** (image restoration) and **Natural Language
Processing** (OCR + text summarisation/correction).

It ships as a full-stack app: a **TensorFlow/Keras** model, a **Flask REST API**,
and a **React** front-end.

---

## ✨ Features

- **Image denoising** with a DnCNN residual autoencoder trained on Gaussian,
  Poisson and salt-and-pepper noise.
- **Document restoration → OCR → NLP** pipeline: denoise a scanned page, extract
  text with **Tesseract OCR**, then **spell-correct**, **summarise** (Hugging
  Face Transformers) and **extract keywords** — a single end-to-end Computer
  Vision + NLP workflow.
- **Quantitative evaluation** with **PSNR** and **SSIM** on a held-out test set.
- **React web UI** with drag-and-drop upload, synthetic-noise testing, and a
  before/after comparison view.
- **Flask REST API** with rate limiting, file validation and a classical
  (OpenCV) denoising fallback.

## 🔬 How the Document pipeline works (CV → NLP)

```
 Noisy scan ──▶ [Autoencoder denoiser] ──▶ Restored image      (Computer Vision)
                                              │
                                              ▼
                        [Tesseract OCR] ──▶ Raw text
                                              │
             ┌────────────────────────────────┼────────────────────────────┐
             ▼                                ▼                             ▼
   [Spell correction]              [Transformer summariser]        [Keyword extraction]   (NLP)
     corrected text                     summary                       keywords
```

## 🗂 Project structure

```
DeepRestore-AI/
├── backend/                 # Flask API + model + training/eval scripts
│   ├── app.py               # REST API (/upload, /status, /restore-document)
│   ├── denoise_core.py      # Shared noise, DnCNN/U-Net models, data generator
│   ├── train.py             # Training entry point (TensorFlow/Keras)
│   ├── evaluate.py          # PSNR / SSIM evaluation on the test set
│   ├── document_pipeline.py # OCR + NLP (spell-correct, summarise, keywords)
│   ├── models/              # Trained weights (.h5) — git-ignored
│   └── requirements.txt
├── frontend/                # React.js UI
├── scripts/                 # Dataset download / preparation
│   └── prepare_dataset.py
├── data/                    # Datasets (git-ignored, created by scripts)
└── README.md
```

## 🚀 Quick start

### 1. Backend (API)

```bash
cd backend
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py            # serves http://localhost:5000
```

> **Document Restore (OCR) prerequisite:** the OCR step needs the **Tesseract**
> engine installed on your system (this is separate from the Python packages).
> - Windows: install from the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki),
>   then set `TESSERACT_CMD` to `tesseract.exe`, e.g.
>   `set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`
> - macOS: `brew install tesseract`   ·   Ubuntu: `sudo apt install tesseract-ocr`
>
> The first Document-Restore request downloads the summarisation model (~300 MB)
> and can take a minute; later requests are fast. Image denoising works without
> any of this.

### 2. Frontend (UI)

```bash
cd frontend
npm install
npm start                # opens http://localhost:3000
```

### 3. (Optional) Prepare data & train

Training is designed to run on **Google Colab (free GPU)**. See
[`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb), or locally:

```bash
python scripts/prepare_dataset.py     # downloads & splits the dataset
cd backend
python train.py                       # trains the denoising autoencoder
python evaluate.py                    # reports PSNR / SSIM on the test set
```

## 📊 Results

*(Filled in after training on the real dataset — PSNR/SSIM table + before/after grid.)*

| Noise type      | Level | PSNR (noisy) | PSNR (denoised) | SSIM (denoised) |
|-----------------|-------|--------------|-----------------|-----------------|
| Gaussian        | 0.10  | –            | –               | –               |
| Poisson         | 0.10  | –            | –               | –               |
| Salt & Pepper   | 0.10  | –            | –               | –               |

## 🧠 Tech stack

**ML/CV:** TensorFlow/Keras, OpenCV, NumPy, scikit-image
**NLP/OCR:** Tesseract (pytesseract), Hugging Face Transformers
**Backend:** Flask, Flask-CORS, Flask-Limiter
**Frontend:** React, Axios

## 📌 Roadmap

- [x] Image denoising autoencoder + REST API + React UI
- [x] Real dataset, train/val/test split, PSNR/SSIM evaluation *(train via Colab)*
- [x] Document restoration → OCR → NLP (spell-correct, summarise, keywords)
- [ ] Live deployment (frontend + backend)

## 📄 License

MIT
