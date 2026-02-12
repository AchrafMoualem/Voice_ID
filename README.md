# 🎧 Speaker Identification & Semantic Speech Analysis

> An end-to-end Deep Learning + NLP web application that combines **Speaker Recognition** and **Semantic Speech Understanding** in a production-ready Flask architecture.

---

## Overview

This project integrates **Audio Deep Learning** and **Natural Language Processing** into a unified intelligent pipeline.

From a single audio recording, the system can:

| Capability | Technology |
|---|---|
| 🎙️ Identify the Speaker | CNN-based model |
| 📝 Transcribe Speech to Text | OpenAI Whisper |
| 🧠 Extract Keywords | KeyBERT |
| 📄 Generate an Automatic Summary | LSA – Sumy |

> It demonstrates how **Deep Learning models and NLP pipelines can be orchestrated together in a clean, modular web application.**

---

## Key Features

- End-to-end audio-to-semantic analysis pipeline
- CNN-based speaker classification using MFCC features
- State-of-the-art Whisper transcription
- Semantic keyword extraction
- Extractive summarization
- Modular Flask architecture
- Production-ready structure

---

## System Architecture

```
Audio Input
    ↓
Audio Preprocessing  ·  Librosa + MFCC
    ↓
CNN Speaker Classification  ·  TensorFlow / Keras
    ↓
Whisper Transcription  ·  OpenAI Whisper
    ↓
Keyword Extraction  ·  KeyBERT
    ↓
Summarization  ·  LSA (Sumy)
    ↓
Flask Web Interface
```

---

## Technologies Used

### 🎧 Audio & Deep Learning

| Library | Role |
|---|---|
| **TensorFlow / Keras** | CNN speaker classification model |
| **Librosa** | MFCC feature extraction |
| **NumPy** | Feature normalization |

### 🧠 NLP & Speech

| Library | Role |
|---|---|
| **OpenAI Whisper** | Speech-to-text transcription |
| **KeyBERT** | Keyword extraction |
| **Sumy (LSA)** | Extractive summarization |
| **NLTK** | Sentence tokenization |

### 🌐 Web Application

| Tool | Role |
|---|---|
| **Flask** | Backend framework |
| **HTML / CSS** | Frontend interface |

---

## Project Structure

```
├── app/
│   ├── __init__.py             # Flask app factory & model loading
│   ├── routes.py               # HTTP routes
│   ├── services/
│   │   ├── speaker_service.py
│   │   ├── transcription_service.py
│   │   ├── keyword_service.py
│   │   └── summarization_service.py
│   ├── utils/
│   │   └── audio_processing.py
│   └── templates/
│       ├── home.html
│       └── index.html
│
├── models/
│   └── final_model.h5
│
├── scripts/
│   ├── train.py
│   └── wav_transform.py
│
├── predict.py
├── config.py
├── run.py
└── requirements.txt
```

---

## How It Works

### 1 — Speaker Identification

Audio is converted to WAV format if required, then MFCC features are extracted using Librosa. Features are normalized using pre-saved mean & standard deviation values, and the CNN model predicts the speaker class.

### 2 — Speech Transcription

Audio is passed directly to Whisper, which performs automatic language detection and outputs clean text transcription.

### 3 — Keyword Extraction

KeyBERT identifies semantically meaningful keywords and returns the top N most relevant words and phrases from the transcription.

### 4 — Automatic Summarization

LSA (Latent Semantic Analysis) selects the most informative sentences from the transcription to produce a concise extractive summary.

---

## Installation & Usage

### 1 — Clone the Repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### 2 — Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 4 — Add Required Model Files

Ensure the following files exist before running:

```
models/final_model.h5
label_mapping.npy
mean.npy
std.npy
```

### 5 — Run the Application

```bash
python run.py
```

### 6 — Open in Your Browser

```
http://localhost:5000/
```

---

## Design Decisions

**Heavy models loaded once at startup** — Whisper, KeyBERT, and the CNN model are initialized once when the app boots, avoiding per-request overhead.

**Clean modular architecture** — The codebase separates concerns clearly:
- `services/` → Business logic
- `utils/` → Reusable helper functions
- `scripts/` → Offline training tools

**Centralized configuration** — All settings live in `config.py` for easy environment management.

**Clear backend/frontend separation** — Flask serves as a pure API layer; HTML/CSS handles presentation independently.

---

## Learning Outcomes

This project demonstrates practical experience across:

- End-to-end ML system integration
- Audio feature engineering (MFCC)
- CNN-based speaker classification
- Modern speech models (Whisper)
- NLP semantic processing
- Production-ready Flask architecture

---

## Future Improvements

- [ ] Add speaker verification (1:1 matching)
- [ ] Real-time streaming transcription
- [ ] Improve transcription formatting
- [ ] Dockerize the application
- [ ] Deploy to AWS / GCP / Azure
- [ ] Replace LSA with Transformer-based summarization
- [ ] Add confidence scores & analytics dashboard

---

## Author

**Achraf Moualem**  
AI & Data Science Student  
*Interested in AI Engineering, Speech Processing & Generative AI*
