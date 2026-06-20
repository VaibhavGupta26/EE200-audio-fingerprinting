# EE200 Signals, Systems & Networks Project: Song Identification System

This repository contains the complete implementation for **Question 3 (Q3A & Q3B)** of the EE200 course project, building a Shazam-style audio fingerprinting and matching system.

## 📁 Repository Structure
```text
audio_fingerprinting/
├── data/
│   ├── songs/              # Bundled library database of 50 MP3 songs (unmodified)
│   └── queries/            # User query/test clips (empty by default)
├── src/
│   ├── spectrogram.py      # Audio loading, DFT magnitude, and manual STFT implementation
│   ├── fingerprint.py      # 2D peak extraction (constellation map) and database indexing
│   ├── matcher.py          # Query matching engine using time-offset histograms
│   └── robustness.py       # Noise injection, pitch/speed shifts, and sweep experiments
├── app/
│   └── app.py              # Interactive Streamlit application (Single-clip & Batch modes)
├── report/
│   ├── images/             # Folder containing plots generated from the experiments
│   └── Q3_report.pdf       # Compiled final PDF report containing all plots & explanations
├── requirements.txt        # Python package dependencies
├── generate_report.py      # Script to automate plot generation and compile the PDF report
└── README.md               # Setup and usage documentation (this file)
```

---

## 🛠️ Setup and Installation

### 1. Requirements
Ensure Python 3.8+ is installed on your system. Install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Audio Backend (Windows)
This application uses `librosa` and `soundfile` to load and decode audio. Standard MP3 decoding is supported natively out-of-the-box by the bundled `libsndfile` backend inside the `soundfile` package. If you encounter any issues loading MP3 files, ensure your system has audio codec decoders or `ffmpeg` installed.

---

## 🚀 How to Run

### 1. Interactive Web Application (Q3B)
Launch the Streamlit app locally:
```bash
streamlit run app/app.py
```
This starts the local web server. The app features:
*   **Auto-indexing:** If `database.pkl` is missing, it will automatically build the song index from `data/songs/` on startup.
*   **Single-Clip Mode:** Upload a query clip. The app will classify it and render:
    1. The query spectrogram.
    2. The constellation map of detected spectral peaks.
    3. The offset alignment histogram of the top matching candidate.
*   **Batch Mode:** Upload a batch of query files, click classify, and download the resulting `results.csv` in the exact auto-grader format.

### 2. Run Robustness Sweeps (Q3A Experiments)
To execute the experiments comparing Single Peaks vs. Paired Hashes under AWGN and pitch/speed shifts:
```bash
python -m src.robustness
```
This builds the database (if missing), runs the sweeps on a random subset of songs, and saves the accuracy curve plots to `report/images/`.

### 3. Compile the PDF Report (Q3A Report)
To generate the final report:
```bash
python generate_report.py
```
This generates all required visualizations (full DFT, resolution tradeoff spectrograms, constellation maps, offset histograms, and robustness curves) and compiles them alongside explanations into `report/Q3_report.pdf`.
