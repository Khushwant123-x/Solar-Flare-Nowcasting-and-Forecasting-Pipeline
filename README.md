# Aditya-L1 Solar Flare Nowcasting and Forecasting Pipeline

An end-to-end telemetry analytics pipeline for detecting and forecasting solar flares using X-ray spectrometers on board ISRO's **Aditya-L1** mission: **SoLEXS** (Soft X-ray Spectrometer) and **HEL1OS** (Hard X-ray Spectrometer).

This repository contains modular Python scripts to ingest FITS archives, execute real-time flare detection (nowcasting) with cross-channel validation, predict flare occurrences with lead time (forecasting), evaluate ML models (XGBoost & LSTM), and display an interactive dashboard built on Streamlit and Plotly.

---

## ☀️ Project Context & Physics

Solar flares are violent eruptions of electromagnetic radiation from the Sun's active regions. They pose a significant hazard to satellite operations, radio communication, and electrical grids.
- **SoLEXS (Soft X-rays)**: Monitors lower-energy (thermal) emission. During a flare, thermal emission rises slightly slower and peaks later as the coronal loops heat up.
- **HEL1OS (Hard X-rays)**: Monitors higher-energy (non-thermal) emission. Non-thermal emission rises rapidly and peaks first during the impulsive phase of particle acceleration.
- **Neupert Effect**: The pipeline exploits the physical time-lag between the non-thermal hard X-ray peak (HEL1OS) and the thermal soft X-ray peak (SoLEXS) to improve both forecasting lead times and nowcasting validation.

---

## 🛠️ Pipeline Architecture

The pipeline is built with a modular, scalable structure:

```
solarflare_ai/
│
├── data/
│   └── raw/               # Contains zipped raw FITS data files from ISSDC
│
├── notebooks/
│   └── 01_explore.ipynb   # Explortory analysis workspace
│
├── src/
│   ├── ingestion.py       # Reads zip archives, handles endianness, aligns time-series, generates synthetic data
│   ├── nowcasting.py      # Detects flares via rolling stats and cross-channel validation, classifications
│   ├── forecasting.py     # Sliding window feature engineering, labels target, trains XGBoost & LSTM
│   └── evaluation.py      # Computes performance stats (TPR, FAR, F1, lead time), confusion matrix, ROC
│
├── app.py                 # Streamlit dashboard & live simulation playback interface
├── requirements.txt       # Project dependencies
└── README.md              # Project methodology and setup guide
```

### 1. Data Ingestion (`src/ingestion.py`)
- Reads `.lc.gz` and `.fits` formats directly from ZIP archives to prevent disk inflation.
- Parses `TIME` timestamps (Unix epoch) and `ISOT` UTC strings.
- Converts big-endian FITS arrays to native little-endian formats to avoid compiler errors.
- Outer-joins both spectrometers and resamples data to a uniform **10-second grid** using linear interpolation.
- Includes a **Synthetic Flare Generator** that models flare profiles (fast rise, slower exponential decay, and channel time lag) for robust model training.

### 2. Nowcasting Module (`src/nowcasting.py`)
- Computes a 30-minute rolling baseline ($\mu$) and standard deviation ($\sigma$).
- Alerts if the flux exceeds $\mu + k \cdot \sigma$ (where $k$ is tunable).
- **Cross-Channel Validation**: Cross-validates detections by requiring both SoLEXS and HEL1OS to spike within $\pm 5$ minutes of each other. This eliminates isolated false alarms from single-detector instrument glitches or cosmic rays.
- Classifies flare magnitude using log-amplitude scale equivalencies representing GOES levels (**A**, **B**, **C**, **M**, and **X**).

### 3. Forecasting Module (`src/forecasting.py`)
- Engineered features: Rolling averages, standard deviations, rate of change (first derivative), soft-to-hard X-ray ratios, and cumulative trends over 5, 10, and 15-minute sliding windows.
- Targets: Predicts if a nowcasted flare will peak or occur within the next $N$ minutes (default 15).
- Models:
  1. **XGBoost Classifier**: Standard gradient boosting trained on sliding window tabular features.
  2. **LSTM (Long Short-Term Memory)**: Recurrent neural network trained on sequential raw flux and derivatives over a 5-minute lookback sequence.
- Estimates forecasted lead times using a probability decay heuristic.

### 4. Evaluation Module (`src/evaluation.py`)
- Calculates TPR (Recall), FAR (False Alarm Rate), Precision, and F1.
- Tracks average lead time (trigger-to-peak difference in minutes).
- Outputs interactive Plotly Heatmap Confusion Matrices and ROC curves.

---

## 🚀 Setup & Installation

### Prerequisiutes
Ensure Python 3.9+ is installed.

### 1. Clone & Initialize Environment
Set up a python virtual environment and install the required dependencies.

```bash
# Navigate to the workspace
cd solarflare_ai

# Initialize virtual environment if not already active
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Dashboard
Start the Streamlit application:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Dashboard Walkthrough & Interactive Features

- **Data Selection**: Toggle between **Synthetic Generator** (to test major flares) and **Real Data** (June 14-16, 2026).
- **Playback Slider**: Drag the timeline slider to simulate real-time telemetry playback. Visual alert banners dynamically transition from **Green (Normal)** to **Yellow (Flare Forecasted)** and **Red (Active Flare Nowcast)**.
- **Plotly Light Curves**: Interactive chart overlaying SoLEXS and HEL1OS counts, nowcast detections, and forecast triggers with toggleable log-scale.
- **Event Catalogue Tab**: Houses the logs of all detected flares (timestamps, durations, peak fluxes, classes, and confidence scores).
- **Predictive Performance Tab**: Displays model metrics (TPR, Precision, Recall, average lead time) along with interactive Confusion Matrices and ROC curves.
