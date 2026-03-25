<p align="center">
  <img src="icon.png" alt="NeuroData Pipeline" width="100"/>
</p>

# NeuroData Pipeline

An enterprise-grade multi-agent MLOps platform for BCI and EEG data analysis. Built with Streamlit, powered by Google Gemini via LangChain, and equipped with XGBoost model training tracked end-to-end by MLflow.

## Overview

Upload a neuroscience data file, configure preprocessing parameters, and let the pipeline handle the rest — from raw signal ingestion and automated data quality profiling, to AI-powered analysis, ML model training, and full experiment tracking. Everything that runs is logged.

## Features

| Feature | Description |
|---------|-------------|
| **Multi-format ingestion** | EDF, BDF, FIF, SET, VHDR, CNT, GDF (via MNE), CSV, TSV, XLSX, NPY, NPZ |
| **Signal preprocessing** | Bandpass filter, notch filter (50/60 Hz), average EEG re-reference |
| **Parquet export** | Processed data saved with configurable compression (snappy / gzip / zstd) |
| **Automated QA report** | LLM-generated data quality summary on every file load |
| **AI chat agent** | Gemini-backed LangChain agent for natural-language analysis and plot generation |
| **XGBoost model training** | Auto-detects classification vs. regression, trains and evaluates the model |
| **MLflow experiment tracking** | Every ETL run and model training run is fully logged with parameters and metrics |
| **Dark-themed visualizations** | Plots saved as PNGs and rendered inline in the UI |

## App Tabs

After uploading and processing a file, the app exposes five tabs:

- **💬 Analysis Agent Chat** — Chat with a Gemini agent that can analyze, plot, and explain your data in natural language
- **🤖 Model Training (XGBoost)** — Select features and a target column, train a model, and view feature importances
- **📊 Data Quality Report** — Auto-generated QA summary with outlier flags, missing value counts, and dtype checks
- **👁️ Signal Preview** — Interactive multi-channel line chart with row-count slider
- **📋 Data Preview** — Raw DataFrame table (first 200 rows)

## MLflow — Experiment Tracking

Every action in the pipeline is logged to MLflow automatically:

### ETL Run (on every file upload)
Logged when you click **Execute Preprocessing**:
- Parameters: `low_freq`, `high_freq`, `notch`, `apply_reference`, `compression`
- Metrics: `num_rows`, `num_columns`

### Model Training Run (XGBoost tab)
Logged when you click **Train XGBoost & Log to MLflow**:
- Parameters: `task_type` (classification / regression), `n_estimators`, `max_depth`, `learning_rate`, `test_size`, `num_features`, `n_classes`
- Metrics: `accuracy` (classification) or `r2_score` (regression)
- Artifact: trained XGBoost model saved as `xgb_model`

### Viewing the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://localhost:5000](http://localhost:5000) to browse all experiments, compare runs, and inspect logged models.

> MLflow stores data locally in `mlflow.db` (SQLite) and `mlruns/`. These are excluded from git via `.gitignore`.

## Project Structure

```
├── app.py              # Streamlit UI — layout, tabs, ETL trigger, MLflow ETL logging
├── agent.py            # LangChain Gemini agents (analysis chat + QA report)
├── ml_engine.py        # XGBoost training engine — auto task detection, MLflow logging
├── data_ingestion.py   # ETL engine — format detection, MNE processing, Parquet export
├── requirements.txt    # Python dependencies (pinned)
├── Dockerfile          # Container image definition
├── .dockerignore
├── .streamlit/
│   └── config.toml     # Streamlit theme config (primary color)
└── icon.png            # App icon (PNG with transparent background)
```

## Setup

### Local (Python)

1. **Clone the repo and create a virtual environment:**
   ```bash
   git clone https://github.com/Talpeleggg/nuero-agent.git
   cd nuero-agent
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your API key:**
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```
   If running on GCP/Kubernetes with Workload Identity, Application Default Credentials (ADC) are used automatically and no key is needed.

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   Open [http://localhost:8501](http://localhost:8501) in your browser.

5. **View MLflow experiments (optional):**
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
   Open [http://localhost:5000](http://localhost:5000).

### Docker

```bash
docker build -t neurodata-pipeline .
docker run -p 8501:8501 --env-file .env neurodata-pipeline
```

## Authentication

The app supports two authentication modes for the Gemini API:

| Mode | When used |
|------|-----------|
| GCP Application Default Credentials (ADC) | Kubernetes / Cloud Run / local `gcloud auth` |
| `GOOGLE_API_KEY` in `.env` | Local development fallback |

## Supported File Formats

| Category | Extensions |
|----------|------------|
| Electrophysiology (MNE) | `.edf`, `.bdf`, `.fif`, `.set`, `.vhdr`, `.cnt`, `.gdf` |
| Tabular | `.csv`, `.tsv`, `.xlsx`, `.xls` |
| NumPy | `.npy`, `.npz` |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| AI Agent | LangChain + Google Gemini 2.5 Flash |
| ML Model | XGBoost (classifier + regressor) |
| Experiment Tracking | MLflow |
| Signal Processing | MNE-Python |
| Data | Pandas, NumPy, PyArrow |
| Containerization | Docker |
