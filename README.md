# NeuroData Pipeline

An enterprise-grade multi-agent platform for BCI and EEG data analysis, built with Streamlit and powered by Google Gemini via LangChain.

## Overview

Upload a neuroscience data file, configure preprocessing parameters, and let the pipeline handle the rest — from raw signal ingestion to an AI-powered interactive analysis session. The app automatically profiles data quality on load and exposes a chat interface backed by a Gemini agent that can plot, analyze, and explain your data.

## Features

- **Multi-format ingestion** — EDF, BDF, FIF, SET, VHDR, CNT, GDF (via MNE), CSV, TSV, XLSX, NPY, NPZ
- **Signal preprocessing** — bandpass filter, notch filter (50/60 Hz), average EEG re-reference
- **Parquet export** — processed data saved with configurable compression (snappy / gzip / zstd)
- **Automated data quality report** — LLM-generated QA summary on every new file load
- **AI chat agent** — Gemini-backed LangChain DataFrame agent for natural-language analysis and plot generation
- **Dark-themed visualizations** — plots saved as PNGs and rendered inline

## Project Structure

```
├── app.py              # Streamlit UI and main application logic
├── agent.py            # LangChain Gemini agents (analysis + QA report)
├── data_ingestion.py   # ETL engine — format detection, MNE processing, Parquet export
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image definition
├── .dockerignore
└── icon7.png           # App icon
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
   If running on GCP/Kubernetes with Workload Identity, Application Default Credentials are used automatically and no key is needed.

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   Open [http://localhost:8501](http://localhost:8501) in your browser.

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
|----------|-----------|
| Electrophysiology (MNE) | `.edf`, `.bdf`, `.fif`, `.set`, `.vhdr`, `.cnt`, `.gdf` |
| Tabular | `.csv`, `.tsv`, `.xlsx`, `.xls` |
| NumPy | `.npy`, `.npz` |
