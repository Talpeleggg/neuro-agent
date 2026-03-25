<p align="center">
  <img src="icon.png" alt="NeuroData Pipeline" width="110"/>
</p>

<h1 align="center">NeuroData Pipeline</h1>

<p align="center">
  An enterprise-grade multi-agent MLOps platform for BCI and EEG data analysis.<br/>
  Built with <strong>Streamlit</strong>, powered by <strong>Google Gemini 2.5 Flash</strong> via LangChain,<br/>
  with <strong>XGBoost</strong> model training tracked end-to-end by <strong>MLflow</strong>.
</p>

---

## What it does

Upload any neuroscience data file, configure preprocessing parameters, and the pipeline handles everything automatically:

1. **Ingests** your file (EDF, CSV, NPY, and more)
2. **Preprocesses** the signal (bandpass filter, notch filter, average re-reference)
3. **Exports** the cleaned data to a compressed Parquet file
4. **Runs an AI Data Quality Agent** that profiles your dataset and flags issues
5. **Lets you chat** with an AI Analysis Agent to explore, plot, and explain your data
6. **Trains an XGBoost model** on any columns you choose and logs everything to MLflow

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-format ingestion** | EDF, BDF, FIF, SET, VHDR, CNT, GDF (via MNE), CSV, TSV, XLSX, NPY, NPZ |
| **Signal preprocessing** | Bandpass filter, notch filter (50/60 Hz), average EEG re-reference |
| **Parquet export** | Processed data saved with configurable compression (snappy / gzip / zstd) |
| **Automated QA report** | AI-generated data quality summary on every file load |
| **AI Analysis Agent** | Gemini-backed LangChain agent — chat, plot, and explain data in natural language |
| **XGBoost model training** | Auto-detects classification vs. regression, trains and evaluates the model |
| **MLflow experiment tracking** | Every ETL run and model training run is fully logged with parameters and metrics |
| **Dark-themed visualizations** | Plots saved as PNGs and rendered inline in the UI |

---

## App Tabs

After uploading and processing a file, the app exposes five tabs:

| Tab | What it does |
|-----|--------------|
| **🤖 Model Training (XGBoost)** | Select feature and target columns, train a model, view feature importances |
| **💬 Analysis Agent Chat** | Chat with the Gemini AI agent — ask for plots, stats, ERP analysis, etc. |
| **📊 Data Quality Report** | Auto-generated QA report with outlier flags, missing value counts, dtype checks |
| **👁️ Signal Preview** | Interactive multi-channel line chart with adjustable row count |
| **📋 Data Preview** | Raw DataFrame table (first 200 rows) |

---

## Supported File Formats

| Category | Extensions |
|----------|------------|
| Electrophysiology (MNE) | `.edf`, `.bdf`, `.fif`, `.set`, `.vhdr`, `.cnt`, `.gdf` |
| Tabular | `.csv`, `.tsv`, `.xlsx`, `.xls` |
| NumPy | `.npy`, `.npz` |

---

## Setup — Local (Python)

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/Talpeleggg/nuero-agent.git
cd nuero-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your Gemini API key

The AI agents (Analysis Chat + Data Quality Report) use Google Gemini 2.5 Flash.
You need an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

> **No key needed** if you are running on GCP/Kubernetes with Workload Identity — Application Default Credentials (ADC) are used automatically.

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. (Optional) View MLflow experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://localhost:5000](http://localhost:5000) to browse all experiments, compare runs, and inspect logged models.

---

## Setup — Docker

```bash
docker build -t neurodata-pipeline .
docker run -p 8501:8501 --env-file .env neurodata-pipeline
```

The app is available at [http://localhost:8501](http://localhost:8501).

---

## How to use the AI Agents

### Analysis Agent Chat (💬 tab)

The Analysis Agent is a LangChain agent powered by Gemini 2.5 Flash. It can read your entire DataFrame and execute Python code to answer your questions.

**Example prompts you can type:**

```
Plot the first 6 channels as a time series with a dark background.
```
```
Calculate band power for Delta, Theta, Alpha, Beta, and Gamma and show a bar chart.
```
```
Generate a correlation heatmap of all numeric columns.
```
```
Are there any obvious artifacts or flatlines in the signal?
```
```
What is the average amplitude of channel Fp1 compared to C3?
```

You can also use the **quick-viz buttons in the sidebar** to fire pre-written prompts instantly (Boxplot, Correlation Heatmap, PSD, Time Series, EEG Band Power).

> All generated plots are saved as PNG files and displayed inline. You can download them with the **💾 Download Graph** button that appears below each plot.

### Data Quality Agent (📊 tab)

This agent runs automatically every time you load a file. It receives descriptive statistics, missing-value counts, and column dtypes, then produces a bullet-point QA report. No action needed — just check the tab after ingestion.

---

## MLflow — Experiment Tracking

Every action is logged to MLflow automatically in a local SQLite database (`mlflow.db`).

### ETL Run (logged on every file load)

| Logged item | Value |
|-------------|-------|
| `low_freq` | Bandpass low cut-off (Hz) |
| `high_freq` | Bandpass high cut-off (Hz) |
| `notch` | Notch filter setting |
| `apply_reference` | Whether CAR was applied |
| `compression` | Parquet compression type |
| `num_rows` | Rows in output DataFrame |
| `num_columns` | Columns in output DataFrame |

### Model Training Run (logged on XGBoost train)

| Logged item | Value |
|-------------|-------|
| `task_type` | `classification` or `regression` (auto-detected) |
| `n_estimators` | 100 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `test_size` | 0.2 |
| `num_features` | Number of selected feature columns |
| `n_classes` | Number of classes (classification only) |
| `accuracy` | Test accuracy (classification) |
| `r2_score` | R² score (regression) |
| Artifact | Trained XGBoost model saved as `xgb_model` |

---

## XGBoost Model Training

The model training tab auto-detects the task type:
- **Classification** — if the target column is categorical or has ≤ 10 unique values
- **Regression** — if the target column is numeric with > 10 unique values

It handles edge cases automatically:
- String/categorical feature columns are label-encoded before training
- Stratified split falls back gracefully if any class has only 1 sample
- A clear error is shown if the dataset has fewer than 20 rows

After training, a **horizontal feature importance chart** is displayed showing which columns the model relied on most.

---

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
│   └── config.toml     # Streamlit theme config
└── icon.png            # App icon (PNG with transparent background)
```

---

## Authentication

| Mode | When used |
|------|-----------|
| GCP Application Default Credentials (ADC) | Kubernetes / Cloud Run / local `gcloud auth` |
| `GOOGLE_API_KEY` in `.env` | Local development fallback |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| AI Agent | LangChain + Google Gemini 2.5 Flash |
| ML Model | XGBoost (classifier + regressor) |
| Experiment Tracking | MLflow (SQLite backend) |
| Signal Processing | MNE-Python |
| Data | Pandas, NumPy, PyArrow |
| Containerization | Docker |

---

## Notes

- MLflow stores data locally in `mlflow.db` (SQLite) and `mlruns/`. Both are excluded from git via `.gitignore`.
- The AI agents require internet access to reach the Gemini API. If the API rate limit is hit, the QA report is skipped gracefully and the rest of the pipeline still completes.
- Plots generated by the Analysis Agent are saved to `./ui_graphs/` and cleared before each new query.
