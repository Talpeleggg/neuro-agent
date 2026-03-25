import base64
import glob
import hashlib
import io
import os

import pandas as pd
import streamlit as st
import mlflow

# Import ML training logic from the external engine module
from ml_engine import train_xgboost_and_log, setup_mlflow

from agent import generate_data_quality_report, get_neural_agent
from data_ingestion import ALL_SUPPORTED, process_neuro_data

# ── Image processing for icons ────────────────────────────────────
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    except FileNotFoundError:
        return ""

# 1. The small icon (for sidebar and browser tab)
# Must use PNG to support transparent background. Ensure icon.png exists in the project root.
_ICON_B64 = get_image_base64("icon.png")
_SMALL_ICON_HTML = f'<img src="data:image/png;base64,{_ICON_B64}" width="26" height="26" style="display:inline-block; vertical-align:middle; margin-right: 10px;" alt="Brain Icon"/>'

# 2. The large transparent logo (for the main hero banner on the right)
_LOGO_B64 = get_image_base64("icon.png")
if _LOGO_B64:
    _LARGE_LOGO_HTML = f'<img src="data:image/png;base64,{_LOGO_B64}" width="180" style="display:block; object-fit:contain;" alt="NeuroData Logo"/>'
else:
    _LARGE_LOGO_HTML = '<div style="color:#ff4b4b; font-size:12px; border:1px dashed #ff4b4b; padding:10px;">Missing:<br>logo_transparent.png</div>'


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='NeuroData Pipeline',
    page_icon=f"data:image/png;base64,{_ICON_B64}" if _ICON_B64 else "🧠",
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* Hero Banner - Dark blue gradient (always dark regardless of theme) */
.hero-banner {{
    background: linear-gradient(130deg, #0D1F45 0%, #1A3A8A 45%, #0F2460 100%);
    border: 1px solid #2A52A8;
    border-radius: 14px;
    margin-bottom: 2rem;
    overflow: hidden;
    display: flex;
    align-items: center;
    box-shadow: 0 6px 28px rgba(20,55,160,0.28), 0 2px 8px rgba(0,0,0,0.30);
}}
.hero-body {{
    flex: 1;
    padding: 2rem 2.4rem;
}}
.hero-title {{
    color: #FFFFFF !important; 
    font-size: 2.2rem !important;
    font-weight: 700 !important; 
    margin-bottom: 0.25rem !important;
    letter-spacing: -0.02em;
    font-family: sans-serif;
}}
.hero-sub {{
    color: #93C5FD !important; 
    font-size: 0.90rem !important; 
    margin: 0 !important;
}}
.hero-art {{
    flex-shrink: 0;
    display: flex; 
    align-items: center; 
    justify-content: center;
    padding: 1rem 2.5rem 1rem 0;
}}

/* Sidebar styling */
.sidebar-header-container {{
    display: flex;
    align-items: center;
    justify-content: flex-start; 
    margin-left: -10px; 
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.2);
}}
.sidebar-header-text {{
    color: var(--text-color); /* adapts to light/dark mode */
    font-weight: 800;
    font-size: 1.2rem;
    margin: 0;
}}

/* --- BUTTON STYLES (Bold & Popping) --- */
/* White background with thick border and shadow to stand out against the blue hero */
div.stButton > button {{
    background-color: #FFFFFF !important;
    color: #1A3A8A !important;
    border: 2px solid #4A78F5 !important;
    border-radius: 8px !important;
    font-weight: 800 !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.3s ease !important;
}}
div.stButton > button:hover {{
    background-color: #4A78F5 !important;
    color: #FFFFFF !important;
    border-color: #4A78F5 !important;
    box-shadow: 0 6px 15px rgba(74, 120, 245, 0.3) !important;
    transform: translateY(-2px) !important;
}}
div.stButton > button:active {{
    background-color: #1A3A8A !important;
    color: #FFFFFF !important;
    border-color: #1A3A8A !important;
}}

/* --- METRICS CARDS (Bold & 3D Effect) --- */
/* Solid white cards with rounded corners and shadow for depth */
div.stMetric {{
    background-color: #FFFFFF !important;
    border-radius: 14px !important;
    padding: 20px 24px !important;
    border: 1px solid rgba(26, 58, 138, 0.1) !important;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06) !important;
}}
/* Metric label — bold, blue */
div.stMetric label {{
    font-weight: 700 !important;
    color: #4A78F5 !important;
    font-size: 1.05rem !important;
}}
/* Metric value — dark navy, large and bold */
div.stMetric [data-testid="stMetricValue"] {{
    font-weight: 900 !important;
    color: #0D1F45 !important;
}}

/* --- DOWNLOAD BUTTON STYLES (matches regular buttons) --- */
div.stDownloadButton > button {{
    background-color: #FFFFFF !important;
    color: #1A3A8A !important;
    border: 2px solid #4A78F5 !important;
    border-radius: 8px !important;
    font-weight: 800 !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.3s ease !important;
    width: 100%;
}}
div.stDownloadButton > button:hover {{
    background-color: #4A78F5 !important;
    color: #FFFFFF !important;
    border-color: #4A78F5 !important;
    box-shadow: 0 6px 15px rgba(74, 120, 245, 0.3) !important;
    transform: translateY(-2px) !important;
}}
div.stDownloadButton > button:active {{
    background-color: #1A3A8A !important;
    color: #FFFFFF !important;
    border-color: #1A3A8A !important;
}}

/* Chat Styles */
.stChatMessage {{ border-radius: 10px; margin-bottom: 10px; }}
.stChatMessage.user {{ background-color: #FFFFFF; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }}
.stChatMessage.assistant {{ background-color: rgba(74, 120, 245, 0.05); border: 1px solid rgba(74, 120, 245, 0.2); }}
</style>
""", unsafe_allow_html=True)

# Main Page Hero
st.markdown(f"""
<div class="hero-banner">
  <div class="hero-body">
    <div class="hero-title">NeuroData Pipeline</div>
    <div class="hero-sub">Enterprise Multi-Agent BCI &amp; EEG Analysis with MLOps</div>
  </div>
  <div class="hero-art">{_LARGE_LOGO_HTML}</div>
</div>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def format_file_size(n: int) -> str:
    if n < 1024 * 1024:
        return f'{n / 1024:.2f} KB'
    return f'{n / (1024 * 1024):.2f} MB'


def df_fingerprint(df: pd.DataFrame) -> str:
    """Lightweight hash for cache-invalidation of the Analysis Agent."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()


# ── Sidebar ────────────────────────────────────────────────────────────────────
SUPPORTED_EXTS = sorted(ALL_SUPPORTED)
SUPPORTED_LABEL = 'EEG/BCI: EDF, BDF, FIF, SET, VHDR, CNT, GDF  |  Tabular: CSV, TSV, XLSX  |  NumPy: NPY, NPZ'

with st.sidebar:
    if _ICON_B64:
        st.markdown(f"""
        <div class="sidebar-header-container">
            {_SMALL_ICON_HTML}
            <h2 class="sidebar-header-text">NeuroData Pipeline</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-header-container"><h2 class="sidebar-header-text">NeuroData Pipeline</h2></div>', unsafe_allow_html=True)
    
    st.header('⚙️ 1. Data Engineering Config')

    uploaded_file = st.file_uploader(
        'Ingest Data File',
        type=SUPPORTED_EXTS,
        help=SUPPORTED_LABEL,
    )

    st.markdown('### 🎛️ 2. ETL Parameters')

    apply_denoise = st.checkbox('Bandpass Filter (EEG/BCI only)', value=True)
    col_l, col_h = st.columns(2)
    l_freq = col_l.number_input('Low (Hz)',  value=1.0,  step=0.5,  min_value=0.1)
    h_freq = col_h.number_input('High (Hz)', value=40.0, step=1.0)

    notch_freq = st.selectbox(
        'Notch Filter',
        options=[None, 50.0, 60.0],
        format_func=lambda x: 'Off' if x is None else f'{x:.0f} Hz',
        help='Remove power-line interference (50 Hz EU / 60 Hz US)',
    )

    apply_reference = st.checkbox('Average EEG Re-reference', value=True,
                                help='Common Average Reference (CAR) — EEG only')

    compression = st.selectbox('Parquet Compression', ['snappy', 'gzip', 'zstd'], index=0)

    execute_pipeline = False
    try:
        # Removed type='primary' to keep uniform blue button styling
        execute_pipeline = st.button('🚀 Execute Preprocessing', use_container_width=True)
        
        OUTPUT_DIR = os.path.abspath('./ui_graphs')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        st.error(f"UI Error: {e}")

    st.markdown('---')
    st.markdown('### 📊 Quick Visualizations')
    st.caption('Click a button then open the **Analysis Agent Chat** tab.')
    q_boxplot     = st.button('📈 Signal Boxplot',          use_container_width=True)
    q_heatmap     = st.button('🔥 Correlation Heatmap',     use_container_width=True)
    q_psd         = st.button('📡 Power Spectral Density',  use_container_width=True)
    q_timeseries  = st.button('📉 Time Series Plot',        use_container_width=True)
    q_bandpower   = st.button('🎯 EEG Band Power',          use_container_width=True)

    st.markdown('---')
    if st.button('🗑️ Reset Pipeline', use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ── Pipeline execution (With MLflow Tracking) ──────────────────────────────────
if execute_pipeline and not uploaded_file:
    st.warning('Please upload a file before executing the pipeline.')

if execute_pipeline and uploaded_file:
    with st.spinner('ETL Engine: Processing file & running Data Quality Agent…'):
        try:
            # Start MLflow tracking for the ETL ingestion run
            setup_mlflow()
            with mlflow.start_run(run_name="Data_Ingestion_ETL"):
                mlflow.log_params({
                    "low_freq": l_freq,
                    "high_freq": h_freq,
                    "notch": str(notch_freq),
                    "apply_reference": apply_reference,
                    "compression": compression
                })

                df, parquet_path, metadata = process_neuro_data(
                    uploaded_file,
                    apply_denoise=apply_denoise,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    compression=compression,
                    notch_freq=notch_freq,
                    apply_reference=apply_reference,
                )
                dq_report = generate_data_quality_report(df, metadata)
                
                # Log dataset dimensions to MLflow
                mlflow.log_metrics({"num_rows": df.shape[0], "num_columns": df.shape[1]})

            st.session_state['df']              = df
            st.session_state['parquet_path']    = parquet_path
            st.session_state['metadata']        = metadata
            st.session_state['dq_report']       = dq_report
            st.session_state['source_filename'] = uploaded_file.name
            st.session_state['data_loaded']     = True
            st.session_state['compression']     = compression
            # Invalidate cached agent and model results when a new file is loaded
            st.session_state.pop('agent',        None)
            st.session_state.pop('agent_hash',   None)
            st.session_state.pop('model_result', None)
            st.session_state.pop('model_fig',    None)
            st.session_state.pop('model_error',  None)
            st.success('Pipeline complete! Parameters logged to MLflow.')
        except Exception as e:
            st.error(f'Pipeline Error: {e}')


# ── Main workspace ─────────────────────────────────────────────────────────────
if st.session_state.get('data_loaded'):
    df              = st.session_state['df']
    parquet_path    = st.session_state['parquet_path']
    metadata        = st.session_state.get('metadata', {})
    source_filename = st.session_state.get('source_filename', 'dataset')
    compression     = st.session_state.get('compression', 'snappy')

    # ── Telemetry row ──────────────────────────────────────────────────────────
    st.markdown('### 📡 Dataset Telemetry')
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric('Channels / Features', df.shape[1])
    m2.metric('Signal Rows', f'{df.shape[0]:,}')
    m3.metric('Missing Values', int(df.isna().sum().sum()))
    m4.metric(f'Parquet ({compression})', format_file_size(os.path.getsize(parquet_path)))

    if 'sfreq' in metadata:
        m5.metric('Sampling Rate', f"{metadata['sfreq']:.0f} Hz")
    elif 'duration_sec' in metadata:
        m5.metric('Duration', f"{metadata['duration_sec']:.1f} s")
    else:
        m5.metric('Columns', df.shape[1])

    # ── Recording metadata expander (MNE only) ─────────────────────────────────
    if metadata.get('source_format') == 'mne':
        with st.expander('🔬 Recording Metadata', expanded=False):
            rc1, rc2, rc3 = st.columns(3)
            rc1.write(f"**Duration:** {metadata.get('duration_sec', 0):.1f} s")
            rc1.write(f"**Total Samples:** {metadata.get('n_samples', 'N/A'):,}")
            rc1.write(f"**Recorded:** {metadata.get('meas_date', 'Unknown')}")

            ch_names = metadata.get('ch_names', [])
            preview  = ', '.join(ch_names[:20]) + (' …' if len(ch_names) > 20 else '')
            rc2.write(f"**Channels ({metadata.get('n_channels', '?')}):** {preview}")

            rc3.write(f"**Bandpass:** {metadata.get('bandpass', 'not applied')}")
            rc3.write(f"**Notch:** {metadata.get('notch', 'not applied')}")
            rc3.write(f"**Reference:** {metadata.get('reference', 'not set')}")

    # ── Export buttons ─────────────────────────────────────────────────────────
    st.markdown('### 💾 Export Processed Data')
    ex1, ex2, ex3 = st.columns(3)

    with open(parquet_path, 'rb') as f:
        ex1.download_button(
            '⬇️ Download .parquet',
            data=f,
            file_name=f'cleaned_{source_filename}.parquet',
            mime='application/octet-stream',
            use_container_width=True,
        )

    ex2.download_button(
        '⬇️ Download .csv',
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=f'cleaned_{source_filename}.csv',
        mime='text/csv',
        use_container_width=True,
    )

    try:
        xlsx_buf = io.BytesIO()
        df.to_excel(xlsx_buf, index=False)
        ex3.download_button(
            '⬇️ Download .xlsx',
            data=xlsx_buf.getvalue(),
            file_name=f'cleaned_{source_filename}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True,
        )
    except ImportError:
        ex3.info('Install `openpyxl` for Excel export.')

    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────────────
    # Added XGBoost model training tab
    tab_chat, tab_model, tab_dq, tab_preview, tab_data = st.tabs([
        '💬 Analysis Agent Chat',
        '🤖 Model Training (XGBoost)',
        '🤖 Data Quality Report',
        '👁️ Signal Preview',
        '📋 Data Preview',
    ])

    # ── Tab: Model Training (XGBoost & MLflow) - MLOps Separation ──────────────
    with tab_model:
        st.markdown('#### Train Machine Learning Classifier')
        st.info("Train an XGBoost model on your processed data and automatically log the run to MLflow.")

        all_cols = df.columns.tolist()
        target_col = st.selectbox("🎯 Select Target (Label) Column:", all_cols, index=len(all_cols)-1 if all_cols else 0)
        feature_cols = st.multiselect("🧠 Select Feature Columns (Channels):", all_cols, default=[c for c in all_cols if c != target_col][:15])

        if st.button("🚀 Train XGBoost & Log to MLflow", use_container_width=True):
            if not feature_cols or not target_col:
                st.warning("Please select at least one feature and a target column.")
            else:
                with st.spinner("Training model and saving to MLflow..."):
                    metric_text, fig, error_msg = train_xgboost_and_log(df, feature_cols, target_col)
                    if error_msg:
                        st.session_state['model_error']  = error_msg
                        st.session_state.pop('model_result', None)
                        st.session_state.pop('model_fig',    None)
                    else:
                        # Save figure as PNG bytes so it survives reruns
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                        import matplotlib.pyplot as plt; plt.close(fig)
                        st.session_state['model_result'] = metric_text
                        st.session_state['model_fig']    = buf.getvalue()
                        st.session_state.pop('model_error', None)

        # Display persisted results — shown on every rerun without re-training
        if st.session_state.get('model_error'):
            st.error(f"Failed to train model: {st.session_state['model_error']}")
        elif st.session_state.get('model_result'):
            st.success(f"🎉 Model trained successfully! **{st.session_state['model_result']}**")
            st.image(st.session_state['model_fig'], use_container_width=True)

    # ── Tab: Data Preview ──────────────────────────────────────────────────────
    with tab_data:
        st.dataframe(df.head(200), use_container_width=True)

    # ── Tab: Signal Preview ────────────────────────────────────────────────────
    with tab_preview:
        st.markdown('#### Interactive Signal Preview')
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            st.info('No numeric columns found for signal preview.')
        else:
            sel_cols = st.multiselect(
                'Select channels / columns:',
                options=numeric_cols,
                default=numeric_cols[:min(6, len(numeric_cols))],
            )
            SLIDER_THRESHOLD = 1_000
            if len(df) > SLIDER_THRESHOLD:
                max_rows = st.slider(
                    'Rows to display',
                    min_value=SLIDER_THRESHOLD,
                    max_value=min(100_000, len(df)),
                    value=min(10_000, len(df)),
                    step=SLIDER_THRESHOLD,
                )
            else:
                max_rows = len(df)
                st.caption(f'Showing all {len(df):,} rows.')
            if sel_cols:
                st.line_chart(df[sel_cols].iloc[:max_rows], use_container_width=True)
            else:
                st.info('Select at least one column above.')

    # ── Tab: Data Quality Report ───────────────────────────────────────────────
    with tab_dq:
        st.info('Auto-generated by the Data Engineering Agent during ingestion.')
        st.markdown(st.session_state['dq_report'])

    # ── Tab: Analysis Agent Chat ───────────────────────────────────────────────
    with tab_chat:
        # Cache the agent — recreate only when the DataFrame changes
        current_hash = df_fingerprint(df)
        if (
            'agent' not in st.session_state
            or st.session_state.get('agent_hash') != current_hash
        ):
            with st.spinner('Initialising Analysis Agent…'):
                st.session_state['agent']      = get_neural_agent(df, OUTPUT_DIR, metadata)
                st.session_state['agent_hash'] = current_hash

        agent = st.session_state['agent']

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

        # Map quick-viz buttons to prompt strings
        QUICK_QUERIES = {
            q_boxplot:    'Generate a boxplot showing the distribution of all numeric signal columns. Use a dark background. Save as png.',
            q_heatmap:    'Generate a correlation heatmap of all numeric signal columns using seaborn. Use a dark background. Save as png.',
            q_psd:        'Calculate and plot the Power Spectral Density (PSD) for each numeric channel using scipy.signal.welch. Log-scale y-axis. Save as png.',
            q_timeseries: 'Plot the first 6 numeric signal channels as vertically offset time series traces. Dark background, clear labels. Save as png.',
            q_bandpower:  'Calculate average band power for Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-45 Hz) and plot as a grouped bar chart. Save as png.',
        }

        query_input = st.chat_input('Ask the agent to analyse or visualise the data…')
        for btn, prompt in QUICK_QUERIES.items():
            if btn:
                query_input = prompt
                break

        if query_input:
            st.session_state.messages.append({'role': 'user', 'content': query_input})
            with st.chat_message('user'):
                st.markdown(query_input)

            with st.chat_message('assistant'):
                with st.spinner('Analysis Agent rendering…'):
                    # Clear previous PNGs before generating new ones
                    for old_png in glob.glob(f'{OUTPUT_DIR}/*.png'):
                        os.remove(old_png)

                    try:
                        answer     = agent.invoke(query_input)
                        clean_text = answer['output'].replace('Final Answer:', '').strip()
                        st.markdown(clean_text)
                        st.session_state.messages.append(
                            {'role': 'assistant', 'content': clean_text}
                        )

                        for img_path in sorted(glob.glob(f'{OUTPUT_DIR}/*.png')):
                            st.image(img_path, use_container_width=True)
                            with open(img_path, 'rb') as img_f:
                                st.download_button(
                                    '💾 Download Graph',
                                    data=img_f,
                                    file_name=os.path.basename(img_path),
                                    mime='image/png',
                                )
                    except Exception as e:
                        st.error(f'Analysis Error: {e}')

else:
    st.info(
        f'👈 Upload a file ({", ".join(sorted(ALL_SUPPORTED)).upper()}), '
        'configure parameters, and click **Execute Preprocessing** to begin.'
    )