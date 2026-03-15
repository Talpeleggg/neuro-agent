import glob
import hashlib
import io
import os

import pandas as pd
import streamlit as st

from agent import generate_data_quality_report, get_neural_agent
from data_ingestion import ALL_SUPPORTED, process_neuro_data

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='NeuroData Pipeline',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero-banner {
    background: linear-gradient(135deg, #101015 0%, #202025 100%);
    padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
    border: 1px solid #303035;
}
.hero-title {
    color: #FFFFFF; font-size: 2.2rem !important;
    font-weight: 700 !important; margin-bottom: 0.2rem !important;
}
.stChatMessage { border-radius: 10px; margin-bottom: 10px; }
.stChatMessage.user { background-color: #262730; }
.stChatMessage.assistant { background-color: #101015; border: 1px solid #202025; }
div.stMetric {
    background-color: #1A1A20; border-radius: 10px;
    padding: 15px; border: 1px solid #202025;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">🧠 NeuroData Pipeline</h1>
    <p style='color: #A0A0A5; margin-bottom: 0;'>
        Enterprise Multi-Agent BCI &amp; EEG Analysis
    </p>
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
    st.header('⚙️ Data Engineering Config')

    uploaded_file = st.file_uploader(
        '1. Ingest Data File',
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

    execute_pipeline = st.button(
        '🚀 Execute Preprocessing', type='primary', use_container_width=True
    )

    OUTPUT_DIR = os.path.abspath('./ui_graphs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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


# ── Pipeline execution ─────────────────────────────────────────────────────────
if execute_pipeline and not uploaded_file:
    st.warning('Please upload a file before executing the pipeline.')

if execute_pipeline and uploaded_file:
    with st.spinner('ETL Engine: Processing file & running Data Quality Agent…'):
        try:
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

            st.session_state['df']              = df
            st.session_state['parquet_path']    = parquet_path
            st.session_state['metadata']        = metadata
            st.session_state['dq_report']       = dq_report
            st.session_state['source_filename'] = uploaded_file.name
            st.session_state['data_loaded']     = True
            st.session_state['compression']     = compression
            # Invalidate cached agent so it picks up new data + metadata
            st.session_state.pop('agent', None)
            st.session_state.pop('agent_hash', None)
            st.success('Pipeline complete!')
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
            type='primary',
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
    tab_chat, tab_dq, tab_preview, tab_data = st.tabs([
        '💬 Analysis Agent Chat',
        '🤖 Data Quality Report',
        '👁️ Signal Preview',
        '📋 Data Preview',
    ])

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
            max_rows = st.slider(
                'Rows to display',
                min_value=1_000,
                max_value=min(100_000, len(df)),
                value=min(10_000, len(df)),
                step=1_000,
            )
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
