import streamlit as st
import os
import glob
import pandas as pd
from agent import get_neural_agent, generate_data_quality_report
from data_ingestion import process_neuro_data

st.set_page_config(page_title="NeuroData Pipeline", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

def format_file_size(size_in_bytes):
    """Fixes the 0.00MB bug dynamically showing KB or MB"""
    if size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.2f} KB"
    return f"{size_in_bytes / (1024 * 1024):.2f} MB"

st.markdown("""
    <style>
    .hero-banner { background: linear-gradient(135deg, #101015 0%, #202025 100%); padding: 2rem; border-radius: 12px; margin-bottom: 2rem; border: 1px solid #303035; }
    .hero-title { color: #FFFFFF; font-size: 2.2rem !important; font-weight: 700 !important; margin-bottom: 0.2rem !important; }
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    .stChatMessage.user { background-color: #262730; }
    .stChatMessage.assistant { background-color: #101015; border: 1px solid #202025; }
    div.stMetric { background-color: #1A1A20; border-radius: 10px; padding: 15px; border: 1px solid #202025; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🧠 NeuroData Pipeline</h1>
        <p style='color: #A0A0A5; margin-bottom: 0;'>Enterprise Multi-Agent BCI Analysis</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR: EXPLICIT ETL GATING
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Data Engineering Config")
    uploaded_file = st.file_uploader("1. Ingest CSV or EDF", type=["csv", "edf"])
    
    st.markdown("### 🎛️ 2. ETL Parameters")
    # Using standard inputs, but they NO LONGER auto-trigger the pipeline
    apply_denoise = st.checkbox("Apply Bandpass Filter (EDF)", value=True)
    l_freq = st.number_input("Low Cutoff (Hz)", value=1.0, step=0.5)
    h_freq = st.number_input("High Cutoff (Hz)", value=40.0, step=1.0)
    compression = st.selectbox("Parquet Compression", ["snappy", "gzip"], index=0)

    # EXPLICIT EXECUTION BUTTON
    execute_pipeline = st.button("🚀 Execute Preprocessing", type="primary", use_container_width=True)
    
    OUTPUT_DIR = os.path.abspath("./ui_graphs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    st.markdown("---")
    st.markdown("### 📊 Standard Visualizations")
    q_btn_1 = st.button("📈 Signal Distribution Boxplot", use_container_width=True)
    q_btn_2 = st.button("🔥 Cross-Channel Heatmap", use_container_width=True)
    
    if st.button("🗑️ Reset Pipeline"):
        st.session_state.clear()
        st.rerun()

# ---------------------------------------------------------
# MAIN LOGIC & PIPELINE EXECUTION
# ---------------------------------------------------------
# Only run the heavy processing if the button is clicked!
if execute_pipeline and uploaded_file:
    with st.spinner("ETL Engine: Compressing & running Data Quality Agent..."):
        try:
            # 1. Process data deterministically
            df, parquet_path = process_neuro_data(uploaded_file, apply_denoise, l_freq, h_freq, compression)
            
            # 2. Trigger the Data Engineering Agent for automated QA
            dq_report = generate_data_quality_report(df)
            
            # 3. Save to session state so it survives UI refreshes
            st.session_state['df'] = df
            st.session_state['parquet_path'] = parquet_path
            st.session_state['dq_report'] = dq_report
            st.session_state['data_loaded'] = True
        except Exception as e:
            st.error(f"Pipeline Error: {e}")

# If the pipeline has been successfully executed, show the workspace
if st.session_state.get('data_loaded'):
    df = st.session_state['df']
    parquet_path = st.session_state['parquet_path']
    
    # --- Top Section: Download & Telemetry ---
    st.markdown("### 📡 Optimized Parquet Telemetry")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Features (Channels)", df.shape[1])
    col2.metric("Signal Rows", f"{df.shape[0]:,}")
    col3.metric("Missing Values", df.isna().sum().sum())
    
    # Dynamic Size Formatter Fix
    file_size_bytes = os.path.getsize(parquet_path)
    col4.metric(f"Size ({compression})", format_file_size(file_size_bytes))
    
    # THE DOWNLOAD BUTTON
    with open(parquet_path, "rb") as file:
        st.download_button(
            label="💾 Download Processed .parquet File",
            data=file,
            file_name=f"cleaned_{uploaded_file.name}.parquet",
            mime="application/octet-stream",
            type="primary"
        )
    
    st.divider()

    # --- Multi-Agent Workspace ---
    tab1, tab2, tab3 = st.tabs(["💬 Analysis Agent Chat", "🤖 Data Quality Report", "📋 Data Preview"])
    
    with tab3:
        st.dataframe(df.head(100), use_container_width=True)
        
    with tab2:
        st.info("This report was automatically generated by the Data Engineering Agent during ingestion.")
        st.markdown(st.session_state['dq_report'])

    with tab1:
        # Initialize the Analysis Agent
        agent = get_neural_agent(df, OUTPUT_DIR)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        query_input = st.chat_input("Request analysis on the compressed dataset...")
        
        if q_btn_1: query_input = "Generate a boxplot showing the distribution of Alpha_Power and Beta_Power. Save as png."
        if q_btn_2: query_input = "Generate a correlation heatmap of all continuous numeric signal columns. Save as png."

        if query_input:
            st.session_state.messages.append({"role": "user", "content": query_input})
            with st.chat_message("user"):
                st.markdown(query_input)

            with st.chat_message("assistant"):
                with st.spinner("Analysis Agent rendering graphs..."):
                    for f in glob.glob(f"{OUTPUT_DIR}/*.png"):
                        os.remove(f)

                    try:
                        answer = agent.invoke(query_input)
                        clean_text = answer['output'].replace("Final Answer:", "").strip()
                        st.markdown(clean_text)
                        st.session_state.messages.append({"role": "assistant", "content": clean_text})

                        generated_images = glob.glob(f"{OUTPUT_DIR}/*.png")
                        if generated_images:
                            for img_path in generated_images:
                                st.image(img_path, use_container_width=True)
                                with open(img_path, "rb") as file:
                                    st.download_button("💾 Download Graph", data=file, file_name=os.path.basename(img_path), mime="image/png")
                    except Exception as e:
                        st.error(f"Analysis Error: {e}")
elif not uploaded_file:
    st.info("👈 Upload an EDF or CSV file, adjust parameters, and click 'Execute Preprocessing' to begin.")