import streamlit as st
import os
import glob
from agent import get_neural_agent
from data_ingestion import process_neuro_data

st.set_page_config(page_title="NeuroData Pipeline", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

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
        <p style='color: #A0A0A5; margin-bottom: 0;'>Enterprise BCI Analysis with Modular Parquet ETL</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Data Engineering Config")
    st.success("Identity: K8s/ADC Managed")
    st.markdown("---")
    
    # 1. Upload
    uploaded_file = st.file_uploader("Ingest CSV or EDF", type=["csv", "edf"])
    
    # 2. Modular Parquet & ETL Controls
    with st.expander("🛠️ Advanced ETL Settings", expanded=True):
        apply_denoise = st.checkbox("Apply Bandpass Filter (EDF)", value=True)
        l_freq = st.number_input("Low Cutoff (Hz)", value=1.0, step=0.5)
        h_freq = st.number_input("High Cutoff (Hz)", value=40.0, step=1.0)
        compression = st.selectbox("Parquet Compression", ["snappy", "gzip", "uncompressed"], index=0)

    # 3. Quick Visualizations
    st.markdown("---")
    st.markdown("### 📊 Standard Visualizations")
    q_btn_1 = st.button("📈 Signal Distribution Boxplot", use_container_width=True)
    q_btn_2 = st.button("🔥 Cross-Channel Heatmap", use_container_width=True)
    q_btn_3 = st.button("📉 Scatter Plot (Alpha vs Beta)", use_container_width=True)
    
    OUTPUT_DIR = os.path.abspath("./ui_graphs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if st.button("🗑️ Clear Pipeline History"):
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.rerun()

if uploaded_file:
    with st.spinner("ETL Engine: Compressing to optimized Parquet..."):
        try:
            # Pass all the modular parameters from the UI!
            df, parquet_path = process_neuro_data(uploaded_file, apply_denoise, l_freq, h_freq, compression)
            agent = get_neural_agent(df, OUTPUT_DIR)
            st.session_state['data_loaded'] = True
        except Exception as e:
            st.error(f"Ingestion Error: {e}")
            st.stop()

    if st.session_state.get('data_loaded'):
        st.markdown("### 📡 Optimized Data Telemetry")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Features (Channels)", df.shape[1])
        col2.metric("Signal Rows", f"{df.shape[0]:,}")
        col3.metric("Missing Values", df.isna().sum().sum())
        col4.metric(f"Parquet Size ({compression})", f"{os.path.getsize(parquet_path) / (1024 * 1024):.2f} MB")
        st.divider()

        tab1, tab2 = st.tabs(["💬 Assistant Chat", "📋 Data Preview"])
        with tab2:
            st.dataframe(df.head(100), use_container_width=True)

        with tab1:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            query_input = st.chat_input("Request analysis on the compressed dataset...")
            
            # Map Quick Action Buttons to Prompts
            if q_btn_1: query_input = "Generate a boxplot showing the distribution of Alpha_Power and Beta_Power. Save as png."
            if q_btn_2: query_input = "Generate a correlation heatmap of all continuous numeric signal columns. Save as png."
            if q_btn_3: query_input = "Generate a scatter plot comparing Alpha_Power to Beta_Power. Save as png."

            if query_input:
                st.session_state.messages.append({"role": "user", "content": query_input})
                with st.chat_message("user"):
                    st.markdown(query_input)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing signals and rendering graphs..."):
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
                                        st.download_button("💾 Download Generated Graph", data=file, file_name=os.path.basename(img_path), mime="image/png")
                        except Exception as e:
                            st.error(f"Analysis Error: {e}")
elif not uploaded_file:
    st.info("👈 System Ready. Upload an EDF or CSV file in the sidebar to begin the ETL pipeline.")