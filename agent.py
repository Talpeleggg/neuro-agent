import os
from typing import Any, Dict, Optional

import google.auth
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent


def _get_credentials() -> Optional[str]:
    """Return an API key string, or None if Application Default Credentials are available."""
    load_dotenv()
    try:
        google.auth.default()
        return None  # ADC available — no key needed
    except google.auth.exceptions.DefaultCredentialsError:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise EnvironmentError(
                'No ADC credentials and no GOOGLE_API_KEY found in .env'
            )
        return api_key


def get_neural_agent(df, output_dir: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Build the primary Analysis Agent for chatting and plotting.

    The agent is a Gemini-backed LangChain DataFrame agent that understands
    EEG / BCI domain context and saves all plots as PNGs to `output_dir`.
    """
    api_key = _get_credentials()
    llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash', temperature=0, api_key=api_key
    )

    # Build optional recording-metadata context block
    meta_block = ''
    if metadata and metadata.get('source_format') == 'mne':
        ch_preview = ', '.join((metadata.get('ch_names') or [])[:20])
        if len(metadata.get('ch_names') or []) > 20:
            ch_preview += ' …'
        meta_block = f"""
RECORDING METADATA (use this to provide accurate neuroscience context):
  • Sampling rate  : {metadata.get('sfreq')} Hz
  • Channels ({metadata.get('n_channels')}): {ch_preview}
  • Duration       : {metadata.get('duration_sec', 0):.1f} s
  • Total samples  : {metadata.get('n_samples', 'N/A'):,}
  • Bandpass       : {metadata.get('bandpass', 'not applied')}
  • Notch          : {metadata.get('notch', 'not applied')}
  • Reference      : {metadata.get('reference', 'not set')}
  • Recorded       : {metadata.get('meas_date', 'Unknown')}
"""

    instructions = f"""
You are an elite Computational Neuroscientist and Lead Data Engineer specialising
in BCI signals, EEG, and brain data analysis.
{meta_block}
VISUALIZATION RULES:
1. Use 'matplotlib' or 'seaborn' for all plots.
2. Apply a dark theme: `plt.style.use('dark_background')` at the start of every plot.
3. ALWAYS save the plot as a '.png' file in EXACTLY this directory: '{output_dir}'.
   Example: plt.savefig('{output_dir}/my_plot.png', dpi=150, bbox_inches='tight')
4. NEVER call plt.show().
5. Label every axis clearly with units (e.g. "Time (s)", "Amplitude (µV)", "Frequency (Hz)").
6. Use descriptive titles and a legend when multiple channels are shown.

ANALYSIS CAPABILITIES:
- Time-series channel plots (offset traces for readability)
- Power Spectral Density (use scipy.signal.welch, log-scale y-axis)
- EEG band-power bar charts (Delta 1-4 Hz, Theta 4-8 Hz, Alpha 8-13 Hz, Beta 13-30 Hz, Gamma 30-45 Hz)
- Correlation / functional-connectivity heatmaps (seaborn.heatmap)
- Signal distribution boxplots
- Event-Related Potential (ERP) averaging
- Artifact flagging (extreme values, flatlines)

CRITICAL: Your final response MUST begin with the exact prefix: "Final Answer: "
"""

    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        prefix=instructions,
        handle_parsing_errors=True,
    )


def generate_data_quality_report(df, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Data Engineering Agent: auto-profiles data health upon ingestion.

    Passes only descriptive statistics (not raw data) to the LLM to keep
    token usage low while still generating an actionable QA report.
    """
    api_key = _get_credentials()
    llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash', temperature=0, api_key=api_key
    )

    stats_str   = df.describe(include='all').to_string()
    missing_str = df.isna().sum().to_string()
    dtype_str   = df.dtypes.to_string()

    meta_str = ''
    if metadata and metadata.get('source_format') == 'mne':
        meta_str = f"""
Recording info: {metadata.get('n_channels')} channels @ {metadata.get('sfreq')} Hz,
duration {metadata.get('duration_sec', 0):.1f} s.
"""

    prompt = f"""You are a strict Data Quality Engineer for a neuroscience lab.
Review the statistical summary, missing-value counts, and dtypes for a newly ingested dataset.
{meta_str}
── Statistics ──
{stats_str}

── Missing values ──
{missing_str}

── Data types ──
{dtype_str}

Write a concise 3–5 bullet-point Data Quality Report. Flag extreme outliers,
significant missing data, suspicious dtypes, or confirm if the data looks clean.
Be specific (mention column names and numbers). Do not write a long essay.
"""

    response = llm.invoke(prompt)
    return response.content
