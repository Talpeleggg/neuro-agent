import os
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mne
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Format registries ──────────────────────────────────────────────────────────

MNE_READERS: Dict[str, Any] = {
    'edf': mne.io.read_raw_edf,
    'bdf': mne.io.read_raw_bdf,
    'fif': mne.io.read_raw_fif,
    'set': mne.io.read_raw_eeglab,
    'vhdr': mne.io.read_raw_brainvision,
    'cnt': mne.io.read_raw_cnt,
    'gdf': mne.io.read_raw_gdf,
}

TABULAR_FORMATS = {'csv', 'tsv', 'xlsx', 'xls'}
NUMPY_FORMATS   = {'npy', 'npz'}
ALL_SUPPORTED   = set(MNE_READERS) | TABULAR_FORMATS | NUMPY_FORMATS


# ── Internal helpers ───────────────────────────────────────────────────────────

def _write_to_disk(uploaded_file, temp_dir: str) -> str:
    """Persist a Streamlit UploadedFile to a temp directory."""
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def _load_mne_raw(file_path: str, ext: str) -> mne.io.BaseRaw:
    """Load an electrophysiology file into an MNE Raw object."""
    reader = MNE_READERS[ext]
    kwargs: Dict[str, Any] = {'preload': True, 'verbose': False}
    if ext == 'fif':
        kwargs['allow_maxshield'] = True
    return reader(file_path, **kwargs)


def _extract_metadata(raw: mne.io.BaseRaw) -> Dict[str, Any]:
    """Pull key recording metadata from an MNE Raw object."""
    return {
        'source_format': 'mne',
        'sfreq':         raw.info['sfreq'],
        'n_channels':    len(raw.ch_names),
        'ch_names':      list(raw.ch_names),
        'ch_types':      raw.get_channel_types(),
        'duration_sec':  float(raw.times[-1]),
        'n_samples':     raw.n_times,
        'meas_date':     str(raw.info.get('meas_date') or 'Unknown'),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def process_neuro_data(
    uploaded_file,
    apply_denoise: bool   = False,
    l_freq: float         = 1.0,
    h_freq: float         = 40.0,
    compression: str      = 'snappy',
    notch_freq: Optional[float] = None,
    apply_reference: bool = True,
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Production ETL engine for neuroscience data.

    Supported formats
    -----------------
    Electrophysiology (MNE): EDF, BDF, FIF, SET, VHDR, CNT, GDF
    Tabular               : CSV, TSV, XLSX / XLS
    NumPy arrays          : NPY, NPZ

    Parameters
    ----------
    uploaded_file   : Streamlit UploadedFile
    apply_denoise   : Apply bandpass filter (MNE formats only)
    l_freq          : Bandpass low cut-off in Hz
    h_freq          : Bandpass high cut-off in Hz
    compression     : Parquet compression – 'snappy', 'gzip', or 'zstd'
    notch_freq      : Notch filter frequency (50 or 60 Hz). None = skip.
    apply_reference : Apply average EEG re-reference (MNE formats only)

    Returns
    -------
    df           : Processed DataFrame
    parquet_path : Absolute path to the compressed Parquet file
    metadata     : Dict with recording metadata (sfreq, ch_names, etc.)
    """
    # ── Input validation ───────────────────────────────────────────────────────
    if apply_denoise and l_freq >= h_freq:
        raise ValueError(
            f"l_freq ({l_freq} Hz) must be strictly less than h_freq ({h_freq} Hz)."
        )

    ext = Path(uploaded_file.name).suffix.lower().lstrip('.')
    if ext not in ALL_SUPPORTED:
        supported_str = ', '.join(f'.{e}' for e in sorted(ALL_SUPPORTED))
        raise ValueError(
            f"Unsupported format '.{ext}'. Supported formats: {supported_str}"
        )

    temp_dir = tempfile.mkdtemp(prefix='neurodata_')
    try:
        file_path = _write_to_disk(uploaded_file, temp_dir)

        # ── Electrophysiology branch ───────────────────────────────────────────
        if ext in MNE_READERS:
            raw      = _load_mne_raw(file_path, ext)
            metadata = _extract_metadata(raw)

            if apply_denoise:
                raw.filter(l_freq=l_freq, h_freq=h_freq,
                           fir_design='firwin', verbose=False)
                metadata['bandpass'] = f'{l_freq}–{h_freq} Hz'

            if notch_freq:
                raw.notch_filter(freqs=notch_freq, verbose=False)
                metadata['notch'] = f'{notch_freq} Hz'

            if apply_reference:
                eeg_picks = mne.pick_types(raw.info, eeg=True)
                if len(eeg_picks) > 0:
                    raw.set_eeg_reference('average', projection=False, verbose=False)
                    metadata['reference'] = 'average'

            df = raw.to_data_frame(scalings='auto')

        # ── Tabular branch ─────────────────────────────────────────────────────
        elif ext == 'csv':
            df       = pd.read_csv(file_path)
            metadata = {'source_format': 'csv'}

        elif ext == 'tsv':
            df       = pd.read_csv(file_path, sep='\t')
            metadata = {'source_format': 'tsv'}

        elif ext in ('xlsx', 'xls'):
            df       = pd.read_excel(file_path)
            metadata = {'source_format': ext}

        # ── NumPy branch ───────────────────────────────────────────────────────
        elif ext == 'npy':
            arr = np.load(file_path, allow_pickle=False)
            if arr.ndim == 1:
                df = pd.DataFrame({'signal': arr})
            elif arr.ndim == 2:
                df = pd.DataFrame(
                    arr.T, columns=[f'ch_{i}' for i in range(arr.shape[0])]
                )
            else:
                raise ValueError(
                    f"NPY array has unsupported shape {arr.shape}. Expected 1-D or 2-D."
                )
            metadata = {'source_format': 'npy', 'shape': list(arr.shape)}

        elif ext == 'npz':
            npz = np.load(file_path, allow_pickle=False)
            df  = pd.DataFrame({k: npz[k].flatten() for k in npz.files})
            metadata = {'source_format': 'npz', 'keys': list(npz.files)}

        # ── Parquet export ─────────────────────────────────────────────────────
        parquet_path = os.path.join(temp_dir, 'optimized_dataset.parquet')
        df.to_parquet(parquet_path, engine='pyarrow', compression=compression, index=False)

        logger.info(
            "Processed '%s' → %d rows × %d cols | compression=%s",
            uploaded_file.name, len(df), df.shape[1], compression,
        )
        return df, parquet_path, metadata

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
