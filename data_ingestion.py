import os
import tempfile
import mne
import pandas as pd

def process_neuro_data(uploaded_file, apply_denoise=False, l_freq=1.0, h_freq=40.0, compression='snappy'):
    """
    Modular ETL engine for Neuroscience data.
    Allows dynamic frequency filtering and optimized Parquet compression.
    """
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_ext = uploaded_file.name.lower().split('.')[-1]
    
    # --- Transformation Layer ---
    if file_ext == 'edf':
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        if apply_denoise:
            # The scientist now controls the bandpass frequencies!
            raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
        df = raw.to_data_frame()
        
    elif file_ext == 'csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported format: {file_ext}")

    # --- Load/Compression Layer ---
    parquet_path = os.path.join(temp_dir, "optimized_dataset.parquet")
    
    # Best Practice: We apply the scientist's chosen compression algorithm
    df.to_parquet(parquet_path, engine='fastparquet', compression=compression)
    
    return df, parquet_path