import os
import zipfile
import io
import gzip
import pandas as pd
import numpy as np
from astropy.io import fits

def load_solexs_from_zip(zip_path, file_in_zip=None):
    """
    Load SoLEXS light curve data from a raw ZIP archive.
    If file_in_zip is None, it searches for the first '.lc.gz' file in the zip.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP archive not found: {zip_path}")
        
    with zipfile.ZipFile(zip_path, 'r') as z:
        if file_in_zip is None:
            lc_files = [f for f in z.namelist() if f.endswith(".lc.gz")]
            if not lc_files:
                raise ValueError(f"No .lc.gz files found in {zip_path}")
            file_in_zip = lc_files[0]
            
        data = gzip.decompress(z.read(file_in_zip))
        with fits.open(io.BytesIO(data)) as hdul:
            rate_data = hdul['RATE'].data
            # Convert to native endianness float64 to prevent pandas Cython operations from crashing
            time_arr = np.array(rate_data['TIME'], dtype=np.float64)
            counts_arr = np.array(rate_data['COUNTS'], dtype=np.float64)
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(time_arr, unit='s'),
                'soft_flux': counts_arr
            })
            # Sort and drop duplicates
            df = df.sort_values('timestamp').drop_duplicates('timestamp')
            return df

def load_helios_from_zip(zip_path, file_in_zip=None, detector='czt1', band='18.00KEV_TO_160.00KEV'):
    """
    Load HEL1OS light curve data from a raw ZIP archive for a specific detector and energy band.
    Detectors can be: 'czt1', 'czt2', 'cdte1', 'cdte2'
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP archive not found: {zip_path}")
        
    with zipfile.ZipFile(zip_path, 'r') as z:
        if file_in_zip is None:
            # Look for light curves of the requested detector
            lc_files = [f for f in z.namelist() if "lightcurve" in f and detector.lower() in f.lower()]
            if not lc_files:
                raise ValueError(f"No lightcurve files found for detector {detector} in {zip_path}")
            file_in_zip = lc_files[0]
            
        with fits.open(io.BytesIO(z.read(file_in_zip))) as hdul:
            # Find the extension that matches the energy band
            ext_name = None
            for hdu in hdul:
                if hdu.name and band.upper() in hdu.name.upper():
                    ext_name = hdu.name
                    break
            
            if ext_name is None:
                # Fallback to the first BinTable extension
                for hdu in hdul:
                    if isinstance(hdu, fits.BinTableHDU):
                        ext_name = hdu.name
                        print(f"Warning: Energy band '{band}' not found. Falling back to extension: {ext_name}")
                        break
            
            if ext_name is None:
                raise ValueError(f"No binary table extensions found in {file_in_zip}")
                
            rate_data = hdul[ext_name].data
            # Convert bytes to string safely
            isot_strs = [x.decode('utf-8').strip() if isinstance(x, bytes) else str(x).strip() for x in rate_data['ISOT']]
            ctr_arr = np.array(rate_data['CTR'], dtype=np.float64)
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(isot_strs),
                'hard_flux': ctr_arr
            })
            df = df.sort_values('timestamp').drop_duplicates('timestamp')
            return df

def align_datasets(df_solexs, df_helios, resample_rule='10s'):
    """
    Align SoLEXS and HEL1OS dataframes onto a uniform time grid.
    Fills missing values using linear interpolation.
    """
    # Set index to timestamp
    df1 = df_solexs.copy().set_index('timestamp')
    df2 = df_helios.copy().set_index('timestamp')
    
    # Resample to uniform time intervals
    df1_res = df1.resample(resample_rule).mean()
    df2_res = df2.resample(resample_rule).mean()
    
    # Merge datasets
    merged = pd.merge(df1_res, df2_res, left_index=True, right_index=True, how='outer')
    
    # Interpolate missing values and forward/backward fill the rest
    merged['soft_flux'] = merged['soft_flux'].interpolate(method='linear').ffill().bfill()
    merged['hard_flux'] = merged['hard_flux'].interpolate(method='linear').ffill().bfill()
    
    return merged

def generate_synthetic_data(duration_days=5, sample_rate_sec=10, flare_rate=2.0, seed=42):
    """
    Generate realistic synthetic solar flare time-series data.
    Simulates:
      - Baseline solar flux with slow variations and noise
      - Occasional flare events following GOES intensity levels
      - Fast rise phase, slower exponential decay
      - Soft X-ray (SoLEXS) rising slightly before and peaking slightly after hard X-ray (HEL1OS) peaks
    """
    np.random.seed(seed)
    
    # Time parameters
    start_time = pd.Timestamp('2026-06-10 00:00:00')
    end_time = start_time + pd.Timedelta(days=duration_days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{sample_rate_sec}s')
    n_points = len(timestamps)
    
    # 1. Baseline Flux
    # Slow baseline variation using sine waves + random walk
    t_hours = np.linspace(0, duration_days * 24, n_points)
    soft_base = 4.0 + 1.0 * np.sin(2 * np.pi * t_hours / 24) + np.cumsum(np.random.normal(0, 0.02, n_points))
    hard_base = 18.0 + 3.0 * np.sin(2 * np.pi * t_hours / 24) + np.cumsum(np.random.normal(0, 0.05, n_points))
    
    # Clip baseline to positive values
    soft_base = np.maximum(soft_base, 1.0)
    hard_base = np.maximum(hard_base, 5.0)
    
    # Add Gaussian noise
    soft_flux = soft_base + np.random.normal(0, 0.3, n_points)
    hard_flux = hard_base + np.random.normal(0, 1.5, n_points)
    
    # Flare attributes
    # Poisson process for flare starts
    dt_day = sample_rate_sec / 86400.0
    prob_flare = flare_rate * dt_day
    
    flare_starts = np.random.random(n_points) < prob_flare
    
    # Information on active flares to track overlaps
    # Flare classes and amplitude scales
    # We will map flare classes to peak amplitudes
    # A: < 10, B: 10-50, C: 50-250, M: 250-1000, X: >= 1000
    classes = ['B', 'C', 'M', 'X']
    class_probs = [0.55, 0.30, 0.12, 0.03]  # Less frequent for higher classes
    
    # We also keep track of flare labels for validation
    true_labels = np.zeros(n_points, dtype=int)
    flare_class_labels = [None] * n_points
    
    i = 0
    while i < n_points:
        if flare_starts[i]:
            # Generate a flare
            f_class = np.random.choice(classes, p=class_probs)
            
            # Amplitude ranges
            if f_class == 'B':
                peak_amp = np.random.uniform(15, 45)
            elif f_class == 'C':
                peak_amp = np.random.uniform(50, 200)
            elif f_class == 'M':
                peak_amp = np.random.uniform(250, 800)
            else:  # 'X'
                peak_amp = np.random.uniform(1000, 3000)
                
            # Rise time in steps (2 to 8 mins)
            rise_time_sec = np.random.uniform(120, 480)
            rise_steps = int(rise_time_sec / sample_rate_sec)
            
            # Decay half-life in seconds (10 to 45 mins)
            decay_half_life_sec = np.random.uniform(600, 2700)
            decay_constant = np.log(2) / decay_half_life_sec
            
            # Time lag: HEL1OS (hard X-ray) peaks slightly before SoLEXS (soft X-ray) peaks.
            # HEL1OS peaks during the rise phase of SoLEXS (impulsive phase).
            lag_sec = np.random.uniform(60, 180)
            lag_steps = int(lag_sec / sample_rate_sec)
            
            # Duration of flare impact: say 5 half-lives
            flare_duration_steps = rise_steps + int(5 * decay_half_life_sec / sample_rate_sec)
            
            # Apply flare profile to both channels
            for step in range(flare_duration_steps):
                idx = i + step
                if idx >= n_points:
                    break
                
                # Relative time in seconds from flare start
                t_sec = step * sample_rate_sec
                
                # True label is 1 during active flare (rise + 2 half lives)
                if t_sec < (rise_time_sec + 2 * decay_half_life_sec):
                    true_labels[idx] = 1
                    flare_class_labels[idx] = f_class
                
                # --- Soft X-Ray Profile (SoLEXS) ---
                # Peaks at t = rise_time_sec
                if t_sec <= rise_time_sec:
                    # Sinusoidal rise
                    factor = np.sin((t_sec / rise_time_sec) * (np.pi / 2)) ** 2
                    soft_flare_contrib = peak_amp * factor
                else:
                    # Exponential decay
                    soft_flare_contrib = peak_amp * np.exp(-decay_constant * (t_sec - rise_time_sec))
                    
                # --- Hard X-Ray Profile (HEL1OS) ---
                # Peaks slightly earlier at t = rise_time_sec - lag_sec
                hard_peak_amp = peak_amp * np.random.uniform(0.6, 1.4)  # Hard X-ray amplitude correlated but distinct
                hard_peak_t = rise_time_sec - lag_sec
                
                # Hard X-ray has faster rise and much faster decay
                hard_decay_constant = decay_constant * np.random.uniform(1.8, 2.5) # decays ~2x faster
                
                if t_sec <= hard_peak_t:
                    # Fast rise
                    factor = np.sin((t_sec / hard_peak_t) * (np.pi / 2)) ** 2
                    hard_flare_contrib = hard_peak_amp * factor
                else:
                    # Fast exponential decay
                    hard_flare_contrib = hard_peak_amp * np.exp(-hard_decay_constant * (t_sec - hard_peak_t))
                
                soft_flux[idx] += soft_flare_contrib
                hard_flux[idx] += hard_flare_contrib
                
            # Skip forward to avoid triggering another flare in the middle of the rise phase
            i += rise_steps
        else:
            i += 1
            
    # Clip fluxes to avoid negative counts
    soft_flux = np.maximum(soft_flux, 0.1)
    hard_flux = np.maximum(hard_flux, 0.5)
    
    df = pd.DataFrame({
        'soft_flux': soft_flux,
        'hard_flux': hard_flux,
        'true_label': true_labels,
        'true_class': flare_class_labels
    }, index=timestamps)
    
    df.index.name = 'timestamp'
    return df

if __name__ == "__main__":
    # Test synthetic generation
    print("Testing synthetic data generation...")
    df = generate_synthetic_data(duration_days=1, sample_rate_sec=10)
    print("Shape:", df.shape)
    print(df.describe())
    print("Flares generated:", df['true_label'].sum() * 10 / 60, "minutes of active flares.")
    print("Classes present:", df['true_class'].value_counts())
