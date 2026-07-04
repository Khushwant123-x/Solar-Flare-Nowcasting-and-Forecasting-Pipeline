import pandas as pd
import numpy as np

def calculate_rolling_stats(df, window_mins=30, sample_rate_sec=10):
    """
    Calculate rolling baseline (mean) and rolling standard deviation for both channels.
    """
    # Calculate window size in terms of samples
    samples_per_min = 60 / sample_rate_sec
    window_size = int(window_mins * samples_per_min)
    
    # We copy the dataframe to avoid modifications
    df_stats = df.copy()
    
    # Calculate rolling mean and std
    df_stats['soft_rolling_mean'] = df_stats['soft_flux'].rolling(window=window_size, min_periods=1, center=False).mean()
    df_stats['soft_rolling_std'] = df_stats['soft_flux'].rolling(window=window_size, min_periods=1, center=False).std()
    
    df_stats['hard_rolling_mean'] = df_stats['hard_flux'].rolling(window=window_size, min_periods=1, center=False).mean()
    df_stats['hard_rolling_std'] = df_stats['hard_flux'].rolling(window=window_size, min_periods=1, center=False).std()
    
    # Fill any NaNs at the beginning
    df_stats['soft_rolling_std'] = df_stats['soft_rolling_std'].fillna(0.0)
    df_stats['hard_rolling_std'] = df_stats['hard_rolling_std'].fillna(0.0)
    
    return df_stats

def classify_flare(peak_flux):
    """
    Classify flare intensity based on peak soft X-ray flux level.
    Returns a class string (e.g., 'C3.2', 'M1.5') and the broad class ('A'/'B'/'C'/'M'/'X').
    """
    if peak_flux < 10.0:
        val = peak_flux / 1.0
        return f"A{val:.1f}", "A"
    elif peak_flux < 50.0:
        val = peak_flux / 10.0
        return f"B{val:.1f}", "B"
    elif peak_flux < 250.0:
        val = peak_flux / 50.0
        return f"C{val:.1f}", "C"
    elif peak_flux < 1000.0:
        val = peak_flux / 250.0
        return f"M{val:.1f}", "M"
    else:
        val = peak_flux / 1000.0
        return f"X{val:.1f}", "X"

def detect_flares(df, k=3.0, window_mins=30, sample_rate_sec=10):
    """
    Detect solar flares in real-time.
    Rules:
      1. soft_flux > soft_rolling_mean + k * soft_rolling_std
      2. hard_flux > hard_rolling_mean + k * hard_rolling_std
      3. Cross-validate: soft and hard detections must overlap within a window.
    Returns:
      - df: DataFrame with detection columns
      - catalogue: List of dicts representing flare events
    """
    # 1. Compute rolling stats
    df_stats = calculate_rolling_stats(df, window_mins, sample_rate_sec)
    
    # 2. Thresholding
    df_stats['soft_threshold'] = df_stats['soft_rolling_mean'] + k * df_stats['soft_rolling_std']
    df_stats['hard_threshold'] = df_stats['hard_rolling_mean'] + k * df_stats['hard_rolling_std']
    
    df_stats['soft_candidate'] = df_stats['soft_flux'] > df_stats['soft_threshold']
    df_stats['hard_candidate'] = df_stats['hard_flux'] > df_stats['hard_threshold']
    
    # 3. Label continuous candidate segments
    # Group contiguous True blocks for soft channel
    soft_blocks = (df_stats['soft_candidate'] != df_stats['soft_candidate'].shift()).cumsum()
    df_stats['soft_block_id'] = np.where(df_stats['soft_candidate'], soft_blocks, 0)
    
    # Group contiguous True blocks for hard channel
    hard_blocks = (df_stats['hard_candidate'] != df_stats['hard_candidate'].shift()).cumsum()
    df_stats['hard_block_id'] = np.where(df_stats['hard_candidate'], hard_blocks, 0)
    
    # Master catalogue of events
    catalogue = []
    
    # Final detection flag in dataframe
    df_stats['nowcast_detected'] = False
    
    # For each soft candidate block, check if there's a hard candidate block that overlaps
    # or is close in time (within ±5 minutes)
    unique_soft_blocks = df_stats['soft_block_id'].unique()
    unique_soft_blocks = unique_soft_blocks[unique_soft_blocks > 0] # exclude 0 (background)
    
    for s_id in unique_soft_blocks:
        block_df = df_stats[df_stats['soft_block_id'] == s_id]
        start_time = block_df.index.min()
        end_time = block_df.index.max()
        
        # Define search window for hard X-ray component (from start_time - 5min to end_time + 5min)
        search_start = start_time - pd.Timedelta(minutes=5)
        search_end = end_time + pd.Timedelta(minutes=5)
        
        # Check if hard channel has any candidate in this range
        hard_in_window = df_stats.loc[search_start:search_end]
        overlapping_hard = hard_in_window[hard_in_window['hard_candidate'] == True]
        
        if not overlapping_hard.empty:
            # Cross-validated flare event detected!
            # Find peak flux in soft channel
            peak_idx = block_df['soft_flux'].idxmax()
            peak_flux = block_df.loc[peak_idx, 'soft_flux']
            peak_hard_flux = block_df['hard_flux'].max()
            
            # GOES-style classification
            flare_class, broad_class = classify_flare(peak_flux)
            
            # Mark the dataframe
            df_stats.loc[start_time:end_time, 'nowcast_detected'] = True
            
            # Confidence calculation:
            # - Peak Signal-to-Noise Ratio (SNR) in soft channel
            baseline = block_df.loc[start_time, 'soft_rolling_mean']
            noise_std = block_df.loc[start_time, 'soft_rolling_std']
            snr = (peak_flux - baseline) / max(noise_std, 0.1)
            
            # - Correlation between soft and hard channels during the flare
            # We take the wider window of active flare (from start of soft to end of soft)
            event_data = df_stats.loc[start_time:end_time]
            if len(event_data) > 2:
                corr = event_data['soft_flux'].corr(event_data['hard_flux'])
                if np.isnan(corr):
                    corr = 0.5
            else:
                corr = 0.5
                
            # Combine SNR and correlation into a confidence score between 0 and 1
            snr_score = min(1.0, max(0.0, snr / 15.0))
            corr_score = max(0.0, corr)
            confidence = 0.4 * snr_score + 0.6 * corr_score
            confidence = min(1.0, max(0.1, confidence)) # bound between 0.1 and 1.0
            
            catalogue.append({
                'flare_id': f"FL-{start_time.strftime('%Y%m%d-%H%M%S')}",
                'start_time': start_time,
                'peak_time': peak_idx,
                'end_time': end_time,
                'duration_mins': (end_time - start_time).total_seconds() / 60.0,
                'peak_soft_flux': peak_flux,
                'peak_hard_flux': peak_hard_flux,
                'class': flare_class,
                'broad_class': broad_class,
                'confidence': confidence
            })
            
    # Convert catalogue to DataFrame
    if catalogue:
        cat_df = pd.DataFrame(catalogue)
        # Sort by start_time
        cat_df = cat_df.sort_values('start_time').reset_index(drop=True)
    else:
        cat_df = pd.DataFrame(columns=[
            'flare_id', 'start_time', 'peak_time', 'end_time', 
            'duration_mins', 'peak_soft_flux', 'peak_hard_flux', 
            'class', 'broad_class', 'confidence'
        ])
        
    return df_stats, cat_df

if __name__ == "__main__":
    from ingestion import generate_synthetic_data
    print("Generating synthetic data for testing nowcasting...")
    df = generate_synthetic_data(duration_days=2)
    
    print("Running flare detection...")
    df_detected, cat = detect_flares(df, k=3.5)
    print(f"Detected {len(cat)} flares.")
    if not cat.empty:
        print("\nDetected Flare Catalogue:")
        print(cat[['flare_id', 'start_time', 'peak_time', 'class', 'confidence']])
