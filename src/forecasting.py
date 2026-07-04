import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

def engineer_features(df, sample_rate_sec=10):
    """
    Engineer rolling features, derivatives, and ratios on sliding windows.
    Windows: 5 min, 10 min, 15 min.
    """
    df_feat = df.copy()
    
    # Calculate step sizes for different windows
    steps_5m = int(5 * 60 / sample_rate_sec)
    steps_10m = int(10 * 60 / sample_rate_sec)
    steps_15m = int(15 * 60 / sample_rate_sec)
    
    # Base features
    df_feat['flux_ratio'] = df_feat['soft_flux'] / (df_feat['hard_flux'] + 1e-5)
    
    # Rolling statistics
    for name, steps in [('5m', steps_5m), ('10m', steps_10m), ('15m', steps_15m)]:
        df_feat[f'soft_mean_{name}'] = df_feat['soft_flux'].rolling(window=steps, min_periods=1).mean()
        df_feat[f'soft_std_{name}'] = df_feat['soft_flux'].rolling(window=steps, min_periods=1).std().fillna(0.0)
        
        df_feat[f'hard_mean_{name}'] = df_feat['hard_flux'].rolling(window=steps, min_periods=1).mean()
        df_feat[f'hard_std_{name}'] = df_feat['hard_flux'].rolling(window=steps, min_periods=1).std().fillna(0.0)
        
        df_feat[f'ratio_mean_{name}'] = df_feat['flux_ratio'].rolling(window=steps, min_periods=1).mean()
        
    # Rate of change (derivatives) - e.g. 1-minute delta
    steps_1m = int(1 * 60 / sample_rate_sec)
    df_feat['soft_deriv_1m'] = df_feat['soft_flux'].diff(periods=steps_1m).fillna(0.0)
    df_feat['hard_deriv_1m'] = df_feat['hard_flux'].diff(periods=steps_1m).fillna(0.0)
    
    # Cumulative flux trends
    df_feat['soft_cum_15m'] = df_feat['soft_flux'].rolling(window=steps_15m, min_periods=1).sum()
    df_feat['hard_cum_15m'] = df_feat['hard_flux'].rolling(window=steps_15m, min_periods=1).sum()
    
    return df_feat

def create_forecast_labels(df, label_col='nowcast_detected', lead_time_mins=15, sample_rate_sec=10):
    """
    Create forward-looking labels: 1 if a flare starts/occurs in next lead_time_mins, else 0.
    """
    lead_steps = int(lead_time_mins * 60 / sample_rate_sec)
    
    # Reversing series to perform forward rolling operation
    # (rolling is backward-looking; on a reversed series, it looks forward)
    reversed_series = df[label_col].iloc[::-1]
    forward_max = reversed_series.rolling(window=lead_steps, min_periods=1).max().iloc[::-1]
    
    df_labeled = df.copy()
    df_labeled['forecast_label'] = forward_max.fillna(0.0).astype(int)
    return df_labeled

def prepare_xgb_data(df, feature_cols, target_col='forecast_label', split_ratio=0.75):
    """
    Prepare train/test split for XGBoost (chronological time-series split).
    """
    # Drop rows with NaNs (which can appear at the beginning due to diff/rolling)
    df_clean = df.dropna(subset=feature_cols + [target_col])
    
    split_idx = int(len(df_clean) * split_ratio)
    
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    return X_train, y_train, X_test, y_test, train_df.index, test_df.index

def train_xgb_model(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost Classifier.
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    return model, scaler

def prepare_lstm_data(df, sequence_length=30, feature_cols=['soft_flux', 'hard_flux'], target_col='forecast_label', split_ratio=0.75):
    """
    Prepare sequential datasets for LSTM training.
    """
    df_clean = df.dropna(subset=feature_cols + [target_col])
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clean[feature_cols])
    
    X, y = [], []
    for i in range(len(df_clean) - sequence_length):
        X.append(scaled_features[i : i + sequence_length])
        y.append(df_clean[target_col].iloc[i + sequence_length])
        
    X = np.array(X)
    y = np.array(y)
    
    # Chronological Split
    split_idx = int(len(X) * split_ratio)
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    # Extract timestamps corresponding to the targets
    target_timestamps = df_clean.index[sequence_length:]
    train_times = target_timestamps[:split_idx]
    test_times = target_timestamps[split_idx:]
    
    return X_train, y_train, X_test, y_test, scaler, train_times, test_times

def build_lstm_model(input_shape):
    """
    Build and compile an LSTM network.
    """
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_lstm_model(X_train, y_train, X_test, y_test):
    """
    Train the LSTM network.
    """
    if not TF_AVAILABLE:
        print("TensorFlow/Keras is not available. Skipping LSTM training.")
        return None
        
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )
    return model

def predict_lead_time(prob, max_horizon_mins=15):
    """
    Heuristic to estimate lead time based on the forecast probability.
    When probability is high, the flare is imminent (lead time is small).
    When probability is close to 0.5, lead time is closer to the horizon.
    """
    if prob < 0.5:
        return 0.0
    # Decaying lead time heuristic
    # maps prob from [0.5, 1.0] to lead_time from [max_horizon_mins, 1.0]
    lead_time = max_horizon_mins * (1.0 - (prob - 0.5) / 0.5)
    return max(1.0, lead_time)

if __name__ == "__main__":
    from ingestion import generate_synthetic_data
    from nowcasting import detect_flares
    
    print("Generating data...")
    df = generate_synthetic_data(duration_days=3)
    df_detected, cat = detect_flares(df, k=3.5)
    
    print("Engineering features...")
    df_features = engineer_features(df_detected)
    df_labeled = create_forecast_labels(df_features, lead_time_mins=15)
    
    feature_cols = [c for c in df_labeled.columns if any(p in c for p in ['mean', 'std', 'deriv', 'cum', 'ratio']) and c not in ['true_class']]
    print("Feature columns count:", len(feature_cols))
    
    X_train, y_train, X_test, y_test, train_idx, test_idx = prepare_xgb_data(df_labeled, feature_cols)
    print(f"XGB Train: {X_train.shape}, Test: {X_test.shape}")
    
    print("Training XGBoost...")
    model_xgb, scaler_xgb = train_xgb_model(X_train, y_train, X_test, y_test)
    print("XGBoost trained successfully.")
    
    if TF_AVAILABLE:
        print("\nTraining LSTM...")
        # Use raw fluxes and 1m derivatives for LSTM
        lstm_feats = ['soft_flux', 'hard_flux', 'soft_deriv_1m', 'hard_deriv_1m']
        X_tr, y_tr, X_ts, y_ts, scaler_lstm, tr_times, ts_times = prepare_lstm_data(df_labeled, feature_cols=lstm_feats)
        print(f"LSTM Train: {X_tr.shape}, Test: {X_ts.shape}")
        model_lstm = train_lstm_model(X_tr, y_tr, X_ts, y_ts)
        print("LSTM trained successfully.")
    else:
        print("\nTensorFlow not available, skipping LSTM test.")
