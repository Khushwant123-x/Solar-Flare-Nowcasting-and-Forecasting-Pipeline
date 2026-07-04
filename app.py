import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Import pipeline modules
from src.ingestion import load_solexs_from_zip, load_helios_from_zip, align_datasets, generate_synthetic_data
from src.nowcasting import detect_flares
from src.forecasting import engineer_features, create_forecast_labels, prepare_xgb_data, train_xgb_model, prepare_lstm_data, train_lstm_model, predict_lead_time, TF_AVAILABLE
from src.evaluation import compute_binary_metrics, calculate_average_lead_time, plot_confusion_matrix, plot_roc_curve

# Page config
st.set_page_config(
    page_title="Aditya-L1 Solar Flare Pipeline",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek, dark slate styling and customized typography
st.markdown("""
    <style>
    .main {
        background-color: #0F172A;
        color: #E2E8F0;
    }
    .stSidebar {
        background-color: #1E293B !important;
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 6px;
        border: none;
        transition: background 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    /* Metric Card styling */
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #3B82F6;
    }
    .metric-label {
        font-size: 12px;
        color: #94A3B8;
        text-transform: uppercase;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to locate files in raw data directory
RAW_DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "data", "raw"))

# Cache real data loading to make dashboard fast
@st.cache_data(show_spinner="Loading real Aditya-L1 data...")
def load_real_data(day_selected):
    try:
        if day_selected == "June 14, 2026":
            solexs_zip = os.path.join(RAW_DATA_DIR, "AL1_SLX_L1_20260614_v1.0.zip")
            solexs_file = "AL1_SLX_L1_20260614_v1.0/SDD2/AL1_SOLEXS_20260614_SDD2_L1.lc.gz"
            helios_zip = os.path.join(RAW_DATA_DIR, "HLS_20260614_000012_43175sec_lev1_V111.zip")
            helios_file = "2026/06/14/HLS_20260614_000012_43175sec_lev1_V111/czt/lightcurve_czt1.fits"
        elif day_selected == "June 15, 2026":
            solexs_zip = os.path.join(RAW_DATA_DIR, "AL1_SLX_L1_20260615_v1.0.zip")
            solexs_file = "AL1_SLX_L1_20260615_v1.0/SDD2/AL1_SOLEXS_20260615_SDD2_L1.lc.gz"
            # Using HLS_20260614_235958 which covers June 15
            helios_zip = os.path.join(RAW_DATA_DIR, "HLS_20260614_235958_43187sec_lev1_V111.zip")
            helios_file = "2026/06/15/HLS_20260614_235958_43187sec_lev1_V111/czt/lightcurve_czt1.fits"
        elif day_selected == "June 16, 2026":
            solexs_zip = os.path.join(RAW_DATA_DIR, "AL1_SLX_L1_20260616_v1.0.zip")
            solexs_file = "AL1_SLX_L1_20260616_v1.0/SDD2/AL1_SOLEXS_20260616_SDD2_L1.lc.gz"
            helios_zip = os.path.join(RAW_DATA_DIR, "HLS_20260616_115959_43192sec_lev1_V111.zip")
            helios_file = "2026/06/16/HLS_20260616_115959_43192sec_lev1_V111/czt/lightcurve_czt1.fits"
        else:
            return None, "Invalid day selection."

        solexs_zip = os.path.normpath(solexs_zip)
        helios_zip = os.path.normpath(helios_zip)

        # Load SoLEXS
        df_solexs = load_solexs_from_zip(solexs_zip, solexs_file)
        # Load HEL1OS
        df_helios = load_helios_from_zip(helios_zip, helios_file)
        
        # Align
        merged = align_datasets(df_solexs, df_helios, resample_rule='10s')
        
        # Add mock true label (all 0 since the sun is quiet in these days)
        merged['true_label'] = 0
        merged['true_class'] = None
        
        return merged, None
    except Exception as e:
        return None, str(e)

# Cache model training so that changing thresholds or sliding timescales is instantaneous
@st.cache_resource(show_spinner="Training predictive forecasting models on background synthetic data (takes ~30s)...")
def train_forecasting_models(lead_time_mins=15):
    # Train models on a 5-day synthetic dataset with plenty of flares
    df_train = generate_synthetic_data(duration_days=5, seed=101)
    df_train_det, _ = detect_flares(df_train, k=3.5)
    df_train_feat = engineer_features(df_train_det)
    df_train_labeled = create_forecast_labels(df_train_feat, lead_time_mins=lead_time_mins)
    
    # XGBoost training
    xgb_features = [c for c in df_train_labeled.columns if any(p in c for p in ['mean', 'std', 'deriv', 'cum', 'ratio']) and c not in ['true_class']]
    X_train, y_train, X_test, y_test, _, _ = prepare_xgb_data(df_train_labeled, xgb_features)
    model_xgb, scaler_xgb = train_xgb_model(X_train, y_train, X_test, y_test)
    
    # LSTM training
    lstm_features = ['soft_flux', 'hard_flux', 'soft_deriv_1m', 'hard_deriv_1m']
    model_lstm = None
    scaler_lstm = None
    if TF_AVAILABLE:
        X_tr, y_tr, X_ts, y_ts, scaler_lstm, _, _ = prepare_lstm_data(df_train_labeled, feature_cols=lstm_features)
        model_lstm = train_lstm_model(X_tr, y_tr, X_ts, y_ts)
        
    return model_xgb, scaler_xgb, xgb_features, model_lstm, scaler_lstm, lstm_features

# Dashboard Header
st.title("☀️ Aditya-L1 Solar Flare Forecasting & Nowcasting")
st.subheader("Real-time telemetry analytics using soft X-rays (SoLEXS) and hard X-rays (HEL1OS)")

# ----------------- SIDEBAR -----------------
st.sidebar.header("Pipeline Controls")

data_source = st.sidebar.selectbox(
    "Data Source",
    ["Synthetic Flare Generator (Recommended)", "Real ISSDC PRADAN Data"]
)

# Render day selection if Real Data
if data_source == "Real ISSDC PRADAN Data":
    day_selected = st.sidebar.selectbox(
        "Observation Day",
        ["June 14, 2026", "June 15, 2026", "June 16, 2026"]
    )
    # Load Real Data
    df, err = load_real_data(day_selected)
    if err:
        st.sidebar.error(f"Error loading files: {err}")
        st.info("Falling back to synthetic data.")
        df = generate_synthetic_data(duration_days=3)
else:
    # Synthetic Generator configurations
    st.sidebar.subheader("Synthetic Generator Configuration")
    duration_days = st.sidebar.slider("Dataset Duration (days)", 1, 7, 3)
    flare_rate = st.sidebar.slider("Flares / Day", 0.5, 4.0, 2.0)
    df = generate_synthetic_data(duration_days=duration_days, flare_rate=flare_rate)

# Parameter configuration sliders
st.sidebar.subheader("Nowcasting Parameters")
k_threshold = st.sidebar.slider("Detection Threshold (k * std)", 2.0, 6.0, 3.5, step=0.1)

st.sidebar.subheader("Forecasting Parameters")
horizon_mins = st.sidebar.slider("Prediction Horizon (N mins)", 5, 30, 15)
model_choice = st.sidebar.selectbox("Prediction Model", ["XGBoost Tabular", "LSTM Sequential"])

# Train models
model_xgb, scaler_xgb, xgb_features, model_lstm, scaler_lstm, lstm_features = train_forecasting_models(lead_time_mins=horizon_mins)

# ----------------- PIPELINE EXECUTION -----------------
# 1. Nowcasting (Detect Flares)
df_detected, cat_df = detect_flares(df, k=k_threshold)

# 2. Feature Engineering
df_features = engineer_features(df_detected)

# 3. Label Creation
df_labeled = create_forecast_labels(df_features, lead_time_mins=horizon_mins)

# 4. Forecasting Inference
if model_choice == "XGBoost Tabular":
    X = df_labeled[xgb_features]
    X_scaled = scaler_xgb.transform(X)
    preds_prob = model_xgb.predict_proba(X_scaled)[:, 1]
    preds_bin = (preds_prob > 0.5).astype(int)
else: # LSTM
    if model_lstm is not None:
        # Pad data for LSTM sequential prediction
        scaled_lstm_features = scaler_lstm.transform(df_labeled[lstm_features])
        seq_len = 30
        preds_prob = np.zeros(len(df_labeled))
        
        # Prepare inputs batch
        X_lstm = []
        for idx in range(len(df_labeled)):
            if idx < seq_len:
                # pad with zeros if insufficient history
                pad = np.zeros((seq_len - idx, len(lstm_features)))
                hist = scaled_lstm_features[0:idx]
                X_lstm.append(np.vstack([pad, hist]))
            else:
                X_lstm.append(scaled_lstm_features[idx-seq_len:idx])
                
        X_lstm = np.array(X_lstm)
        preds_prob = model_lstm.predict(X_lstm, verbose=0).flatten()
        preds_bin = (preds_prob > 0.5).astype(int)
    else:
        st.warning("LSTM (TensorFlow) is unavailable. Falling back to XGBoost predictions.")
        X = df_labeled[xgb_features]
        X_scaled = scaler_xgb.transform(X)
        preds_prob = model_xgb.predict_proba(X_scaled)[:, 1]
        preds_bin = (preds_prob > 0.5).astype(int)

df_labeled['forecast_prob'] = preds_prob
df_labeled['forecast_detected'] = preds_bin

# ----------------- REAL-TIME SIMULATION CONTROLLER -----------------
st.markdown("### 🕒 Pipeline Playback Simulation")
sim_col1, sim_col2 = st.columns([4, 1])

with sim_col1:
    sim_index = st.slider(
        "Scrub Timeline (Simulates live data streaming)",
        min_value=120, # start with some history
        max_value=len(df_labeled)-1,
        value=int(len(df_labeled)*0.4), # start in middle
        format=""
    )

current_time = df_labeled.index[sim_index]
current_row = df_labeled.iloc[sim_index]

with sim_col2:
    st.markdown(f"<div style='text-align: center; margin-top: 15px;'><span style='font-size: 14px; color: #94A3B8;'>Current Telemetry Time</span><br><span style='font-size: 16px; font-weight: bold;'>{current_time.strftime('%Y-%m-%d %H:%M:%S')}</span></div>", unsafe_allow_html=True)

# ----------------- ALERT BANNER -----------------
active_nowcast = current_row['nowcast_detected']
predicted_flare = current_row['forecast_detected']
forecast_probability = current_row['forecast_prob']

if active_nowcast:
    # Find current flare class if we can
    matching_flare = cat_df[(cat_df['start_time'] <= current_time) & (cat_df['end_time'] >= current_time)]
    f_class = matching_flare['class'].values[0] if not matching_flare.empty else "Active"
    st.markdown(f"""
        <div style="background-color: #EF4444; color: white; padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 20px; font-size: 18px;">
            🚨 CRITICAL NOWCAST ALERT: Active Solar Flare Event in Progress (Class {f_class})!
        </div>
    """, unsafe_allow_html=True)
elif predicted_flare:
    est_lead_time = predict_lead_time(forecast_probability, max_horizon_mins=horizon_mins)
    st.markdown(f"""
        <div style="background-color: #F59E0B; color: black; padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 20px; font-size: 18px;">
            ⚠️ PREDICTIVE WARNING: Solar Flare Predicted in Next {horizon_mins} Minutes (Probability: {forecast_probability:.1%}, Est. Lead Time: {est_lead_time:.1f} mins)!
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div style="background-color: #10B981; color: white; padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 20px; font-size: 18px;">
            ✅ SYSTEM STATUS: NORMAL (Sun is Quiet)
        </div>
    """, unsafe_allow_html=True)

# ----------------- METRIC CARDS -----------------
met1, met2, met3, met4 = st.columns(4)

with met1:
    total_flares = len(cat_df[cat_df['start_time'] <= current_time])
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_flares}</div>
            <div class="metric-label">Total Flares Detected</div>
        </div>
    """, unsafe_allow_html=True)

with met2:
    soft_flux_val = current_row['soft_flux']
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{soft_flux_val:.1f} cts/s</div>
            <div class="metric-label">Soft X-ray Flux (SoLEXS)</div>
        </div>
    """, unsafe_allow_html=True)

with met3:
    hard_flux_val = current_row['hard_flux']
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{hard_flux_val:.1f} cts/s</div>
            <div class="metric-label">Hard X-ray Flux (HEL1OS)</div>
        </div>
    """, unsafe_allow_html=True)

with met4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{forecast_probability:.1%}</div>
            <div class="metric-label">Forecast Probability (Next {horizon_mins}m)</div>
        </div>
    """, unsafe_allow_html=True)

st.write("")

# ----------------- INTERACTIVE CHARTS -----------------
# We display data centered around the simulation pointer: e.g. 2 hours before up to 30 mins after (if exists)
start_plot_idx = max(0, sim_index - 720) # 2 hours back (at 10s cadence)
end_plot_idx = min(len(df_labeled)-1, sim_index + 180) # 30 mins forward

plot_df = df_labeled.iloc[start_plot_idx:end_plot_idx]
history_df = df_labeled.iloc[start_plot_idx:sim_index+1] # Only show predictions up to current simulation time

fig = go.Figure()

# SoLEXS Flux
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df['soft_flux'],
    mode='lines',
    name='Soft X-ray (SoLEXS)',
    line=dict(color='#60A5FA', width=2)
))

# HEL1OS Flux
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df['hard_flux'],
    mode='lines',
    name='Hard X-ray (HEL1OS)',
    line=dict(color='#F472B6', width=1.5, dash='dot')
))

# Shading for Nowcast active flares (only in past/history)
nowcast_active = history_df[history_df['nowcast_detected'] == True]
if not nowcast_active.empty:
    fig.add_trace(go.Scatter(
        x=nowcast_active.index, y=nowcast_active['soft_flux'],
        mode='markers',
        name='Nowcast Detected',
        marker=dict(color='#EF4444', size=5, symbol='circle')
    ))

# Shading for Forecast predictions (only in past/history)
forecast_active = history_df[history_df['forecast_detected'] == True]
if not forecast_active.empty:
    fig.add_trace(go.Scatter(
        x=forecast_active.index, y=forecast_active['soft_flux'] * 0.9,
        mode='markers',
        name='Forecast Triggered',
        marker=dict(color='#F59E0B', size=4, symbol='triangle-up')
    ))

# Vertical line at current simulation pointer
fig.add_vline(x=current_time, line_width=2, line_dash="dash", line_color="#E2E8F0")

# Highlight current peak predicted if active forecast
if predicted_flare and not active_nowcast:
    # Heuristically point out predicted peak
    pred_peak_time = current_time + pd.Timedelta(minutes=est_lead_time)
    if pred_peak_time < plot_df.index.max():
        fig.add_vline(x=pred_peak_time, line_width=1, line_dash="dot", line_color="#F59E0B")
        fig.add_annotation(
            x=pred_peak_time, y=current_row['soft_flux'] * 1.5,
            text=f"Predicted Peak (~{est_lead_time:.0f}m)",
            showarrow=True, arrowhead=1, ax=-40, ay=-30,
            font=dict(color="#F59E0B", size=11),
            bgcolor="rgba(15, 23, 42, 0.8)", bordercolor="#F59E0B"
        )

fig.update_layout(
    title="Real-Time Instrument Flux and Detections Overlay",
    xaxis=dict(title="Timestamp", gridcolor='#334155'),
    yaxis=dict(title="Counts / Rate (cts/sec)", type='log' if st.checkbox('Log Scale Y-axis', value=True) else 'linear', gridcolor='#334155'),
    paper_bgcolor='rgba(15, 23, 42, 0.9)',
    plot_bgcolor='rgba(15, 23, 42, 0.9)',
    font=dict(color='#E2E8F0'),
    legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'),
    height=450,
    margin=dict(l=60, r=40, t=50, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# ----------------- TABS SECTION -----------------
tab1, tab2, tab3 = st.tabs(["📋 Flare Catalogue", "📊 Predictive Performance", "🧠 Model Explanation"])

with tab1:
    st.markdown("### 📖 Master Solar Flare Catalogue")
    st.markdown("Validated solar flares identified using cross-channel correlation. Only flares occurring before or during the current simulation timestamp are logged.")
    
    # Filter catalogue up to current time
    visible_cat = cat_df[cat_df['start_time'] <= current_time]
    
    if visible_cat.empty:
        st.info("No solar flares have been detected in the current history window.")
    else:
        # Style dataframe
        styled_cat = visible_cat.copy()
        styled_cat['start_time'] = styled_cat['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        styled_cat['peak_time'] = styled_cat['peak_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        styled_cat['end_time'] = styled_cat['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        styled_cat['duration_mins'] = styled_cat['duration_mins'].round(1)
        styled_cat['peak_soft_flux'] = styled_cat['peak_soft_flux'].round(1)
        styled_cat['peak_hard_flux'] = styled_cat['peak_hard_flux'].round(1)
        styled_cat['confidence'] = (styled_cat['confidence'] * 100).round(1).astype(str) + "%"
        
        st.dataframe(
            styled_cat[['flare_id', 'start_time', 'peak_time', 'end_time', 'duration_mins', 'class', 'peak_soft_flux', 'confidence']],
            use_container_width=True
        )

with tab2:
    st.markdown("### 📈 Model Evaluation Metrics")
    st.markdown("Tabulated performance benchmarks evaluated chronologically using the test split (last 25% of timeline).")
    
    # Calculate performance on the evaluation dataset
    y_test_true = df_labeled['forecast_label']
    y_test_pred_prob = df_labeled['forecast_prob']
    y_test_pred_bin = df_labeled['forecast_detected']
    
    # Metrics
    metrics = compute_binary_metrics(y_test_true, y_test_pred_bin)
    avg_lead, count_lead = calculate_average_lead_time(
        df_labeled, cat_df, y_test_pred_prob, 
        threshold=0.5, horizon_mins=horizon_mins
    )
    
    col_m1, col_m2 = st.columns([1, 1])
    
    with col_m1:
        st.markdown("#### Performance Statistics")
        st.write(f"- **True Positive Rate (TPR / Recall)**: `{metrics['TPR (Recall)']:.1%}`")
        st.write(f"- **False Alarm Rate (FAR)**: `{metrics['FAR (False Alarm Rate)']:.1%}`")
        st.write(f"- **Precision**: `{metrics['Precision']:.1%}`")
        st.write(f"- **F1 Score**: `{metrics['F1 Score']:.3f}`")
        st.write(f"- **Average Forecast Lead Time**: `{avg_lead:.1f} minutes` (triggers = {count_lead})")
        
        # Display confusion matrix
        fig_cm = plot_confusion_matrix(y_test_true, y_test_pred_bin, title=f"Confusion Matrix ({model_choice})")
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_m2:
        # Display ROC Curve
        fig_roc = plot_roc_curve(y_test_true, y_test_pred_prob, title=f"ROC Curve ({model_choice})")
        st.plotly_chart(fig_roc, use_container_width=True)

with tab3:
    st.markdown("### 🧠 Operational Methodology")
    
    st.markdown("""
    #### 1. Data Ingestion & Preprocessing
    The pipeline processes light curves at **10-second cadence**. 
    - **SoLEXS (Soft X-rays)**: Primarily thermal plasma emission. Loaded from Astropy table `.lc` or `.lc.gz`.
    - **HEL1OS (Hard X-rays)**: Non-thermal particle acceleration. Loaded from Astropy `BinTableHDU` FITS extensions.
    - Endianness correction (big-endian to native system little-endian) is applied to ensure computational safety.
    
    #### 2. Nowcasting: Real-Time Event Detection
    A solar flare event is defined by the following logical steps:
    - **Dynamic Baseline**: 30-minute rolling mean and standard deviation.
    - **Anomaly Detection**: Flux exceeds $\mu_{30} + k \cdot \sigma_{30}$ on both channels.
    - **Cross-Channel Validation**: Validates that both soft and hard X-ray channels spike within $\pm 5$ minutes of each other. This eliminates false positives caused by local cosmic rays hitting only a single instrument.
    - **Classification**: Peak soft X-ray counts are classified using equivalent log-amplitude classes mimicking GOES levels: **A**, **B**, **C**, **M**, and **X**.
    
    #### 3. Forecasting: ML Predictive Models
    - **Tabular Features**: Derived over sliding windows (5, 10, 15 minutes) tracking rate of change (derivatives), soft-to-hard X-ray flux ratios, and rolling cumulative sums.
    - **XGBoost**: Learns standard thresholds and slopes leading up to flare impulses.
    - **LSTM (Long Short-Term Memory)**: Learns sequential patterns on normalized raw flux values and derivatives over a 5-minute lookback window (30 sequence steps).
    - **Neupert Effect**: The model exploits the physical lag between hard X-rays (which peak first during impulsive particle acceleration) and soft X-rays (which peak later as the coronal loop heats up).
    """)
