import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
import plotly.graph_objects as go
import plotly.express as px

def compute_binary_metrics(y_true, y_pred_bin):
    """
    Compute standard binary classification metrics: TPR, FAR, Precision, Recall, F1.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Alarm Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'TPR (Recall)': tpr,
        'FAR (False Alarm Rate)': far,
        'Precision': precision,
        'F1 Score': f1
    }

def calculate_average_lead_time(df, cat_df, y_pred_prob, threshold=0.5, horizon_mins=15, sample_rate_sec=10):
    """
    Calculate the average lead time (in minutes) before flare peaks.
    For each flare in the catalogue:
      Find the first time the forecast probability crossed 'threshold'
      within 'horizon_mins' before the flare peak.
      Lead time = Peak Time - First Trigger Time.
    """
    if cat_df.empty:
        return 0.0, 0
        
    lead_times = []
    horizon_steps = int(horizon_mins * 60 / sample_rate_sec)
    
    # Ensure y_pred_prob is aligned with df
    prob_series = pd.Series(y_pred_prob, index=df.index)
    
    for _, flare in cat_df.iterrows():
        peak_time = flare['peak_time']
        
        # Define search window for prediction: horizon_mins before peak up to the peak time
        search_start = peak_time - pd.Timedelta(minutes=horizon_mins)
        search_window = prob_series.loc[search_start:peak_time]
        
        # Find triggers in this window
        triggers = search_window[search_window >= threshold]
        
        if not triggers.empty:
            trigger_time = triggers.index.min()
            lead_time_min = (peak_time - trigger_time).total_seconds() / 60.0
            lead_times.append(lead_time_min)
            
    if lead_times:
        return np.mean(lead_times), len(lead_times)
    else:
        return 0.0, 0

def plot_confusion_matrix(y_true, y_pred_bin, title="Confusion Matrix"):
    """
    Generate an interactive Plotly Heatmap for the Confusion Matrix.
    Uses modern, premium dark aesthetics.
    """
    cm = confusion_matrix(y_true, y_pred_bin)
    
    # Style: Deep, elegant blue/purple gradient
    z = cm
    x = ['Predicted Quiet', 'Predicted Flare']
    y = ['Actual Quiet', 'Actual Flare']
    
    # Annotations
    annot = [
        [f"TN: {cm[0][0]}<br>(Correct Quiet)", f"FP: {cm[0][1]}<br>(False Alarm)"],
        [f"FN: {cm[1][0]}<br>(Missed Flare)", f"TP: {cm[1][1]}<br>(Correct Prediction)"]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        text=annot,
        texttemplate="%{text}",
        colorscale='Viridis',
        showscale=False
    ))
    
    fig.update_layout(
        title={
            'text': title,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'color': '#FFFFFF'}
        },
        paper_bgcolor='rgba(15, 23, 42, 0.9)', # Sleek dark slate
        plot_bgcolor='rgba(15, 23, 42, 0.9)',
        font=dict(color='#E2E8F0'),
        width=450,
        height=400,
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12)),
        margin=dict(l=60, r=30, t=80, b=40)
    )
    
    return fig

def plot_roc_curve(y_true, y_pred_prob, title="Receiver Operating Characteristic (ROC)"):
    """
    Generate an interactive Plotly ROC curve.
    Uses premium, vibrant aesthetics.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # ROC Curve line (Vibrant gold/amber)
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='#F59E0B', width=3),
        fill='tozeroy',
        fillcolor='rgba(245, 158, 11, 0.1)'
    ))
    
    # Random guess diagonal line (dashed grey)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Guess',
        line=dict(color='#64748B', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'color': '#FFFFFF'}
        },
        xaxis=dict(title='False Alarm Rate (FPR)', gridcolor='#334155', zeroline=False),
        yaxis=dict(title='True Positive Rate (TPR)', gridcolor='#334155', zeroline=False),
        paper_bgcolor='rgba(15, 23, 42, 0.9)',
        plot_bgcolor='rgba(15, 23, 42, 0.9)',
        font=dict(color='#E2E8F0'),
        width=500,
        height=400,
        legend=dict(x=0.5, y=0.15, xanchor='center', bgcolor='rgba(15, 23, 42, 0.8)'),
        margin=dict(l=60, r=30, t=80, b=60)
    )
    
    return fig
