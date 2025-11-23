"""
Model performance visualizations: confusion matrix, ROC curve, metrics.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import Optional, Dict
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    height: int = 500
) -> go.Figure:
    """
    Plot confusion matrix heatmap.

    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (default: [0, 1])
        normalize: Normalize by row (show percentages)
        title: Plot title
        height: Plot height in pixels

    Returns:
        Plotly figure object
    """
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        text_format = '.2%'
    else:
        text_format = 'd'

    # Create annotations
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            if normalize:
                text = f"{value:.2%}"
            else:
                text = f"{value:,}"
            annotations.append(
                dict(
                    x=j, y=i,
                    text=text,
                    font=dict(color='white' if value > cm.max() / 2 else 'black', size=14),
                    showarrow=False
                )
            )

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Predicted {label}" for label in labels],
        y=[f"Actual {label}" for label in labels],
        colorscale='Blues',
        showscale=True,
        hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        annotations=annotations,
        height=height,
        margin=dict(l=100, r=100, t=100, b=100),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    height: int = 500
) -> go.Figure:
    """
    Plot ROC curve with AUC score.

    Parameters:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        title: Plot title
        height: Plot height in pixels

    Returns:
        Plotly figure object
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC = {roc_auc:.3f})',
        line=dict(color='steelblue', width=3),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=True
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=height,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='closest',
        plot_bgcolor='white',
        legend=dict(x=0.6, y=0.1)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[0, 1])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[0, 1])

    return fig


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC)

    Returns:
        Dictionary of metric values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }

    if y_proba is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            metrics['auc'] = auc(fpr, tpr)
        except:
            warnings.warn("Could not calculate AUC", UserWarning)

    return metrics


def plot_metrics_summary(
    metrics: Dict[str, float],
    title: str = "Model Performance Metrics",
    height: int = 400
) -> go.Figure:
    """
    Plot summary bar chart of classification metrics.

    Parameters:
        metrics: Dictionary of metric name -> value
        title: Plot title
        height: Plot height in pixels

    Returns:
        Plotly figure object
    """
    df = pd.DataFrame([
        {'metric': k.upper().replace('_', ' '), 'value': v}
        for k, v in metrics.items()
    ])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['metric'],
        y=df['value'],
        marker=dict(
            color=df['value'],
            colorscale='RdYlGn',
            cmin=0,
            cmax=1,
            showscale=False
        ),
        text=df['value'].round(3),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="",
        yaxis_title="Score",
        height=height,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='closest',
        plot_bgcolor='white',
        showlegend=False,
        yaxis=dict(range=[0, 1.1])
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    height: int = 500
) -> go.Figure:
    """
    Plot calibration curve showing predicted vs actual probabilities.

    Parameters:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        n_bins: Number of bins for calibration
        title: Plot title
        height: Plot height in pixels

    Returns:
        Plotly figure object
    """
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Assign predictions to bins
    bin_indices = np.digitize(y_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Calculate actual frequency in each bin
    bin_true = []
    bin_pred = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_true.append(y_true[mask].mean())
            bin_pred.append(y_proba[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_true.append(np.nan)
            bin_pred.append(bin_centers[i])
            bin_counts.append(0)

    bin_true = np.array(bin_true)
    bin_pred = np.array(bin_pred)

    fig = go.Figure()

    # Calibration curve
    valid = ~np.isnan(bin_true)
    fig.add_trace(go.Scatter(
        x=bin_pred[valid],
        y=bin_true[valid],
        mode='lines+markers',
        name='Calibration',
        line=dict(color='steelblue', width=3),
        marker=dict(size=8),
        hovertemplate='Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>'
    ))

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', width=2, dash='dash')
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Frequency",
        height=height,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='closest',
        plot_bgcolor='white',
        legend=dict(x=0.05, y=0.95)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[0, 1])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[0, 1])

    return fig
