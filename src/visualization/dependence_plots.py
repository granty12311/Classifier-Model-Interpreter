"""
Dependence plots: beeswarm, scatter, partial dependence for feature analysis.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, List
import warnings


def plot_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 10,
    max_display: int = 1000,
    title: str = "SHAP Beeswarm Plot",
    height: int = 600
) -> go.Figure:
    """
    Create beeswarm plot showing SHAP value distributions for top features.

    Parameters:
        shap_values: SHAP values array (n_samples, n_features)
        X: Feature matrix (for feature values and coloring)
        feature_names: Feature names
        top_n: Number of top features to display
        max_display: Maximum samples to display (for performance)
        title: Plot title
        height: Plot height in pixels

    Returns:
        Plotly figure object
    """
    # Calculate feature importance
    importance = np.abs(shap_values).mean(axis=0)

    # Get top features (limit to actual number of features available)
    n_features = len(feature_names)
    top_n = min(top_n, n_features)
    top_indices = np.argsort(importance)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]

    # Sample if needed
    n_samples = shap_values.shape[0]
    if n_samples > max_display:
        sample_idx = np.random.choice(n_samples, max_display, replace=False)
    else:
        sample_idx = np.arange(n_samples)

    # Create figure
    fig = go.Figure()

    # Add trace for each feature
    for rank, (feat_idx, feat_name) in enumerate(zip(top_indices, top_features)):
        shap_vals = shap_values[sample_idx, feat_idx]
        feat_vals = X.iloc[sample_idx, feat_idx].values

        # Normalize feature values for coloring (0-1 range)
        if feat_vals.std() > 0:
            feat_vals_norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min())
        else:
            feat_vals_norm = np.zeros_like(feat_vals)

        # Add jitter to y-axis for visibility
        y_vals = np.ones(len(shap_vals)) * rank
        y_jitter = np.random.randn(len(shap_vals)) * 0.15

        fig.add_trace(go.Scatter(
            x=shap_vals,
            y=y_vals + y_jitter,
            mode='markers',
            marker=dict(
                size=4,
                color=feat_vals_norm,
                colorscale='RdBu_r',
                showscale=(rank == 0),  # Show colorbar only once
                colorbar=dict(
                    title="Feature<br>Value",
                    tickvals=[0, 0.5, 1],
                    ticktext=['Low', 'Mid', 'High']
                ),
                line=dict(width=0.5, color='white')
            ),
            name=feat_name,
            customdata=feat_vals,
            hovertemplate=f'<b>{feat_name}</b><br>SHAP: %{{x:.4f}}<br>Value: %{{customdata:.2f}}<extra></extra>',
            showlegend=False
        ))

    # Number of features actually displayed
    num_displayed = len(top_features)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="SHAP Value (impact on model output)",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(num_displayed)),
            ticktext=top_features,
            range=[-0.5, num_displayed - 0.5]
        ),
        height=height,
        margin=dict(l=200, r=50, t=80, b=50),
        hovermode='closest',
        plot_bgcolor='white'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=True, zerolinewidth=2, zerolinecolor='black')
    fig.update_yaxes(showgrid=False)

    return fig


def plot_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_name: str,
    feature_names: List[str],
    interaction_feature: Optional[str] = None,
    max_display: int = 1000,
    title: Optional[str] = None,
    height: int = 500
) -> go.Figure:
    """
    Create dependence plot showing relationship between feature value and SHAP value.

    Parameters:
        shap_values: SHAP values array (n_samples, n_features)
        X: Feature matrix
        feature_name: Feature to plot
        feature_names: All feature names
        interaction_feature: Optional feature for color coding (auto-detect if None)
        max_display: Maximum samples to display
        title: Plot title (default: "Dependence Plot: {feature_name}")
        height: Plot height in pixels

    Returns:
        Plotly figure object
    """
    if feature_name not in feature_names:
        raise ValueError(f"Feature '{feature_name}' not found in feature_names")

    if title is None:
        title = f"Dependence Plot: {feature_name}"

    feat_idx = feature_names.index(feature_name)

    # Sample if needed
    n_samples = shap_values.shape[0]
    if n_samples > max_display:
        sample_idx = np.random.choice(n_samples, max_display, replace=False)
    else:
        sample_idx = np.arange(n_samples)

    shap_vals = shap_values[sample_idx, feat_idx]
    feat_vals = X.iloc[sample_idx][feature_name].values

    # Auto-detect interaction feature if not specified
    if interaction_feature is None:
        # Find feature with highest correlation to SHAP values
        correlations = []
        for i, name in enumerate(feature_names):
            if name != feature_name:
                corr = np.abs(np.corrcoef(shap_values[sample_idx, i], shap_vals)[0, 1])
                correlations.append((corr, name))

        if correlations:
            correlations.sort(reverse=True)
            interaction_feature = correlations[0][1]

    # Get interaction feature values for coloring
    if interaction_feature and interaction_feature in feature_names:
        int_feat_idx = feature_names.index(interaction_feature)
        int_feat_vals = X.iloc[sample_idx][interaction_feature].values

        # Normalize for coloring
        if int_feat_vals.std() > 0:
            int_feat_vals_norm = (int_feat_vals - int_feat_vals.min()) / (int_feat_vals.max() - int_feat_vals.min())
        else:
            int_feat_vals_norm = np.zeros_like(int_feat_vals)

        color = int_feat_vals_norm
        colorbar_title = f"{interaction_feature}"
    else:
        color = 'steelblue'
        colorbar_title = None

    # Create scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=feat_vals,
        y=shap_vals,
        mode='markers',
        marker=dict(
            size=6,
            color=color,
            colorscale='Viridis' if interaction_feature else None,
            showscale=bool(interaction_feature),
            colorbar=dict(title=colorbar_title) if colorbar_title else None,
            line=dict(width=0.5, color='white')
        ),
        hovertemplate=f'<b>{feature_name}</b>: %{{x:.2f}}<br>SHAP: %{{y:.4f}}<extra></extra>',
        showlegend=False
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title=feature_name,
        yaxis_title=f"SHAP Value",
        height=height,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='closest',
        plot_bgcolor='white',
        showlegend=False
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=True, zerolinewidth=2, zerolinecolor='black')

    return fig


def plot_dependence_categorical(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_name: str,
    feature_names: List[str],
    value_labels: Optional[dict] = None,
    title: Optional[str] = None,
    height: int = 500
) -> go.Figure:
    """
    Create dependence plot for categorical (label-encoded) feature.

    Parameters:
        shap_values: SHAP values array
        X: Feature matrix
        feature_name: Categorical feature name
        feature_names: All feature names
        value_labels: Optional dict mapping encoded values to labels (e.g., {0: 'A', 1: 'B'})
        title: Plot title
        height: Plot height

    Returns:
        Plotly figure object
    """
    if feature_name not in feature_names:
        raise ValueError(f"Feature '{feature_name}' not found")

    if title is None:
        title = f"Categorical Dependence: {feature_name}"

    feat_idx = feature_names.index(feature_name)

    shap_vals = shap_values[:, feat_idx]
    feat_vals = X[feature_name].values

    # Get unique values
    unique_vals = sorted(X[feature_name].unique())

    # Create box plot data
    box_data = []
    for val in unique_vals:
        mask = feat_vals == val
        val_shap = shap_vals[mask]

        label = str(value_labels.get(val, val)) if value_labels else str(val)

        box_data.append(go.Box(
            y=val_shap,
            name=label,
            marker=dict(color='steelblue'),
            boxmean='sd',
            hovertemplate=f'<b>{label}</b><br>SHAP: %{{y:.4f}}<extra></extra>'
        ))

    fig = go.Figure(data=box_data)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title=feature_name,
        yaxis_title="SHAP Value",
        height=height,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='closest',
        plot_bgcolor='white',
        showlegend=False
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=True, zerolinewidth=2, zerolinecolor='black')

    return fig
