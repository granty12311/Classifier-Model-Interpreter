"""
Interaction plots: 2D scatter, heatmaps, contours, facets for feature interactions.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List
import warnings


def plot_interaction_scatter(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    feature_1: str,
    feature_2: str,
    max_display: int = 1000,
    title: Optional[str] = None,
    height: int = 600
) -> go.Figure:
    """
    Create 2D scatter plot showing interaction between two features.

    Parameters:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: Feature names
        feature_1: First feature name
        feature_2: Second feature name
        max_display: Maximum samples to display
        title: Plot title
        height: Plot height

    Returns:
        Plotly figure object
    """
    if feature_1 not in feature_names or feature_2 not in feature_names:
        raise ValueError(f"Features not found")

    if title is None:
        title = f"Interaction: {feature_1} × {feature_2}"

    idx_1 = feature_names.index(feature_1)
    idx_2 = feature_names.index(feature_2)

    # Sample if needed
    n_samples = shap_values.shape[0]
    if n_samples > max_display:
        sample_idx = np.random.choice(n_samples, max_display, replace=False)
    else:
        sample_idx = np.arange(n_samples)

    vals_1 = X.iloc[sample_idx][feature_1].values
    vals_2 = X.iloc[sample_idx][feature_2].values
    shap_1 = shap_values[sample_idx, idx_1]

    # Color by SHAP value of feature 1
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=vals_1,
        y=vals_2,
        mode='markers',
        marker=dict(
            size=6,
            color=shap_1,
            colorscale='RdBu_r',
            showscale=True,
            colorbar=dict(title=f"SHAP<br>{feature_1}"),
            line=dict(width=0.5, color='white'),
            cmin=-np.abs(shap_1).max(),
            cmax=np.abs(shap_1).max()
        ),
        hovertemplate=f'<b>{feature_1}</b>: %{{x:.2f}}<br><b>{feature_2}</b>: %{{y:.2f}}<br>SHAP: %{{marker.color:.4f}}<extra></extra>',
        showlegend=False
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title=feature_1,
        yaxis_title=feature_2,
        height=height,
        margin=dict(l=50, r=120, t=80, b=50),
        hovermode='closest',
        plot_bgcolor='white'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def plot_interaction_heatmap(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    feature_1: str,
    feature_2: str,
    bins: int = 10,
    title: Optional[str] = None,
    height: int = 600
) -> go.Figure:
    """
    Create heatmap showing average SHAP values across feature value combinations.

    Parameters:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: Feature names
        feature_1: First feature name
        feature_2: Second feature name
        bins: Number of bins for continuous features
        title: Plot title
        height: Plot height

    Returns:
        Plotly figure object
    """
    if feature_1 not in feature_names or feature_2 not in feature_names:
        raise ValueError(f"Features not found")

    if title is None:
        title = f"Interaction Heatmap: {feature_1} × {feature_2}"

    idx_1 = feature_names.index(feature_1)

    vals_1 = X[feature_1].values
    vals_2 = X[feature_2].values
    shap_1 = shap_values[:, idx_1]

    # Bin the features
    if X[feature_1].nunique() <= 10:
        # Categorical - map values to indices
        bins_1 = sorted(X[feature_1].unique())
        labels_1 = [str(b) for b in bins_1]
        value_to_idx_1 = {val: idx for idx, val in enumerate(bins_1)}
        binned_1 = np.array([value_to_idx_1[v] for v in vals_1])
    else:
        # Numeric
        bins_1 = np.linspace(vals_1.min(), vals_1.max(), bins + 1)
        binned_1 = np.digitize(vals_1, bins_1[:-1]) - 1
        labels_1 = [f'{bins_1[i]:.1f}-{bins_1[i+1]:.1f}' for i in range(len(bins_1) - 1)]

    if X[feature_2].nunique() <= 10:
        # Categorical - map values to indices
        bins_2 = sorted(X[feature_2].unique())
        labels_2 = [str(b) for b in bins_2]
        value_to_idx_2 = {val: idx for idx, val in enumerate(bins_2)}
        binned_2 = np.array([value_to_idx_2[v] for v in vals_2])
    else:
        # Numeric
        bins_2 = np.linspace(vals_2.min(), vals_2.max(), bins + 1)
        binned_2 = np.digitize(vals_2, bins_2[:-1]) - 1
        labels_2 = [f'{bins_2[i]:.1f}-{bins_2[i+1]:.1f}' for i in range(len(bins_2) - 1)]

    # Compute average SHAP per bin combination
    n_bins_1 = len(labels_1)
    n_bins_2 = len(labels_2)
    heatmap_data = np.zeros((n_bins_2, n_bins_1))
    counts = np.zeros((n_bins_2, n_bins_1))

    for i in range(len(vals_1)):
        b1 = int(binned_1[i])
        b2 = int(binned_2[i])

        if 0 <= b1 < n_bins_1 and 0 <= b2 < n_bins_2:
            heatmap_data[b2, b1] += shap_1[i]
            counts[b2, b1] += 1

    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_data = np.where(counts > 0, heatmap_data / counts, np.nan)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=labels_1,
        y=labels_2,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title=f"Avg SHAP<br>{feature_1}"),
        hovertemplate=f'{feature_1}: %{{x}}<br>{feature_2}: %{{y}}<br>Avg SHAP: %{{z:.4f}}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title=feature_1,
        yaxis_title=feature_2,
        height=height,
        margin=dict(l=50, r=120, t=80, b=50)
    )

    return fig


def plot_interaction_matrix(
    interaction_matrix: pd.DataFrame,
    title: str = "Feature Interaction Matrix",
    height: int = 700
) -> go.Figure:
    """
    Plot heatmap of feature interaction matrix.

    Parameters:
        interaction_matrix: Square DataFrame with interaction strengths
        title: Plot title
        height: Plot height

    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=interaction_matrix.values,
        x=interaction_matrix.columns.tolist(),
        y=interaction_matrix.index.tolist(),
        colorscale='Blues',
        colorbar=dict(title="Interaction<br>Strength"),
        hovertemplate='%{x} × %{y}<br>Strength: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="",
        yaxis_title="",
        height=height,
        margin=dict(l=150, r=50, t=80, b=150),
        xaxis=dict(tickangle=-45)
    )

    return fig


def plot_interaction_contour(
    model,
    X: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    n_grid: int = 50,
    other_features_percentile: float = 0.5,
    title: Optional[str] = None,
    height: int = 600,
    value_labels_1: Optional[dict] = None,
    value_labels_2: Optional[dict] = None
) -> go.Figure:
    """
    Create blocky heatmap showing predicted probability across two features.

    **Smart axis handling:**
    - Discrete features (<10 unique values): Shows ONLY actual values on axis (e.g., 0,10,20,30)
    - Continuous features: Uses n_grid points with auto-formatted, clean tick marks
    - All features: Blocky heatmap (no smoothing) for consistent interpretation

    Parameters:
        model: Trained model with predict_proba method
        X: Feature matrix
        feature_1: First feature name (X-axis)
        feature_2: Second feature name (Y-axis)
        n_grid: Number of grid points per axis (for continuous features, default=50)
        other_features_percentile: Value to use for other features (0.5 = median)
        title: Plot title
        height: Plot height
        value_labels_1: Optional dict mapping feature_1 values to labels (e.g., {0: 'professional', 1: 'student'})
        value_labels_2: Optional dict mapping feature_2 values to labels

    Returns:
        Plotly figure object
    """
    if title is None:
        title = f"Predicted Probability: {feature_1} × {feature_2}"

    # Detect if features are discrete (low cardinality)
    n_unique_1 = X[feature_1].nunique()
    n_unique_2 = X[feature_2].nunique()

    is_discrete_1 = n_unique_1 < 10
    is_discrete_2 = n_unique_2 < 10

    # Create grid based on feature type
    if is_discrete_1:
        feat1_grid = sorted(X[feature_1].unique())
    else:
        feat1_min, feat1_max = X[feature_1].min(), X[feature_1].max()
        feat1_grid = np.linspace(feat1_min, feat1_max, n_grid)

    if is_discrete_2:
        feat2_grid = sorted(X[feature_2].unique())
    else:
        feat2_min, feat2_max = X[feature_2].min(), X[feature_2].max()
        feat2_grid = np.linspace(feat2_min, feat2_max, n_grid)

    # Create base data point using percentiles
    base_point = X.quantile(other_features_percentile).to_frame().T

    # Generate predictions for grid
    n1, n2 = len(feat1_grid), len(feat2_grid)
    predictions = np.zeros((n2, n1))

    for i, val1 in enumerate(feat1_grid):
        for j, val2 in enumerate(feat2_grid):
            # Create data point
            point = base_point.copy()
            point[feature_1] = val1
            point[feature_2] = val2

            # Predict
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(point)[0, 1]  # Probability of positive class
            else:
                prob = model.predict(point)[0]

            predictions[j, i] = prob

    # Create labels/values for axes
    # For continuous features: use numeric values for clean auto-ticks
    # For discrete features: use string labels to show all actual values
    if is_discrete_1:
        if value_labels_1:
            x_data = [str(value_labels_1.get(v, v)) for v in feat1_grid]
        else:
            x_data = [str(int(v)) if v == int(v) else str(v) for v in feat1_grid]
        x_tickmode = 'linear'
        x_dtick = 1
    else:
        # Continuous: use numeric values, let Plotly auto-format ticks
        x_data = feat1_grid
        x_tickmode = 'auto'
        x_dtick = None

    if is_discrete_2:
        if value_labels_2:
            y_data = [str(value_labels_2.get(v, v)) for v in feat2_grid]
        else:
            y_data = [str(int(v)) if v == int(v) else str(v) for v in feat2_grid]
        y_tickmode = 'linear'
        y_dtick = 1
    else:
        # Continuous: use numeric values, let Plotly auto-format ticks
        y_data = feat2_grid
        y_tickmode = 'auto'
        y_dtick = None

    # Create heatmap (blocky for all - no smoothing)
    fig = go.Figure(data=go.Heatmap(
        x=x_data,
        y=y_data,
        z=predictions,
        colorscale='RdYlGn',
        colorbar=dict(title="Predicted<br>Probability"),
        hovertemplate=f'{feature_1}: %{{x}}<br>{feature_2}: %{{y}}<br>Probability: %{{z:.3f}}<extra></extra>',
        zsmooth=False  # Blocky appearance for all features
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis=dict(
            title=feature_1,
            tickmode=x_tickmode,
            dtick=x_dtick
        ),
        yaxis=dict(
            title=feature_2,
            tickmode=y_tickmode,
            dtick=y_dtick
        ),
        height=height,
        margin=dict(l=150, r=120, t=80, b=80)
    )

    return fig


def plot_interaction_surface_3d(
    model,
    X: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    n_grid: int = 30,
    other_features_percentile: float = 0.5,
    title: Optional[str] = None,
    height: int = 700,
    value_labels_1: Optional[dict] = None,
    value_labels_2: Optional[dict] = None
) -> go.Figure:
    """
    Create 3D surface plot showing predicted probability.

    **Smart axis handling:**
    - Discrete features (<10 unique values): Shows ONLY actual values on axis
    - Continuous features: Uses n_grid points with auto-formatted, clean tick marks
    - Interactive 3D visualization (rotate, zoom, hover)

    Parameters:
        model: Trained model with predict_proba method
        X: Feature matrix
        feature_1: First feature name (X-axis)
        feature_2: Second feature name (Y-axis)
        n_grid: Number of grid points per axis (for continuous features, default=30)
        other_features_percentile: Value to use for other features
        title: Plot title
        height: Plot height
        value_labels_1: Optional dict mapping feature_1 values to labels
        value_labels_2: Optional dict mapping feature_2 values to labels

    Returns:
        Plotly figure object
    """
    if title is None:
        title = f"3D Prediction Surface: {feature_1} × {feature_2}"

    # Detect if features are discrete (low cardinality)
    n_unique_1 = X[feature_1].nunique()
    n_unique_2 = X[feature_2].nunique()

    is_discrete_1 = n_unique_1 < 10
    is_discrete_2 = n_unique_2 < 10

    # Create grid based on feature type
    if is_discrete_1:
        feat1_grid = sorted(X[feature_1].unique())
    else:
        feat1_min, feat1_max = X[feature_1].min(), X[feature_1].max()
        feat1_grid = np.linspace(feat1_min, feat1_max, n_grid)

    if is_discrete_2:
        feat2_grid = sorted(X[feature_2].unique())
    else:
        feat2_min, feat2_max = X[feature_2].min(), X[feature_2].max()
        feat2_grid = np.linspace(feat2_min, feat2_max, n_grid)

    # Create base data point
    base_point = X.quantile(other_features_percentile).to_frame().T

    # Generate predictions
    n1, n2 = len(feat1_grid), len(feat2_grid)
    predictions = np.zeros((n2, n1))

    for i, val1 in enumerate(feat1_grid):
        for j, val2 in enumerate(feat2_grid):
            point = base_point.copy()
            point[feature_1] = val1
            point[feature_2] = val2

            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(point)[0, 1]
            else:
                prob = model.predict(point)[0]

            predictions[j, i] = prob

    # Create labels for axes
    if value_labels_1:
        x_ticktext = [str(value_labels_1.get(v, v)) for v in feat1_grid]
    else:
        x_ticktext = None

    if value_labels_2:
        y_ticktext = [str(value_labels_2.get(v, v)) for v in feat2_grid]
    else:
        y_ticktext = None

    # Create surface plot
    fig = go.Figure(data=go.Surface(
        x=feat1_grid,
        y=feat2_grid,
        z=predictions,
        colorscale='RdYlGn',
        colorbar=dict(title="Predicted<br>Probability"),
        hovertemplate=f'{feature_1}: %{{x:.2f}}<br>{feature_2}: %{{y:.2f}}<br>Probability: %{{z:.3f}}<extra></extra>'
    ))

    # Build scene configuration
    scene_config = dict(
        zaxis_title="Predicted Probability"
    )

    # Configure axis ticks based on feature type
    # Discrete: show all actual values with custom labels
    # Continuous: use auto-formatting for clean, interpretable ticks
    if is_discrete_1:
        scene_config['xaxis'] = dict(
            title=feature_1,
            tickmode='array',
            tickvals=feat1_grid,
            ticktext=x_ticktext if x_ticktext else [str(int(v)) if v == int(v) else str(v) for v in feat1_grid]
        )
    else:
        scene_config['xaxis'] = dict(
            title=feature_1,
            tickmode='auto'
        )

    if is_discrete_2:
        scene_config['yaxis'] = dict(
            title=feature_2,
            tickmode='array',
            tickvals=feat2_grid,
            ticktext=y_ticktext if y_ticktext else [str(int(v)) if v == int(v) else str(v) for v in feat2_grid]
        )
    else:
        scene_config['yaxis'] = dict(
            title=feature_2,
            tickmode='auto'
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=scene_config,
        height=height,
        margin=dict(l=0, r=0, t=80, b=0)
    )

    return fig


def plot_conditional_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    feature_name: str,
    condition_feature: str,
    n_bins: int = 4,
    title: Optional[str] = None,
    height: int = 600,
    value_labels: Optional[dict] = None
) -> go.Figure:
    """
    Create conditional dependence plot showing how feature effect varies by another feature.

    Shows dependence of feature_name on SHAP, with separate lines for each bin of condition_feature.
    If lines differ significantly, there's an interaction between the features.

    For discrete conditioning features (<10 unique values), uses actual values with optional labels.
    For continuous features, bins into quantiles.

    Parameters:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: Feature names
        feature_name: Feature to plot dependence for
        condition_feature: Feature to condition on (split into bins)
        n_bins: Number of bins for continuous condition_feature (ignored if discrete)
        title: Plot title
        height: Plot height
        value_labels: Optional dict mapping condition_feature values to labels (e.g., {0: 'student', 1: 'professional'})

    Returns:
        Plotly figure object
    """
    if title is None:
        title = f"Conditional Dependence: {feature_name} | {condition_feature}"

    feat_idx = feature_names.index(feature_name)
    cond_idx = feature_names.index(condition_feature)

    # Get values
    feat_vals = X[feature_name].values
    cond_vals = X[condition_feature].values
    shap_vals = shap_values[:, feat_idx]

    # Detect if features are discrete
    n_unique_feat = X[feature_name].nunique()
    n_unique_cond = X[condition_feature].nunique()

    is_discrete_feat = n_unique_feat < 10
    is_discrete_cond = n_unique_cond < 10

    # Bin the conditioning feature
    if is_discrete_cond:
        # Discrete - use actual unique values
        bins = sorted(X[condition_feature].unique())
        if value_labels:
            bin_labels = [str(value_labels.get(b, b)) for b in bins]
        else:
            bin_labels = [str(int(b)) if b == int(b) else str(b) for b in bins]
        n_bins_actual = len(bins)
    else:
        # Continuous - use quantiles
        quantiles = [X[condition_feature].quantile(q) for q in np.linspace(0, 1, n_bins + 1)]
        bins = quantiles
        bin_labels = [f'{quantiles[i]:.1f}-{quantiles[i+1]:.1f}' for i in range(n_bins)]
        n_bins_actual = n_bins

    # Create figure
    fig = go.Figure()

    colors = px.colors.qualitative.Set2[:n_bins_actual]

    for i in range(n_bins_actual):
        if is_discrete_cond:
            # Discrete - exact match
            if i < len(bins):
                mask = cond_vals == bins[i]
                label = bin_labels[i]
            else:
                continue
        else:
            # Continuous - range
            if i < len(bins) - 1:
                mask = (cond_vals >= bins[i]) & (cond_vals < bins[i + 1])
                label = bin_labels[i]
            else:
                continue

        if mask.sum() < 10:
            continue

        # Get data for this bin
        x_bin = feat_vals[mask]
        y_bin = shap_vals[mask]

        # Sort for trend line
        sort_idx = np.argsort(x_bin)
        x_sorted = x_bin[sort_idx]
        y_sorted = y_bin[sort_idx]

        # Add scatter
        fig.add_trace(go.Scatter(
            x=x_bin,
            y=y_bin,
            mode='markers',
            marker=dict(size=4, color=colors[i], opacity=0.4),
            name=f'{condition_feature}={label}',
            legendgroup=f'group{i}',
            hovertemplate=f'{feature_name}: %{{x:.2f}}<br>SHAP: %{{y:.4f}}<extra></extra>'
        ))

        # Add trend line
        try:
            from scipy.ndimage import gaussian_filter1d
            if len(y_sorted) > 10:
                y_smooth = gaussian_filter1d(y_sorted, sigma=max(2, len(y_sorted) // 30))
                fig.add_trace(go.Scatter(
                    x=x_sorted,
                    y=y_smooth,
                    mode='lines',
                    line=dict(color=colors[i], width=3),
                    name=f'{label} (trend)',
                    legendgroup=f'group{i}',
                    showlegend=False,
                    hoverinfo='skip'
                ))
        except:
            pass

    # Configure X-axis for discrete features
    xaxis_config = dict(
        title=feature_name,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )

    if is_discrete_feat:
        # Discrete feature - show only actual values
        unique_feat_vals = sorted(X[feature_name].unique())
        xaxis_config['tickmode'] = 'array'
        xaxis_config['tickvals'] = unique_feat_vals
        xaxis_config['ticktext'] = [str(int(v)) if v == int(v) else str(v) for v in unique_feat_vals]

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis=xaxis_config,
        yaxis_title=f"SHAP Value",
        height=height,
        margin=dict(l=50, r=50, t=100, b=50),
        hovermode='closest',
        plot_bgcolor='white',
        legend=dict(
            title=dict(text=f"{condition_feature}<br>bins"),
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02
        )
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=True, zerolinewidth=2, zerolinecolor='black')

    return fig
