"""
Local Explanation Plots: Waterfall and Force plots for individual predictions.

Explains HOW a single prediction was made by showing each feature's contribution.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional, Union


def plot_waterfall(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    base_value: float,
    observation_idx: int,
    top_n: int = 10,
    title: Optional[str] = None,
    height: int = 500
) -> go.Figure:
    """
    Create waterfall chart showing feature contributions for a single prediction.

    Shows how each feature pushes the prediction from the base value to the
    final prediction, making individual predictions interpretable.

    Parameters:
        shap_values: SHAP values array (n_samples, n_features)
        X: Feature matrix
        feature_names: List of feature names
        base_value: Base/expected value (average model output)
        observation_idx: Index of observation to explain
        top_n: Number of top features to show (rest grouped as "other")
        title: Plot title
        height: Plot height

    Returns:
        Plotly figure
    """
    # Get SHAP values and feature values for this observation
    obs_shap = shap_values[observation_idx]
    obs_features = X.iloc[observation_idx] if hasattr(X, 'iloc') else X[observation_idx]

    # Calculate final prediction
    final_prediction = base_value + obs_shap.sum()

    # Sort features by absolute SHAP contribution
    sorted_idx = np.argsort(np.abs(obs_shap))[::-1]

    # Take top N features
    top_idx = sorted_idx[:top_n]
    other_idx = sorted_idx[top_n:]

    # Build waterfall data
    labels = ['Base Value']
    values = [base_value]
    measures = ['absolute']
    hover_texts = [f'Average model output: {base_value:.4f}']

    cumulative = base_value

    for idx in top_idx:
        feat_name = feature_names[idx]
        feat_val = obs_features.iloc[idx] if hasattr(obs_features, 'iloc') else obs_features[idx]
        shap_val = obs_shap[idx]

        # Format feature value for display
        if isinstance(feat_val, (int, np.integer)):
            val_str = str(int(feat_val))
        elif isinstance(feat_val, (float, np.floating)):
            val_str = f'{feat_val:.2f}'
        else:
            val_str = str(feat_val)

        labels.append(f'{feat_name} = {val_str}')
        values.append(shap_val)
        measures.append('relative')

        cumulative += shap_val
        direction = "increases" if shap_val > 0 else "decreases"
        hover_texts.append(
            f'{feat_name} = {val_str}<br>'
            f'SHAP: {shap_val:+.4f}<br>'
            f'{direction} prediction<br>'
            f'Running total: {cumulative:.4f}'
        )

    # Add "other" features if any
    if len(other_idx) > 0:
        other_sum = obs_shap[other_idx].sum()
        if abs(other_sum) > 0.001:  # Only show if meaningful
            labels.append(f'Other ({len(other_idx)} features)')
            values.append(other_sum)
            measures.append('relative')
            cumulative += other_sum
            hover_texts.append(
                f'Sum of {len(other_idx)} other features<br>'
                f'SHAP: {other_sum:+.4f}<br>'
                f'Running total: {cumulative:.4f}'
            )

    # Add final prediction
    labels.append('Prediction')
    values.append(final_prediction)
    measures.append('total')
    hover_texts.append(f'Final prediction: {final_prediction:.4f}')

    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name='SHAP',
        orientation='v',
        measure=measures,
        x=labels,
        y=values,
        text=[f'{v:+.3f}' if m == 'relative' else f'{v:.3f}'
              for v, m in zip(values, measures)],
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text',
        connector={'line': {'color': 'rgb(63, 63, 63)'}},
        decreasing={'marker': {'color': '#EF553B'}},
        increasing={'marker': {'color': '#636EFA'}},
        totals={'marker': {'color': '#00CC96'}}
    ))

    if title is None:
        title = f'Prediction Explanation (Observation #{observation_idx})'

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        yaxis_title='Prediction Value',
        height=height,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=120),
        xaxis=dict(tickangle=-45)
    )

    return fig


def plot_force_horizontal(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    base_value: float,
    observation_idx: int,
    top_n: int = 10,
    title: Optional[str] = None,
    height: int = 400
) -> go.Figure:
    """
    Create horizontal force plot showing feature contributions.

    Alternative to waterfall - shows positive/negative contributions side by side.

    Parameters:
        shap_values: SHAP values array (n_samples, n_features)
        X: Feature matrix
        feature_names: List of feature names
        base_value: Base/expected value
        observation_idx: Index of observation to explain
        top_n: Number of features to show
        title: Plot title
        height: Plot height

    Returns:
        Plotly figure
    """
    obs_shap = shap_values[observation_idx]
    obs_features = X.iloc[observation_idx] if hasattr(X, 'iloc') else X[observation_idx]

    final_prediction = base_value + obs_shap.sum()

    # Sort by SHAP value (not absolute)
    sorted_idx = np.argsort(obs_shap)

    # Take top positive and top negative
    positive_idx = sorted_idx[obs_shap[sorted_idx] > 0][-top_n//2:][::-1]
    negative_idx = sorted_idx[obs_shap[sorted_idx] < 0][:top_n//2]

    # Combine and maintain order
    show_idx = list(negative_idx) + list(positive_idx)

    # Build data
    labels = []
    shap_vals = []
    colors = []

    for idx in show_idx:
        feat_name = feature_names[idx]
        feat_val = obs_features.iloc[idx] if hasattr(obs_features, 'iloc') else obs_features[idx]
        shap_val = obs_shap[idx]

        if isinstance(feat_val, (int, np.integer)):
            val_str = str(int(feat_val))
        elif isinstance(feat_val, (float, np.floating)):
            val_str = f'{feat_val:.2f}'
        else:
            val_str = str(feat_val)

        labels.append(f'{feat_name} = {val_str}')
        shap_vals.append(shap_val)
        colors.append('#636EFA' if shap_val > 0 else '#EF553B')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels,
        x=shap_vals,
        orientation='h',
        marker_color=colors,
        text=[f'{v:+.3f}' for v in shap_vals],
        textposition='outside',
        hovertemplate='%{y}<br>SHAP: %{x:+.4f}<extra></extra>'
    ))

    # Add base value and prediction annotations
    fig.add_vline(x=0, line_dash='dash', line_color='gray')

    if title is None:
        title = f'Force Plot (Observation #{observation_idx})'

    fig.update_layout(
        title=dict(
            text=f'{title}<br><sub>Base: {base_value:.3f} → Prediction: {final_prediction:.3f}</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='SHAP Value (impact on prediction)',
        yaxis_title='',
        height=height,
        margin=dict(l=200, r=100, t=100, b=50),
        showlegend=False
    )

    return fig


def get_observation_explanation(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    base_value: float,
    observation_idx: int,
    top_n: int = 10
) -> Dict:
    """
    Get structured explanation for a single observation.

    Parameters:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: List of feature names
        base_value: Base value
        observation_idx: Index of observation
        top_n: Number of top features to include

    Returns:
        Dictionary with explanation details
    """
    obs_shap = shap_values[observation_idx]
    obs_features = X.iloc[observation_idx] if hasattr(X, 'iloc') else X[observation_idx]

    final_prediction = base_value + obs_shap.sum()

    # Sort by absolute contribution
    sorted_idx = np.argsort(np.abs(obs_shap))[::-1][:top_n]

    contributions = []
    for idx in sorted_idx:
        feat_name = feature_names[idx]
        feat_val = obs_features.iloc[idx] if hasattr(obs_features, 'iloc') else obs_features[idx]
        shap_val = obs_shap[idx]

        contributions.append({
            'feature': feat_name,
            'value': feat_val,
            'shap': float(shap_val),
            'direction': 'positive' if shap_val > 0 else 'negative',
            'abs_contribution': float(abs(shap_val))
        })

    return {
        'observation_idx': observation_idx,
        'base_value': float(base_value),
        'prediction': float(final_prediction),
        'total_shap': float(obs_shap.sum()),
        'contributions': contributions
    }


def explain_observation_text(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    base_value: float,
    observation_idx: int,
    top_n: int = 5
) -> str:
    """
    Generate text explanation for an observation.

    Parameters:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: List of feature names
        base_value: Base value
        observation_idx: Index of observation
        top_n: Number of top features

    Returns:
        Human-readable explanation string
    """
    explanation = get_observation_explanation(
        shap_values, X, feature_names, base_value, observation_idx, top_n
    )

    lines = []
    lines.append(f"Observation #{observation_idx} Explanation")
    lines.append("=" * 50)
    lines.append(f"Base prediction (average): {explanation['base_value']:.3f}")
    lines.append(f"Final prediction: {explanation['prediction']:.3f}")
    lines.append(f"Total SHAP adjustment: {explanation['total_shap']:+.3f}")
    lines.append("")
    lines.append("Top Contributing Features:")
    lines.append("-" * 50)

    for c in explanation['contributions']:
        direction = "increases" if c['direction'] == 'positive' else "decreases"
        lines.append(
            f"  {c['feature']} = {c['value']}"
            f"  ({c['shap']:+.4f}, {direction} prediction)"
        )

    return "\n".join(lines)


def plot_multiple_observations(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    base_value: float,
    observation_indices: List[int],
    top_n: int = 8,
    height_per_obs: int = 300
) -> go.Figure:
    """
    Create comparison plot for multiple observations.

    Parameters:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: List of feature names
        base_value: Base value
        observation_indices: List of observation indices to compare
        top_n: Number of features per observation
        height_per_obs: Height per observation subplot

    Returns:
        Plotly figure with subplots
    """
    from plotly.subplots import make_subplots

    n_obs = len(observation_indices)

    fig = make_subplots(
        rows=n_obs, cols=1,
        subplot_titles=[f'Observation #{idx}' for idx in observation_indices],
        vertical_spacing=0.1
    )

    for i, obs_idx in enumerate(observation_indices):
        obs_shap = shap_values[obs_idx]
        obs_features = X.iloc[obs_idx] if hasattr(X, 'iloc') else X[obs_idx]
        final_pred = base_value + obs_shap.sum()

        # Get top features
        sorted_idx = np.argsort(np.abs(obs_shap))[::-1][:top_n]

        labels = []
        values = []
        colors = []

        for idx in sorted_idx:
            feat_name = feature_names[idx]
            feat_val = obs_features.iloc[idx] if hasattr(obs_features, 'iloc') else obs_features[idx]
            shap_val = obs_shap[idx]

            if isinstance(feat_val, (int, np.integer)):
                val_str = str(int(feat_val))
            elif isinstance(feat_val, (float, np.floating)):
                val_str = f'{feat_val:.1f}'
            else:
                val_str = str(feat_val)[:10]

            labels.append(f'{feat_name}={val_str}')
            values.append(shap_val)
            colors.append('#636EFA' if shap_val > 0 else '#EF553B')

        fig.add_trace(
            go.Bar(
                y=labels,
                x=values,
                orientation='h',
                marker_color=colors,
                showlegend=False,
                hovertemplate='%{y}<br>SHAP: %{x:+.4f}<extra></extra>'
            ),
            row=i+1, col=1
        )

        # Add prediction annotation - use paper reference for positioning
        # Annotations with subplot domains can be tricky, using simple approach
        max_abs_shap = max(abs(v) for v in values) if values else 0.1
        fig.add_annotation(
            x=max_abs_shap * 0.9,
            y=len(labels) - 0.5,
            xref=f'x{i+1}', yref=f'y{i+1}',
            text=f'Pred: {final_pred:.3f}',
            showarrow=False,
            font=dict(size=11, color='black'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )

    fig.update_layout(
        title=dict(text='Observation Comparison', x=0.5, xanchor='center'),
        height=height_per_obs * n_obs,
        showlegend=False,
        margin=dict(l=150, r=100, t=80, b=50)
    )

    for i in range(n_obs):
        fig.update_xaxes(title_text='SHAP Value', row=i+1, col=1)

    return fig
