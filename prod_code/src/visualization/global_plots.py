"""
Global interpretation visualizations: feature importance, distributions.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional, Dict
import warnings


def plot_global_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
    title: str = "Global Feature Importance",
    height: int = 500,
    show_values: bool = True
) -> go.Figure:
    """
    Plot global feature importance using mean absolute SHAP values.

    Parameters:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: Feature names
        top_n: Number of top features to display
        title: Plot title
        height: Plot height in pixels
        show_values: Show importance values on bars

    Returns:
        Plotly figure object
    """
    # Calculate mean absolute SHAP values
    importance = np.abs(shap_values).mean(axis=0)

    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Take top N
    top_features = importance_df.head(top_n)

    # Create horizontal bar chart (reverse order for top-to-bottom)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top_features['feature'][::-1],
        x=top_features['importance'][::-1],
        orientation='h',
        marker=dict(
            color=top_features['importance'][::-1],
            colorscale='Blues',
            showscale=False
        ),
        text=top_features['importance'][::-1].round(4) if show_values else None,
        textposition='outside' if show_values else None,
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        height=height,
        margin=dict(l=200, r=50, t=80, b=50),
        hovermode='closest',
        plot_bgcolor='white',
        showlegend=False
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=False)

    return fig


def plot_feature_distribution(
    X: pd.DataFrame,
    feature_name: str,
    show_percentiles: bool = True,
    bins: int = 50,
    title: Optional[str] = None,
    height: int = 400
) -> go.Figure:
    """
    Plot distribution of a single feature with percentile markers.

    Parameters:
        X: Feature matrix
        feature_name: Name of feature to plot
        show_percentiles: Show 25th, 50th, 75th percentile lines
        bins: Number of histogram bins (for numeric features)
        title: Plot title (default: "Distribution of {feature_name}")
        height: Plot height in pixels

    Returns:
        Plotly figure object
    """
    if feature_name not in X.columns:
        raise ValueError(f"Feature '{feature_name}' not found in data")

    if title is None:
        title = f"Distribution of {feature_name}"

    feature_data = X[feature_name].dropna()

    # Check if numeric or categorical
    is_numeric = pd.api.types.is_numeric_dtype(feature_data)

    fig = go.Figure()

    if is_numeric:
        # Histogram for numeric features
        fig.add_trace(go.Histogram(
            x=feature_data,
            nbinsx=bins,
            marker=dict(color='steelblue', line=dict(color='white', width=1)),
            hovertemplate='Value: %{x}<br>Count: %{y}<extra></extra>',
            name='Distribution'
        ))

        # Add percentile lines
        if show_percentiles:
            percentiles = [25, 50, 75]
            colors = ['orange', 'red', 'orange']
            for pct, color in zip(percentiles, colors):
                value = np.percentile(feature_data, pct)
                fig.add_vline(
                    x=value,
                    line=dict(color=color, dash='dash', width=2),
                    annotation=dict(
                        text=f"P{pct}: {value:.2f}",
                        font=dict(size=10, color=color)
                    )
                )

        fig.update_xaxes(title_text=feature_name)
        fig.update_yaxes(title_text="Count")

    else:
        # Bar chart for categorical features
        value_counts = feature_data.value_counts().sort_values(ascending=False)

        fig.add_trace(go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            marker=dict(color='steelblue'),
            hovertemplate='%{x}<br>Count: %{y}<extra></extra>',
            name='Count'
        ))

        fig.update_xaxes(title_text=feature_name, type='category')
        fig.update_yaxes(title_text="Count")

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        height=height,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='closest',
        plot_bgcolor='white',
        showlegend=False
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def plot_importance_comparison(
    shap_values_list: List[np.ndarray],
    feature_names: List[str],
    group_names: List[str],
    top_n: int = 10,
    title: str = "Feature Importance Comparison",
    height: int = 500
) -> go.Figure:
    """
    Compare feature importance across multiple groups/segments.

    Parameters:
        shap_values_list: List of SHAP value arrays (one per group)
        feature_names: Feature names
        group_names: Names of groups being compared
        top_n: Number of top features to display
        title: Plot title
        height: Plot height in pixels

    Returns:
        Plotly figure object
    """
    # Calculate importance for each group
    importance_dfs = []
    for i, shap_vals in enumerate(shap_values_list):
        importance = np.abs(shap_vals).mean(axis=0)
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'group': group_names[i]
        })
        importance_dfs.append(df)

    # Combine and get overall top features
    all_importance = pd.concat(importance_dfs)
    overall_top = (all_importance.groupby('feature')['importance']
                   .mean()
                   .sort_values(ascending=False)
                   .head(top_n)
                   .index.tolist())

    # Filter to top features
    plot_data = all_importance[all_importance['feature'].isin(overall_top)]

    # Create grouped bar chart
    fig = go.Figure()

    for group in group_names:
        group_data = plot_data[plot_data['group'] == group].set_index('feature').loc[overall_top[::-1]]
        fig.add_trace(go.Bar(
            y=group_data.index,
            x=group_data['importance'],
            name=group,
            orientation='h',
            hovertemplate='<b>%{y}</b><br>%{x:.4f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        height=height,
        margin=dict(l=200, r=50, t=80, b=80),
        hovermode='closest',
        plot_bgcolor='white',
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=False)

    return fig


def plot_categorical_breakdown(
    category_breakdown: Dict[str, Dict],
    category_name: str,
    top_n: int = 10,
    title: Optional[str] = None,
    height: int = 400
) -> go.Figure:
    """
    Plot breakdown of categorical feature SHAP contributions.

    Parameters:
        category_breakdown: Breakdown dict from aggregate_categorical_shap
        category_name: Name of categorical feature
        top_n: Number of top category values to show
        title: Plot title (default: "Breakdown of {category_name}")
        height: Plot height in pixels

    Returns:
        Plotly figure object
    """
    if category_name not in category_breakdown:
        warnings.warn(f"Category '{category_name}' not found in breakdown data")
        return go.Figure()

    if title is None:
        title = f"Breakdown of {category_name}"

    info = category_breakdown[category_name]

    # Create DataFrame from member importance
    df = pd.DataFrame([
        {
            'category_value': member.replace(f'{category_name}_', ''),
            'importance': importance,
            'pct_of_total': importance / info['total_importance'] * 100
        }
        for member, importance in info['member_importance'].items()
    ]).sort_values('importance', ascending=False).head(top_n)

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df['category_value'][::-1],
        x=df['importance'][::-1],
        orientation='h',
        marker=dict(color='steelblue'),
        text=df['pct_of_total'][::-1].round(1).astype(str) + '%',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<br>% of Total: %{text}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        height=height,
        margin=dict(l=150, r=50, t=80, b=50),
        hovermode='closest',
        plot_bgcolor='white',
        showlegend=False
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=False)

    return fig
