"""
Segment Discovery: Find groups with distinct SHAP patterns.

Clusters observations by HOW the model explains them, not by their features.
This reveals behavioral segments that traditional demographic segmentation misses.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings


def discover_segments(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    y: Optional[np.ndarray] = None,
    n_segments: int = 4,
    method: str = 'kmeans',
    features_to_use: Optional[List[str]] = None,
    top_n_features: int = 10,
    min_segment_size: int = 50,
    random_state: int = 42
) -> Dict:
    """
    Discover segments based on SHAP value patterns.

    Clusters observations by their SHAP patterns (how the model explains them),
    not by their feature values. This reveals behavioral segments.

    Parameters:
        shap_values: SHAP values array (n_samples, n_features)
        X: Feature matrix
        feature_names: List of feature names
        y: Target variable (optional, for segment profiling)
        n_segments: Number of segments to discover
        method: Clustering method ('kmeans', 'hierarchical')
        features_to_use: Specific features to cluster on (default: top N by importance)
        top_n_features: Number of top features to use if features_to_use not specified
        min_segment_size: Minimum samples per segment
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with segment analysis:
        - 'n_segments': Number of segments found
        - 'segments': List of segment profiles
        - 'segment_labels': Array of segment assignments
        - 'feature_importance_by_segment': DataFrame comparing importance
    """
    n_samples, n_features = shap_values.shape

    # Select features to cluster on
    if features_to_use is None:
        # Use top N features by global importance
        importance = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(importance)[::-1][:top_n_features]
        features_to_use = [feature_names[i] for i in top_idx]
        feature_idx = top_idx
    else:
        feature_idx = [feature_names.index(f) for f in features_to_use]

    # Extract SHAP values for selected features
    shap_subset = shap_values[:, feature_idx]

    # Standardize SHAP values for clustering
    scaler = StandardScaler()
    shap_scaled = scaler.fit_transform(shap_subset)

    # Perform clustering
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_segments, random_state=random_state, n_init=10)
        labels = clusterer.fit_predict(shap_scaled)
    elif method == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        clusterer = AgglomerativeClustering(n_clusters=n_segments)
        labels = clusterer.fit_predict(shap_scaled)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'hierarchical'")

    # Check segment sizes and merge small segments if needed
    unique_labels, counts = np.unique(labels, return_counts=True)
    small_segments = unique_labels[counts < min_segment_size]

    if len(small_segments) > 0:
        warnings.warn(
            f"Segments {list(small_segments)} have fewer than {min_segment_size} samples. "
            f"Consider reducing n_segments.",
            UserWarning
        )

    # Profile each segment
    segments = []
    for seg_id in range(n_segments):
        mask = labels == seg_id
        seg_size = mask.sum()

        if seg_size == 0:
            continue

        # SHAP values for this segment
        seg_shap = shap_values[mask]
        seg_X = X.iloc[mask] if hasattr(X, 'iloc') else X[mask]

        # Calculate importance within segment
        seg_importance = np.abs(seg_shap).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': seg_importance,
            'mean_shap': seg_shap.mean(axis=0),
            'direction': ['positive' if s > 0 else 'negative' for s in seg_shap.mean(axis=0)]
        }).sort_values('importance', ascending=False)

        # Top drivers for this segment
        top_drivers = []
        for _, row in importance_df.head(5).iterrows():
            top_drivers.append({
                'feature': row['feature'],
                'importance': float(row['importance']),
                'mean_shap': float(row['mean_shap']),
                'direction': row['direction']
            })

        # Segment statistics
        seg_profile = {
            'segment_id': int(seg_id),
            'size': int(seg_size),
            'pct_of_total': float(seg_size / n_samples * 100),
            'top_drivers': top_drivers,
            'importance_df': importance_df
        }

        # Add prediction info if y provided
        if y is not None:
            seg_y = y[mask]
            seg_profile['avg_target'] = float(seg_y.mean())
            seg_profile['target_rate'] = float(seg_y.mean())

        # Generate segment description
        seg_profile['description'] = _generate_segment_description(top_drivers, seg_profile)

        segments.append(seg_profile)

    # Create comparison DataFrame
    comparison_data = []
    for seg in segments:
        row = {'segment_id': seg['segment_id'], 'size': seg['size']}
        for driver in seg['top_drivers'][:3]:
            row[f"{driver['feature']}_importance"] = driver['importance']
            row[f"{driver['feature']}_direction"] = driver['direction']
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Feature importance by segment matrix
    importance_matrix = pd.DataFrame(index=feature_names)
    for seg in segments:
        seg_id = seg['segment_id']
        importance_matrix[f'Segment_{seg_id}'] = seg['importance_df'].set_index('feature')['importance']

    return {
        'n_segments': len(segments),
        'method': method,
        'features_used': features_to_use,
        'segments': segments,
        'segment_labels': labels,
        'comparison_df': comparison_df,
        'importance_by_segment': importance_matrix
    }


def _generate_segment_description(top_drivers: List[Dict], profile: Dict) -> str:
    """Generate human-readable segment description."""

    if not top_drivers:
        return "No clear drivers identified"

    # Check if segment has strong vs weak drivers
    top_importance = top_drivers[0]['importance']
    second_importance = top_drivers[1]['importance'] if len(top_drivers) > 1 else 0

    if top_importance < 0.05:
        return "Weak signal segment - model has low confidence on these observations"

    primary = top_drivers[0]
    direction = "high" if primary['direction'] == 'positive' else "low"

    desc = f"{primary['feature'].replace('_', ' ').title()} Driven"

    # Add detail about dominant driver
    if top_importance > 2 * second_importance:
        desc += f" (strongly dominated by {primary['feature']})"
    elif len(top_drivers) >= 2:
        secondary = top_drivers[1]
        desc += f" (with {secondary['feature']})"

    return desc


def plot_segment_profiles(
    segment_results: Dict,
    top_n_features: int = 8,
    height: int = 500
):
    """
    Create visualization comparing segment profiles.

    Parameters:
        segment_results: Output from discover_segments()
        top_n_features: Number of features to show
        height: Plot height

    Returns:
        Plotly figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    segments = segment_results['segments']
    n_segments = len(segments)

    if n_segments == 0:
        raise ValueError("No segments to plot")

    # Get top features across all segments
    all_importance = segment_results['importance_by_segment']
    avg_importance = all_importance.mean(axis=1).sort_values(ascending=False)
    top_features = avg_importance.head(top_n_features).index.tolist()

    # Create subplot for each segment
    fig = make_subplots(
        rows=1, cols=n_segments,
        subplot_titles=[f"Segment {s['segment_id']}<br>({s['size']} obs, {s['pct_of_total']:.1f}%)"
                       for s in segments],
        shared_yaxes=True
    )

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']

    for i, seg in enumerate(segments):
        seg_importance = seg['importance_df'].set_index('feature')

        # Get values for top features
        values = [seg_importance.loc[f, 'importance'] if f in seg_importance.index else 0
                  for f in top_features]
        shap_means = [seg_importance.loc[f, 'mean_shap'] if f in seg_importance.index else 0
                      for f in top_features]

        # Color by direction
        bar_colors = ['#EF553B' if s > 0 else '#636EFA' for s in shap_means]

        fig.add_trace(
            go.Bar(
                y=top_features,
                x=values,
                orientation='h',
                marker_color=bar_colors,
                name=f'Segment {seg["segment_id"]}',
                showlegend=False,
                hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
            ),
            row=1, col=i+1
        )

    fig.update_layout(
        title=dict(
            text='Feature Importance by Discovered Segment<br><sub>Red = Positive SHAP (increases prediction), Blue = Negative SHAP (decreases prediction)</sub>',
            x=0.5,
            xanchor='center'
        ),
        height=height,
        showlegend=False,
        margin=dict(l=150, r=50, t=100, b=50)
    )

    # Update all x-axes
    for i in range(n_segments):
        fig.update_xaxes(title_text="Mean |SHAP|", row=1, col=i+1)

    return fig


def plot_segment_comparison(
    segment_results: Dict,
    feature_names_to_show: Optional[List[str]] = None,
    top_n: int = 10,
    height: int = 600
):
    """
    Create side-by-side comparison of feature importance across segments.

    Parameters:
        segment_results: Output from discover_segments()
        feature_names_to_show: Specific features to compare (default: top N)
        top_n: Number of features if not specified
        height: Plot height

    Returns:
        Plotly figure
    """
    import plotly.graph_objects as go

    importance_df = segment_results['importance_by_segment']

    # Select features to show
    if feature_names_to_show is None:
        avg_importance = importance_df.mean(axis=1).sort_values(ascending=False)
        feature_names_to_show = avg_importance.head(top_n).index.tolist()

    # Filter to selected features
    plot_df = importance_df.loc[feature_names_to_show]

    fig = go.Figure()

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']

    for i, col in enumerate(plot_df.columns):
        fig.add_trace(go.Bar(
            name=col,
            y=plot_df.index,
            x=plot_df[col],
            orientation='h',
            marker_color=colors[i % len(colors)]
        ))

    fig.update_layout(
        title=dict(text='Feature Importance Comparison Across Segments', x=0.5, xanchor='center'),
        xaxis_title='Mean |SHAP|',
        yaxis_title='',
        barmode='group',
        height=height,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=150, r=50, t=100, b=50)
    )

    return fig


def get_segment_summary(segment_results: Dict) -> str:
    """
    Generate text summary of discovered segments.

    Parameters:
        segment_results: Output from discover_segments()

    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("=" * 70)
    lines.append("DISCOVERED SEGMENTS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"\nMethod: {segment_results['method']}")
    lines.append(f"Number of segments: {segment_results['n_segments']}")
    lines.append(f"Features used for clustering: {', '.join(segment_results['features_used'][:5])}...")

    for seg in segment_results['segments']:
        lines.append("\n" + "-" * 70)
        lines.append(f"SEGMENT {seg['segment_id']}: {seg['description']}")
        lines.append("-" * 70)
        lines.append(f"Size: {seg['size']} observations ({seg['pct_of_total']:.1f}% of total)")

        if 'target_rate' in seg:
            lines.append(f"Target rate: {seg['target_rate']:.1%}")

        lines.append("\nTop Drivers:")
        for driver in seg['top_drivers'][:5]:
            direction = "+" if driver['direction'] == 'positive' else "-"
            lines.append(f"  {driver['feature']:30s} {direction}{driver['importance']:.4f}")

    return "\n".join(lines)
