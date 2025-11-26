"""
Threshold Detection: Find significant breakpoints in feature-SHAP relationships.

Identifies values where a feature's effect on predictions changes significantly.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from scipy import stats


def detect_thresholds(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    feature: str,
    method: str = 'changepoint',
    min_segment_size: int = 50,
    significance_level: float = 0.05,
    max_thresholds: int = 3
) -> Dict:
    """
    Detect significant thresholds where feature effect changes.

    Parameters:
        shap_values: SHAP values array (n_samples, n_features)
        X: Feature matrix
        feature_names: List of feature names
        feature: Feature name to analyze
        method: 'changepoint' (statistical) or 'gradient' (slope-based)
        min_segment_size: Minimum samples on each side of threshold
        significance_level: p-value threshold for significance
        max_thresholds: Maximum number of thresholds to return

    Returns:
        Dictionary with threshold analysis results
    """
    if feature not in feature_names:
        raise ValueError(f"Feature '{feature}' not found")

    feat_idx = feature_names.index(feature)
    feat_values = X[feature].values
    shap_vals = shap_values[:, feat_idx]

    # Sort by feature value
    sort_idx = np.argsort(feat_values)
    feat_sorted = feat_values[sort_idx]
    shap_sorted = shap_vals[sort_idx]

    # Get unique values for potential thresholds
    unique_values = np.unique(feat_sorted)

    if len(unique_values) < 3:
        return {
            'feature': feature,
            'thresholds': [],
            'message': 'Not enough unique values for threshold detection'
        }

    # Find thresholds based on method
    if method == 'changepoint':
        thresholds = _detect_changepoints(
            feat_sorted, shap_sorted, unique_values,
            min_segment_size, significance_level, max_thresholds
        )
    elif method == 'gradient':
        thresholds = _detect_gradient_changes(
            feat_sorted, shap_sorted, unique_values,
            min_segment_size, max_thresholds
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'changepoint' or 'gradient'")

    # Generate interpretations
    for t in thresholds:
        direction = "increases" if t['shap_after'] > t['shap_before'] else "decreases"
        magnitude = abs(t['effect_change'])
        if magnitude > 0.2:
            strength = "dramatically"
        elif magnitude > 0.1:
            strength = "significantly"
        else:
            strength = "moderately"

        t['interpretation'] = (
            f"Effect {strength} {direction} at {feature}={t['value']:.2f}: "
            f"from {t['shap_before']:.3f} to {t['shap_after']:.3f}"
        )

    return {
        'feature': feature,
        'method': method,
        'n_samples': len(feat_values),
        'thresholds': thresholds
    }


def _detect_changepoints(
    feat_sorted: np.ndarray,
    shap_sorted: np.ndarray,
    unique_values: np.ndarray,
    min_segment_size: int,
    significance_level: float,
    max_thresholds: int
) -> List[Dict]:
    """Detect changepoints using statistical tests."""

    candidates = []
    n = len(feat_sorted)

    # Test each unique value as potential threshold
    for val in unique_values:
        left_mask = feat_sorted < val
        right_mask = feat_sorted >= val

        n_left = left_mask.sum()
        n_right = right_mask.sum()

        # Skip if segments too small
        if n_left < min_segment_size or n_right < min_segment_size:
            continue

        shap_left = shap_sorted[left_mask]
        shap_right = shap_sorted[right_mask]

        # T-test for difference in means
        t_stat, p_value = stats.ttest_ind(shap_left, shap_right)

        if p_value < significance_level:
            mean_left = shap_left.mean()
            mean_right = shap_right.mean()
            effect_change = mean_right - mean_left

            candidates.append({
                'value': float(val),
                'shap_before': float(mean_left),
                'shap_after': float(mean_right),
                'effect_change': float(effect_change),
                'p_value': float(p_value),
                'confidence': float(1 - p_value),
                'n_below': int(n_left),
                'n_above': int(n_right)
            })

    # Sort by absolute effect change and take top N
    candidates.sort(key=lambda x: abs(x['effect_change']), reverse=True)

    # Remove overlapping thresholds (keep most significant)
    final_thresholds = []
    for c in candidates:
        # Check if too close to existing threshold
        too_close = False
        for existing in final_thresholds:
            # Consider "too close" if within 10% of range
            range_val = unique_values.max() - unique_values.min()
            if abs(c['value'] - existing['value']) < range_val * 0.1:
                too_close = True
                break

        if not too_close:
            final_thresholds.append(c)

        if len(final_thresholds) >= max_thresholds:
            break

    return final_thresholds


def _detect_gradient_changes(
    feat_sorted: np.ndarray,
    shap_sorted: np.ndarray,
    unique_values: np.ndarray,
    min_segment_size: int,
    max_thresholds: int
) -> List[Dict]:
    """Detect thresholds based on gradient/slope changes."""

    # Compute rolling mean SHAP for smoothing
    window = max(min_segment_size // 2, 10)

    # Group by unique values and compute mean SHAP
    value_means = []
    for val in unique_values:
        mask = feat_sorted == val
        if mask.sum() > 0:
            value_means.append({
                'value': val,
                'mean_shap': shap_sorted[mask].mean(),
                'count': mask.sum()
            })

    if len(value_means) < 3:
        return []

    # Compute gradients between consecutive unique values
    gradients = []
    for i in range(1, len(value_means)):
        delta_x = value_means[i]['value'] - value_means[i-1]['value']
        delta_y = value_means[i]['mean_shap'] - value_means[i-1]['mean_shap']

        if delta_x > 0:
            gradient = delta_y / delta_x
            gradients.append({
                'value': value_means[i]['value'],
                'gradient': gradient,
                'shap_before': value_means[i-1]['mean_shap'],
                'shap_after': value_means[i]['mean_shap']
            })

    if len(gradients) < 2:
        return []

    # Find points where gradient changes significantly
    candidates = []
    gradient_values = [g['gradient'] for g in gradients]
    gradient_std = np.std(gradient_values)

    for i in range(1, len(gradients)):
        gradient_change = abs(gradients[i]['gradient'] - gradients[i-1]['gradient'])

        # Significant if change > 1.5 std deviations
        if gradient_change > 1.5 * gradient_std and gradient_std > 0:
            val = gradients[i]['value']

            # Get samples before and after
            left_mask = feat_sorted < val
            right_mask = feat_sorted >= val

            n_left = left_mask.sum()
            n_right = right_mask.sum()

            if n_left >= min_segment_size and n_right >= min_segment_size:
                mean_left = shap_sorted[left_mask].mean()
                mean_right = shap_sorted[right_mask].mean()

                candidates.append({
                    'value': float(val),
                    'shap_before': float(mean_left),
                    'shap_after': float(mean_right),
                    'effect_change': float(mean_right - mean_left),
                    'gradient_change': float(gradient_change),
                    'confidence': float(min(gradient_change / (3 * gradient_std), 0.99)),
                    'n_below': int(n_left),
                    'n_above': int(n_right)
                })

    # Sort by effect change magnitude
    candidates.sort(key=lambda x: abs(x['effect_change']), reverse=True)

    return candidates[:max_thresholds]


def detect_all_thresholds(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    top_n_features: int = 10,
    **kwargs
) -> pd.DataFrame:
    """
    Detect thresholds for top N most important features.

    Parameters:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: List of feature names
        top_n_features: Number of top features to analyze
        **kwargs: Arguments passed to detect_thresholds()

    Returns:
        DataFrame with all detected thresholds
    """
    # Get top features by importance
    importance = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(importance)[::-1][:top_n_features]
    top_features = [feature_names[i] for i in top_idx]

    all_thresholds = []

    for feature in top_features:
        result = detect_thresholds(
            shap_values, X, feature_names, feature, **kwargs
        )

        for t in result.get('thresholds', []):
            t['feature'] = feature
            all_thresholds.append(t)

    if not all_thresholds:
        return pd.DataFrame(columns=['feature', 'value', 'effect_change', 'interpretation'])

    df = pd.DataFrame(all_thresholds)
    df = df.sort_values('effect_change', key=abs, ascending=False)

    return df
