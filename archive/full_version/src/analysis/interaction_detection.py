"""
Interaction detection: identify feature pairs with strong interaction effects.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from scipy.stats import spearmanr
import warnings


def detect_interactions(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 10,
    method: str = 'shap_variance'
) -> pd.DataFrame:
    """
    Detect feature interactions using SHAP values.

    Parameters:
        shap_values: SHAP values array (n_samples, n_features)
        X: Feature matrix
        feature_names: Feature names
        top_n: Number of top interactions to return
        method: 'shap_variance' or 'correlation'
            - shap_variance: Measures how much feature B affects feature A's SHAP values
            - correlation: Correlation between SHAP values of feature pairs

    Returns:
        DataFrame with interaction strength for feature pairs
    """
    n_features = len(feature_names)
    interactions = []

    if method == 'shap_variance':
        # For each feature pair (A, B), measure how much B's value affects A's SHAP values
        for i in range(n_features):
            for j in range(i + 1, n_features):
                feat_a = feature_names[i]
                feat_b = feature_names[j]

                shap_a = shap_values[:, i]
                feat_b_vals = X.iloc[:, j].values

                # Split by feature B value (median split for numeric, by value for categorical)
                if X.iloc[:, j].nunique() <= 10:
                    # Categorical or low-cardinality
                    unique_vals = X.iloc[:, j].unique()
                    variances = []

                    for val in unique_vals:
                        mask = feat_b_vals == val
                        if mask.sum() > 10:  # Enough samples
                            var = np.var(shap_a[mask])
                            variances.append(var)

                    if len(variances) > 1:
                        # Interaction strength = difference in variance across groups
                        interaction_strength = np.std(variances)
                    else:
                        interaction_strength = 0.0

                else:
                    # Numeric feature - split by median
                    median = np.median(feat_b_vals)
                    mask_low = feat_b_vals <= median
                    mask_high = feat_b_vals > median

                    var_low = np.var(shap_a[mask_low])
                    var_high = np.var(shap_a[mask_high])

                    # Interaction strength = absolute difference in variance
                    interaction_strength = abs(var_high - var_low)

                interactions.append({
                    'feature_1': feat_a,
                    'feature_2': feat_b,
                    'interaction_strength': interaction_strength
                })

    elif method == 'correlation':
        # Correlation between SHAP values
        for i in range(n_features):
            for j in range(i + 1, n_features):
                feat_a = feature_names[i]
                feat_b = feature_names[j]

                shap_a = shap_values[:, i]
                shap_b = shap_values[:, j]

                # Use Spearman correlation (robust to outliers)
                corr, _ = spearmanr(shap_a, shap_b)

                interactions.append({
                    'feature_1': feat_a,
                    'feature_2': feat_b,
                    'interaction_strength': abs(corr)
                })

    # Create DataFrame and sort
    df = pd.DataFrame(interactions)
    df = df.sort_values('interaction_strength', ascending=False).head(top_n)

    return df


def analyze_interaction(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    feature_1: str,
    feature_2: str
) -> Dict:
    """
    Analyze specific interaction between two features.

    Parameters:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: Feature names
        feature_1: First feature name
        feature_2: Second feature name

    Returns:
        Dictionary with interaction analysis
    """
    if feature_1 not in feature_names or feature_2 not in feature_names:
        raise ValueError(f"Features not found in feature_names")

    idx_1 = feature_names.index(feature_1)
    idx_2 = feature_names.index(feature_2)

    shap_1 = shap_values[:, idx_1]
    shap_2 = shap_values[:, idx_2]
    vals_1 = X[feature_1].values
    vals_2 = X[feature_2].values

    # Compute interaction metrics
    corr_shap, _ = spearmanr(shap_1, shap_2)
    corr_values, _ = spearmanr(vals_1, vals_2)

    # Conditional analysis
    # Split feature 2 into groups and analyze feature 1's SHAP in each group
    if X[feature_2].nunique() <= 10:
        # Categorical
        groups = X[feature_2].unique()
        conditional_means = {}

        for group in groups:
            mask = vals_2 == group
            if mask.sum() > 5:
                conditional_means[str(group)] = {
                    'mean_shap': float(shap_1[mask].mean()),
                    'std_shap': float(shap_1[mask].std()),
                    'count': int(mask.sum())
                }
    else:
        # Numeric - split by quartiles
        quartiles = np.percentile(vals_2, [25, 50, 75])
        groups = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
        conditional_means = {}

        masks = [
            vals_2 <= quartiles[0],
            (vals_2 > quartiles[0]) & (vals_2 <= quartiles[1]),
            (vals_2 > quartiles[1]) & (vals_2 <= quartiles[2]),
            vals_2 > quartiles[2]
        ]

        for group, mask in zip(groups, masks):
            if mask.sum() > 5:
                conditional_means[group] = {
                    'mean_shap': float(shap_1[mask].mean()),
                    'std_shap': float(shap_1[mask].std()),
                    'count': int(mask.sum())
                }

    return {
        'feature_1': feature_1,
        'feature_2': feature_2,
        'correlation_shap_values': float(corr_shap),
        'correlation_feature_values': float(corr_values),
        'conditional_analysis': conditional_means
    }


def compute_interaction_matrix(
    shap_values: np.ndarray,
    feature_names: List[str],
    method: str = 'correlation'
) -> pd.DataFrame:
    """
    Compute pairwise interaction matrix for all features.

    Parameters:
        shap_values: SHAP values array
        feature_names: Feature names
        method: 'correlation' or 'mutual_info'

    Returns:
        DataFrame with interaction matrix (symmetric)
    """
    n_features = len(feature_names)
    matrix = np.zeros((n_features, n_features))

    if method == 'correlation':
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    corr, _ = spearmanr(shap_values[:, i], shap_values[:, j])
                    matrix[i, j] = abs(corr)

    df = pd.DataFrame(matrix, index=feature_names, columns=feature_names)
    return df
