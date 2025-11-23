"""
Categorical variable handling: detection, grouping, aggregation of SHAP values.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import re


def detect_categorical_groups(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Auto-detect one-hot encoded categorical groups.

    Detection patterns:
    - prefix_value1, prefix_value2 → prefix
    - is_value1, is_value2 → original categorical

    Parameters:
        feature_names: List of feature names

    Returns:
        Dictionary mapping category name to list of dummy variables
    """
    groups = {}

    # Pattern 1: prefix_suffix (e.g., color_red, color_blue)
    prefix_groups = {}
    for name in feature_names:
        if '_' in name:
            parts = name.rsplit('_', 1)
            if len(parts) == 2:
                prefix, suffix = parts
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(name)

    # Keep only groups with multiple members
    for prefix, members in prefix_groups.items():
        if len(members) > 1:
            groups[prefix] = members

    return groups


def aggregate_categorical_shap(
    shap_values: np.ndarray,
    feature_names: List[str],
    categorical_features: Optional[List[str]] = None,
    auto_detect: bool = True,
    aggregation: str = 'sum'
) -> Tuple[np.ndarray, List[str], Dict[str, Dict]]:
    """
    Aggregate SHAP values for categorical features.

    Parameters:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: Original feature names
        categorical_features: User-specified categorical features
        auto_detect: Auto-detect one-hot encoded groups
        aggregation: 'sum', 'mean', or 'max'

    Returns:
        aggregated_shap: SHAP values with categoricals combined
        aggregated_names: Feature names with categoricals grouped
        category_breakdown: Dict with breakdown info per category
    """
    # Detect categorical groups
    categorical_groups = {}

    if auto_detect:
        categorical_groups = detect_categorical_groups(feature_names)

    # Add user-specified groups (if provided as dict)
    if categorical_features:
        if isinstance(categorical_features, dict):
            categorical_groups.update(categorical_features)
        # If list, check if they're already in detected groups
        # (otherwise they're non-encoded categoricals, keep as-is)

    if not categorical_groups:
        # No categoricals to aggregate
        return shap_values, feature_names, {}

    # Build aggregated arrays
    aggregated_shap = []
    aggregated_names = []
    category_breakdown = {}

    # Track which features are already grouped
    grouped_features = set()
    for group_name, members in categorical_groups.items():
        grouped_features.update(members)

    # Process features
    for i, name in enumerate(feature_names):
        if name in grouped_features:
            # Skip, will be handled as part of group
            continue

        # Check if this is a group prefix
        if name in categorical_groups:
            # Aggregate group
            member_indices = [feature_names.index(m) for m in categorical_groups[name]]
            group_shap = shap_values[:, member_indices]

            if aggregation == 'sum':
                agg_shap = group_shap.sum(axis=1)
            elif aggregation == 'mean':
                agg_shap = group_shap.mean(axis=1)
            elif aggregation == 'max':
                agg_shap = group_shap.max(axis=1)
            else:
                agg_shap = group_shap.sum(axis=1)

            aggregated_shap.append(agg_shap)
            aggregated_names.append(name)

            # Store breakdown
            category_breakdown[name] = {
                'members': categorical_groups[name],
                'member_importance': {
                    member: float(np.abs(shap_values[:, feature_names.index(member)]).mean())
                    for member in categorical_groups[name]
                },
                'total_importance': float(np.abs(agg_shap).mean())
            }
        else:
            # Regular feature
            aggregated_shap.append(shap_values[:, i])
            aggregated_names.append(name)

    # Check for groups not already added
    for group_name, members in categorical_groups.items():
        if group_name not in aggregated_names:
            # Aggregate this group
            member_indices = [feature_names.index(m) for m in members]
            group_shap = shap_values[:, member_indices]

            if aggregation == 'sum':
                agg_shap = group_shap.sum(axis=1)
            elif aggregation == 'mean':
                agg_shap = group_shap.mean(axis=1)
            else:
                agg_shap = group_shap.sum(axis=1)

            aggregated_shap.append(agg_shap)
            aggregated_names.append(group_name)

            # Store breakdown
            category_breakdown[group_name] = {
                'members': members,
                'member_importance': {
                    member: float(np.abs(shap_values[:, feature_names.index(member)]).mean())
                    for member in members
                },
                'total_importance': float(np.abs(agg_shap).mean())
            }

    # Convert to array
    aggregated_shap = np.column_stack(aggregated_shap)

    return aggregated_shap, aggregated_names, category_breakdown


def get_category_breakdown(
    category_name: str,
    category_breakdown: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Get breakdown of categorical feature contributions.

    Parameters:
        category_name: Name of categorical feature
        category_breakdown: Breakdown dict from aggregate_categorical_shap

    Returns:
        DataFrame with member contributions
    """
    if category_name not in category_breakdown:
        return pd.DataFrame()

    info = category_breakdown[category_name]

    df = pd.DataFrame([
        {'category_value': member.replace(f'{category_name}_', ''),
         'importance': importance}
        for member, importance in info['member_importance'].items()
    ])

    df = df.sort_values('importance', ascending=False)

    # Calculate percentage as share of sum of member importances (ensures sum = 100%)
    total_member_importance = df['importance'].sum()
    if total_member_importance > 0:
        df['pct_of_total'] = df['importance'] / total_member_importance * 100
    else:
        df['pct_of_total'] = 0.0

    return df
