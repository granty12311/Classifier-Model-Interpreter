"""
Data processing utilities: sampling, validation, preprocessing.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional


def sample_data(
    X: pd.DataFrame,
    y: Optional[np.ndarray] = None,
    sample_size: int = 2000,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, Optional[np.ndarray], np.ndarray]:
    """
    Sample data if larger than sample_size.

    Parameters:
        X: Feature matrix
        y: Target variable (optional)
        sample_size: Maximum number of samples
        random_state: Random seed
        stratify: Stratify sampling by target (if y provided)

    Returns:
        X_sampled, y_sampled (if y provided), sample_indices
    """
    n_samples = len(X)

    if n_samples <= sample_size:
        # No sampling needed
        return X, y, np.arange(n_samples)

    # Sample data
    if y is not None and stratify:
        # Stratified sampling
        from sklearn.model_selection import train_test_split
        _, X_sampled, _, y_sampled, _, sample_idx = train_test_split(
            X, y, np.arange(n_samples),
            train_size=n_samples - sample_size,
            test_size=sample_size,
            stratify=y,
            random_state=random_state
        )
    else:
        # Random sampling
        np.random.seed(random_state)
        sample_idx = np.random.choice(n_samples, size=sample_size, replace=False)
        X_sampled = X.iloc[sample_idx]
        y_sampled = y[sample_idx] if y is not None else None

    warnings.warn(
        f"Dataset has {n_samples:,} samples. Using random sample of {sample_size:,} for SHAP computation. "
        f"Results represent overall patterns but may miss rare cases.",
        UserWarning
    )

    return X_sampled, y_sampled, sample_idx


def validate_data(
    X: pd.DataFrame,
    y: Optional[np.ndarray] = None,
    warn: bool = True
) -> dict:
    """
    Validate data quality and issue warnings (not errors).

    Parameters:
        X: Feature matrix
        y: Target variable (optional)
        warn: Whether to issue warnings

    Returns:
        Dictionary of validation results
    """
    issues = {
        'missing_values': {},
        'constant_features': [],
        'suspicious_values': {},
        'warnings': []
    }

    # Check for missing values
    missing = X.isnull().sum()
    for col, count in missing[missing > 0].items():
        pct = count / len(X) * 100
        issues['missing_values'][col] = pct
        if pct > 50 and warn:
            warnings.warn(f"Feature '{col}' has {pct:.1f}% missing values", UserWarning)

    # Check for constant features
    for col in X.columns:
        if X[col].nunique() == 1:
            issues['constant_features'].append(col)
            if warn:
                warnings.warn(f"Feature '{col}' has only one unique value", UserWarning)

    # Check for suspicious values in numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Check for extreme outliers (beyond 5 std)
        if X[col].std() > 0:  # Avoid division by zero
            z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
            n_outliers = (z_scores > 5).sum()
            if n_outliers > 0:
                issues['suspicious_values'][col] = int(n_outliers)
                if n_outliers > len(X) * 0.01 and warn:  # > 1% outliers
                    warnings.warn(
                        f"Feature '{col}' has {n_outliers} extreme outliers (>5 std)",
                        UserWarning
                    )

    # Check target if provided
    if y is not None:
        unique_classes = np.unique(y)
        class_counts = pd.Series(y).value_counts()

        if len(unique_classes) > 10 and warn:
            warnings.warn(
                f"Target has {len(unique_classes)} classes. "
                f"This tool is designed for binary/multiclass classification.",
                UserWarning
            )

        # Check for class imbalance
        min_class_pct = class_counts.min() / len(y) * 100
        if min_class_pct < 5 and warn:
            warnings.warn(
                f"Severe class imbalance detected. "
                f"Smallest class represents only {min_class_pct:.1f}% of data.",
                UserWarning
            )

    return issues


def detect_categorical_features(
    X: pd.DataFrame,
    max_unique_ratio: float = 0.05
) -> list:
    """
    Auto-detect categorical features.

    Parameters:
        X: Feature matrix
        max_unique_ratio: Max ratio of unique values to consider categorical

    Returns:
        List of categorical column names
    """
    categorical = []

    # By dtype
    categorical.extend(X.select_dtypes(include=['object', 'category']).columns.tolist())

    # By cardinality (numeric with few unique values)
    for col in X.select_dtypes(include=[np.number]).columns:
        unique_ratio = X[col].nunique() / len(X)
        if unique_ratio < max_unique_ratio:
            categorical.append(col)

    return list(set(categorical))
