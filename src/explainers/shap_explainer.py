"""
SHAP value computation with automatic algorithm selection and sampling.
"""

import numpy as np
import pandas as pd
import shap
import warnings
from typing import Tuple, Optional

from ..utils.model_utils import get_model_type, wrap_model


def compute_shap_values(
    model,
    X: pd.DataFrame,
    algorithm: str = 'auto',
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, shap.Explainer]:
    """
    Compute SHAP values using appropriate algorithm.

    Parameters:
        model: Trained model
        X: Feature matrix
        algorithm: 'auto', 'tree', 'kernel', 'linear'
        sample_size: Max samples (None = use all)
        random_state: Random seed

    Returns:
        shap_values (n_samples, n_features), explainer object
    """
    # Sample if needed
    if sample_size and len(X) > sample_size:
        from ..utils.data_utils import sample_data
        X_sample, _, _ = sample_data(X, None, sample_size, random_state, stratify=False)
    else:
        X_sample = X

    # Auto-detect algorithm
    if algorithm == 'auto':
        model_type = get_model_type(model)
        if model_type == 'tree':
            algorithm = 'tree'
        elif model_type == 'linear':
            algorithm = 'linear'
        else:
            algorithm = 'kernel'
            warnings.warn(
                f"Model type not recognized. Using KernelExplainer (slower). "
                f"For faster computation, use tree-based or linear models.",
                UserWarning
            )

    # Create explainer
    try:
        if algorithm == 'tree':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # Handle multi-class output (use positive class for binary)
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values = shap_values[1]  # Positive class
                else:
                    warnings.warn(
                        f"Multi-class model detected ({len(shap_values)} classes). "
                        f"Using class 1 SHAP values. For full analysis, analyze each class separately.",
                        UserWarning
                    )
                    shap_values = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # Handle 3D array (newer SHAP versions)
                if shap_values.shape[2] == 2:
                    shap_values = shap_values[:, :, 1]  # Positive class
                else:
                    warnings.warn(
                        f"Multi-class model detected ({shap_values.shape[2]} classes). "
                        f"Using class 1 SHAP values. For full analysis, analyze each class separately.",
                        UserWarning
                    )
                    shap_values = shap_values[:, :, 1]

        elif algorithm == 'linear':
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)

            # Handle multi-output
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]

        else:  # kernel
            # Use a background sample for kernel explainer
            background_size = min(100, len(X_sample))
            background = shap.sample(X_sample, background_size, random_state=random_state)

            predict_fn = wrap_model(model)
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_sample, nsamples=100)

            # Handle multi-output
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]

    except Exception as e:
        raise RuntimeError(
            f"SHAP computation failed with {algorithm} algorithm. "
            f"Error: {str(e)}. Try a different algorithm or check model compatibility."
        )

    # Ensure 2D array
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    return shap_values, explainer


def get_base_value(explainer: shap.Explainer) -> float:
    """
    Extract base value from SHAP explainer.

    Parameters:
        explainer: SHAP explainer object

    Returns:
        Base value (average model output)
    """
    if hasattr(explainer, 'expected_value'):
        base_value = explainer.expected_value

        # Handle list (multi-class)
        if isinstance(base_value, (list, np.ndarray)):
            if len(base_value) == 2:
                return float(base_value[1])  # Positive class
            return float(base_value[0])

        return float(base_value)

    return 0.0
