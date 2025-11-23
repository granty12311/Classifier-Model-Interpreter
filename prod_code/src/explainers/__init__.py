"""
Explainer modules for SHAP computation and categorical handling.
"""

from .shap_explainer import compute_shap_values
from .categorical_handler import aggregate_categorical_shap, detect_categorical_groups

__all__ = ['compute_shap_values', 'aggregate_categorical_shap', 'detect_categorical_groups']
