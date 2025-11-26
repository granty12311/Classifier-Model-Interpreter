"""
Simplified Interpreter class - primary API for model interpretation.

Provides core visualization capabilities:
- Global feature importance
- Beeswarm plots
- Dependence plots
- Prediction surface visualizations (2D contour and 3D surface)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
import warnings

from .config import Config, PRESETS
from ..explainers.shap_explainer import compute_shap_values, get_base_value
from ..utils.data_utils import validate_data
from ..utils.model_utils import check_model_fitted
from ..visualization.global_plots import plot_global_importance
from ..visualization.performance_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    calculate_metrics,
    plot_metrics_summary
)
from ..visualization.dependence_plots import (
    plot_beeswarm,
    plot_dependence,
    plot_dependence_categorical
)
from ..visualization.interaction_plots import (
    plot_interaction_contour,
    plot_interaction_surface_3d
)


class Interpreter:
    """
    Simple model interpretation interface using SHAP values.

    Core visualizations:
    - plot_global_importance(): Bar chart of feature importance
    - plot_beeswarm(): Distribution of SHAP values per feature
    - plot_dependence(): How feature values affect predictions
    - plot_dependence_categorical(): Dependence for categorical features
    - plot_prediction_surface(): 2D heatmap of predictions across two features
    - plot_prediction_surface_3d(): 3D surface of predictions

    Examples:
        >>> from src.core import Interpreter
        >>> interp = Interpreter(model, X_test, y_test)
        >>> interp.plot_global_importance()
        >>> interp.plot_beeswarm()
        >>> interp.plot_dependence('feature_name')
        >>> interp.plot_prediction_surface('feature_1', 'feature_2')
    """

    def __init__(
        self,
        model,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        sample_size: int = 2000,
        compute_shap_now: bool = True
    ):
        """
        Initialize interpreter.

        Parameters:
            model: Trained classifier model (must have predict_proba method)
            X: Feature matrix (pandas DataFrame)
            y: Target variable (optional, used for performance metrics)
            sample_size: Max samples for SHAP computation (default: 2000)
            compute_shap_now: Compute SHAP values immediately (default: True)
        """
        self.model = model
        self.X = X
        self.y = y
        self.sample_size = sample_size

        # Validate model
        if not check_model_fitted(model):
            warnings.warn("Model may not be fitted. Results may be unreliable.", UserWarning)

        # Storage for computed values
        self.shap_values = None
        self.explainer = None
        self.base_value = None
        self.feature_names = list(X.columns)

        # X_shap: The subset of X used for SHAP computation (may be sampled)
        self.X_shap = None
        self.y_shap = None

        # Compute SHAP if requested
        if compute_shap_now:
            self.compute_shap()

    def compute_shap(self, force_recompute: bool = False):
        """
        Compute SHAP values.

        Parameters:
            force_recompute: Recompute even if already computed
        """
        if self.shap_values is not None and not force_recompute:
            warnings.warn("SHAP values already computed. Use force_recompute=True to recompute.", UserWarning)
            return

        # Compute SHAP values and get the sampled X used
        self.shap_values, self.explainer, self.X_shap, sample_idx = compute_shap_values(
            self.model,
            self.X,
            algorithm='auto',
            sample_size=self.sample_size,
            random_state=42,
            return_sample=True
        )

        # Also sample y to match X_shap (using the same indices)
        if self.y is not None and len(self.X_shap) < len(self.X):
            self.y_shap = self.y[sample_idx]
        else:
            self.y_shap = self.y

        self.base_value = get_base_value(self.explainer)

    def get_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.

        Parameters:
            top_n: Return top N features

        Returns:
            DataFrame with feature names and importance values
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        importance = np.abs(self.shap_values).mean(axis=0)

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        if top_n:
            df = df.head(top_n)

        return df

    def plot_global_importance(self, top_n: int = 10, **kwargs):
        """
        Plot global feature importance bar chart.

        Parameters:
            top_n: Number of top features to show
            **kwargs: Additional arguments (title, height)

        Returns:
            Plotly figure
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        return plot_global_importance(
            self.shap_values,
            self.feature_names,
            top_n=top_n,
            **kwargs
        )

    def plot_beeswarm(self, top_n: int = 10, max_display: int = 1000, **kwargs):
        """
        Create beeswarm plot showing SHAP value distributions.

        Each dot represents one sample. X-axis shows SHAP value (impact on prediction),
        color shows feature value (red=high, blue=low).

        Parameters:
            top_n: Number of top features to show
            max_display: Maximum samples to display
            **kwargs: Additional arguments (title, height)

        Returns:
            Plotly figure
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        return plot_beeswarm(
            self.shap_values,
            self.X_shap,
            self.feature_names,
            top_n=top_n,
            max_display=max_display,
            **kwargs
        )

    def plot_dependence(self, feature_name: str, color_feature: Optional[str] = None, **kwargs):
        """
        Create dependence plot for a feature.

        Shows how a feature's value affects its SHAP value (prediction impact).
        - X-axis: Feature value
        - Y-axis: SHAP value (impact on prediction)
        - Color: Values of another feature (auto-detected or specified)

        Parameters:
            feature_name: Feature to plot
            color_feature: Feature to use for color coding (auto-detected if None)
            **kwargs: Additional arguments (title, height)

        Returns:
            Plotly figure
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        return plot_dependence(
            self.shap_values,
            self.X_shap,
            feature_name,
            self.feature_names,
            interaction_feature=color_feature,
            **kwargs
        )

    def plot_dependence_categorical(self, feature_name: str, value_labels: Optional[dict] = None, **kwargs):
        """
        Create dependence plot for categorical (label-encoded) feature.

        Shows box plots of SHAP values for each category.

        Parameters:
            feature_name: Categorical feature name
            value_labels: Dict mapping encoded values to labels (e.g., {0: 'Low', 1: 'High'})
            **kwargs: Additional arguments (title, height)

        Returns:
            Plotly figure
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        return plot_dependence_categorical(
            self.shap_values,
            self.X_shap,
            feature_name,
            self.feature_names,
            value_labels=value_labels,
            **kwargs
        )

    def plot_prediction_surface(
        self,
        feature_1: str,
        feature_2: str,
        n_grid: int = 50,
        value_labels_1: Optional[dict] = None,
        value_labels_2: Optional[dict] = None,
        **kwargs
    ):
        """
        Create 2D heatmap showing predicted probability across two features.

        Shows how combinations of two features affect the model's predicted probability.
        Other features are held at their median values.

        Parameters:
            feature_1: First feature name (X-axis)
            feature_2: Second feature name (Y-axis)
            n_grid: Grid resolution for continuous features
            value_labels_1: Dict mapping feature_1 values to labels
            value_labels_2: Dict mapping feature_2 values to labels
            **kwargs: Additional arguments (title, height)

        Returns:
            Plotly figure
        """
        return plot_interaction_contour(
            self.model,
            self.X_shap if self.X_shap is not None else self.X,
            feature_1,
            feature_2,
            n_grid=n_grid,
            other_features_percentile=0.5,
            value_labels_1=value_labels_1,
            value_labels_2=value_labels_2,
            **kwargs
        )

    def plot_prediction_surface_3d(
        self,
        feature_1: str,
        feature_2: str,
        n_grid: int = 30,
        value_labels_1: Optional[dict] = None,
        value_labels_2: Optional[dict] = None,
        **kwargs
    ):
        """
        Create 3D surface plot showing predicted probability across two features.

        Z-axis represents predicted probability, making peaks and valleys visible.
        Other features are held at their median values.

        Parameters:
            feature_1: First feature name (X-axis)
            feature_2: Second feature name (Y-axis)
            n_grid: Grid resolution for continuous features
            value_labels_1: Dict mapping feature_1 values to labels
            value_labels_2: Dict mapping feature_2 values to labels
            **kwargs: Additional arguments (title, height)

        Returns:
            Plotly figure
        """
        return plot_interaction_surface_3d(
            self.model,
            self.X_shap if self.X_shap is not None else self.X,
            feature_1,
            feature_2,
            n_grid=n_grid,
            other_features_percentile=0.5,
            value_labels_1=value_labels_1,
            value_labels_2=value_labels_2,
            **kwargs
        )

    def plot_performance(self):
        """
        Plot model performance metrics.

        Returns:
            Dictionary of plotly figures:
            - 'metrics_summary': Key metrics (accuracy, precision, recall, F1, AUC)
            - 'confusion_matrix': Confusion matrix heatmap
            - 'roc_curve': ROC curve (if binary classification)
        """
        if self.y is None:
            raise RuntimeError("Target variable y not provided. Cannot compute performance metrics.")

        y_pred = self.model.predict(self.X)
        y_proba = None

        if hasattr(self.model, 'predict_proba'):
            y_proba_full = self.model.predict_proba(self.X)
            if y_proba_full.shape[1] == 2:
                y_proba = y_proba_full[:, 1]

        metrics = calculate_metrics(self.y, y_pred, y_proba)

        plots = {
            'metrics_summary': plot_metrics_summary(metrics),
            'confusion_matrix': plot_confusion_matrix(self.y, y_pred)
        }

        if y_proba is not None:
            plots['roc_curve'] = plot_roc_curve(self.y, y_proba)

        return plots

    def summary(self) -> Dict:
        """
        Generate summary of key interpretation insights.

        Returns:
            Dictionary with model info, top features, and performance metrics
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        summary = {
            'model_type': self.model.__class__.__name__,
            'n_samples': len(self.X),
            'n_samples_shap': len(self.X_shap),
            'n_features': len(self.feature_names),
            'base_value': self.base_value,
            'top_features': self.get_importance(top_n=10).to_dict('records')
        }

        if self.y is not None:
            y_pred = self.model.predict(self.X)
            y_proba = None
            if hasattr(self.model, 'predict_proba'):
                y_proba_full = self.model.predict_proba(self.X)
                if y_proba_full.shape[1] == 2:
                    y_proba = y_proba_full[:, 1]
            summary['performance'] = calculate_metrics(self.y, y_pred, y_proba)

        return summary
