"""
Main Interpreter class - primary API for model interpretation.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
import warnings

from .config import Config, PRESETS
from ..explainers.shap_explainer import compute_shap_values, get_base_value
from ..explainers.categorical_handler import (
    aggregate_categorical_shap,
    detect_categorical_groups,
    get_category_breakdown
)
from ..utils.data_utils import sample_data, validate_data
from ..utils.model_utils import check_model_fitted
from ..visualization.global_plots import (
    plot_global_importance,
    plot_feature_distribution,
    plot_importance_comparison,
    plot_categorical_breakdown
)
from ..visualization.performance_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    calculate_metrics,
    plot_metrics_summary,
    plot_calibration_curve
)
from ..visualization.dependence_plots import (
    plot_beeswarm,
    plot_dependence,
    plot_dependence_categorical
)
from ..visualization.interaction_plots import (
    plot_interaction_scatter,
    plot_interaction_heatmap,
    plot_interaction_matrix,
    plot_interaction_contour,
    plot_interaction_surface_3d,
    plot_conditional_dependence
)
from ..analysis.interaction_detection import (
    detect_interactions,
    analyze_interaction,
    compute_interaction_matrix
)


class Interpreter:
    """
    Model interpretation interface using SHAP values.

    Examples:
        # Quick start with defaults
        >>> interp = Interpreter(model, X, y)
        >>> interp.plot_global_importance()
        >>> interp.summary()

        # Use preset configuration
        >>> interp = Interpreter(model, X, y, config='detailed_analysis')

        # Custom configuration
        >>> config = Config(sample_size=5000, top_n_default=20)
        >>> interp = Interpreter(model, X, y, config=config)
    """

    def __init__(
        self,
        model,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        config: Optional[str | Config] = None,
        compute_shap_now: bool = True
    ):
        """
        Initialize interpreter.

        Parameters:
            model: Trained classifier model
            X: Feature matrix
            y: Target variable (optional, used for validation and performance metrics)
            config: Config object, preset name, or None (default config)
            compute_shap_now: Compute SHAP values immediately (default: True)
        """
        self.model = model
        self.X = X
        self.y = y

        # Handle configuration
        if config is None:
            self.config = Config()
        elif isinstance(config, str):
            if config not in PRESETS:
                raise ValueError(f"Unknown preset: {config}. Available: {list(PRESETS.keys())}")
            self.config = PRESETS[config]
        else:
            self.config = config

        # Validate model
        if not check_model_fitted(model):
            warnings.warn("Model may not be fitted. Results may be unreliable.", UserWarning)

        # Validate data
        if self.config.warn_on_issues:
            self.data_issues = validate_data(X, y, warn=True)

        # Storage for computed values
        self.shap_values = None
        self.explainer = None
        self.base_value = None
        self.feature_names = list(X.columns)
        self.categorical_groups = None
        self.category_breakdown = None

        # Aggregated results (after categorical aggregation)
        self.shap_values_agg = None
        self.feature_names_agg = None

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

        # Compute SHAP values
        self.shap_values, self.explainer = compute_shap_values(
            self.model,
            self.X,
            algorithm=self.config.shap_algorithm,
            sample_size=self.config.sample_size,
            random_state=42
        )

        self.base_value = get_base_value(self.explainer)

        # Aggregate categorical features
        self.shap_values_agg, self.feature_names_agg, self.category_breakdown = aggregate_categorical_shap(
            self.shap_values,
            self.feature_names,
            categorical_features=self.config.categorical_features,
            auto_detect=self.config.auto_detect_categorical,
            aggregation=self.config.aggregation_method
        )

        # Store categorical groups
        if self.config.auto_detect_categorical:
            self.categorical_groups = detect_categorical_groups(self.feature_names)

    def get_importance(self, aggregated: bool = True, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.

        Parameters:
            aggregated: Use aggregated features (with categorical combined)
            top_n: Return top N features (default: config.top_n_default)

        Returns:
            DataFrame with feature names and importance values
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        if top_n is None:
            top_n = self.config.top_n_default

        # Choose which SHAP values to use
        if aggregated and self.shap_values_agg is not None:
            shap_vals = self.shap_values_agg
            feat_names = self.feature_names_agg
        else:
            shap_vals = self.shap_values
            feat_names = self.feature_names

        # Calculate importance
        importance = np.abs(shap_vals).mean(axis=0)

        df = pd.DataFrame({
            'feature': feat_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        if top_n:
            df = df.head(top_n)

        return df

    def plot_global_importance(self, aggregated: bool = True, top_n: Optional[int] = None, **kwargs):
        """
        Plot global feature importance.

        Parameters:
            aggregated: Use aggregated features
            top_n: Number of top features (default: config.top_n_default)
            **kwargs: Additional arguments for plot_global_importance()

        Returns:
            Plotly figure
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        if top_n is None:
            top_n = self.config.top_n_default

        if aggregated and self.shap_values_agg is not None:
            shap_vals = self.shap_values_agg
            feat_names = self.feature_names_agg
        else:
            shap_vals = self.shap_values
            feat_names = self.feature_names

        return plot_global_importance(shap_vals, feat_names, top_n=top_n, **kwargs)

    def plot_feature_distribution(self, feature_name: str, **kwargs):
        """
        Plot distribution of a feature.

        Parameters:
            feature_name: Name of feature to plot
            **kwargs: Additional arguments for plot_feature_distribution()

        Returns:
            Plotly figure
        """
        return plot_feature_distribution(self.X, feature_name, **kwargs)

    def plot_categorical_breakdown(self, category_name: str, top_n: Optional[int] = None, **kwargs):
        """
        Plot breakdown of categorical feature.

        Parameters:
            category_name: Name of categorical feature
            top_n: Number of top values (default: config.top_n_default)
            **kwargs: Additional arguments for plot_categorical_breakdown()

        Returns:
            Plotly figure
        """
        if self.category_breakdown is None:
            raise RuntimeError("No categorical breakdown available. Ensure auto_detect_categorical=True or provide categorical_features.")

        if top_n is None:
            top_n = self.config.top_n_default

        return plot_categorical_breakdown(self.category_breakdown, category_name, top_n=top_n, **kwargs)

    def get_categorical_breakdown(self, category_name: str) -> pd.DataFrame:
        """
        Get breakdown DataFrame for a categorical feature.

        Parameters:
            category_name: Name of categorical feature

        Returns:
            DataFrame with category values and their importance
        """
        if self.category_breakdown is None:
            raise RuntimeError("No categorical breakdown available.")

        return get_category_breakdown(category_name, self.category_breakdown)

    def plot_performance(self, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None):
        """
        Plot model performance metrics.

        Parameters:
            y_pred: Predicted labels (if None, will call model.predict())
            y_proba: Predicted probabilities (if None, will call model.predict_proba())

        Returns:
            Dictionary of plotly figures
        """
        if self.y is None:
            raise RuntimeError("Target variable y not provided. Cannot compute performance metrics.")

        # Get predictions if not provided
        if y_pred is None:
            y_pred = self.model.predict(self.X)

        if y_proba is None and hasattr(self.model, 'predict_proba'):
            y_proba_full = self.model.predict_proba(self.X)
            if y_proba_full.shape[1] == 2:
                y_proba = y_proba_full[:, 1]  # Binary positive class
            else:
                y_proba = y_proba_full

        # Calculate metrics
        metrics = calculate_metrics(self.y, y_pred, y_proba)

        plots = {
            'metrics_summary': plot_metrics_summary(metrics),
            'confusion_matrix': plot_confusion_matrix(self.y, y_pred)
        }

        # Add ROC curve if binary classification with probabilities
        if y_proba is not None and len(y_proba.shape) == 1:
            plots['roc_curve'] = plot_roc_curve(self.y, y_proba)
            plots['calibration_curve'] = plot_calibration_curve(self.y, y_proba)

        return plots

    def summary(self, top_n: Optional[int] = None) -> Dict:
        """
        Generate summary of key interpretation insights.

        Parameters:
            top_n: Number of top features to include

        Returns:
            Dictionary with summary information
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        if top_n is None:
            top_n = self.config.top_n_default

        summary = {
            'model_type': self.model.__class__.__name__,
            'n_samples': len(self.X),
            'n_features_original': len(self.feature_names),
            'n_features_aggregated': len(self.feature_names_agg) if self.feature_names_agg else len(self.feature_names),
            'base_value': self.base_value,
            'top_features': self.get_importance(aggregated=True, top_n=top_n).to_dict('records')
        }

        # Add categorical info if available
        if self.categorical_groups:
            summary['categorical_groups'] = {
                k: len(v) for k, v in self.categorical_groups.items()
            }

        # Add performance metrics if y provided
        if self.y is not None:
            y_pred = self.model.predict(self.X)
            y_proba = None
            if hasattr(self.model, 'predict_proba'):
                y_proba_full = self.model.predict_proba(self.X)
                if y_proba_full.shape[1] == 2:
                    y_proba = y_proba_full[:, 1]

            summary['performance'] = calculate_metrics(self.y, y_pred, y_proba)

        return summary

    def segment_analysis(
        self,
        segment_column: str,
        segment_values: Optional[List] = None,
        top_n: Optional[int] = None
    ) -> Dict:
        """
        Compare feature importance across segments.

        Parameters:
            segment_column: Column name to segment by
            segment_values: Specific values to compare (default: all unique values)
            top_n: Number of top features to show

        Returns:
            Dictionary with segment analysis results
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        if segment_column not in self.X.columns:
            raise ValueError(f"Segment column '{segment_column}' not found in data")

        if top_n is None:
            top_n = self.config.top_n_default

        # Get segment values
        if segment_values is None:
            segment_values = self.X[segment_column].unique()

        # Filter to segments with enough samples
        segment_values = [
            val for val in segment_values
            if (self.X[segment_column] == val).sum() >= self.config.min_samples_per_segment
        ]

        if len(segment_values) < 2:
            warnings.warn(
                f"Not enough segments with min {self.config.min_samples_per_segment} samples. "
                f"Found {len(segment_values)} segments.",
                UserWarning
            )

        # Compute importance per segment
        shap_values_list = []
        segment_names = []
        segment_sizes = []

        for val in segment_values:
            mask = self.X[segment_column] == val
            segment_shap = self.shap_values[mask]
            shap_values_list.append(segment_shap)
            segment_names.append(str(val))
            segment_sizes.append(mask.sum())

        # Create comparison plot
        comparison_plot = plot_importance_comparison(
            shap_values_list,
            self.feature_names,
            segment_names,
            top_n=top_n,
            title=f"Feature Importance by {segment_column}"
        )

        # Calculate importance for each segment
        segment_importance = {}
        for i, name in enumerate(segment_names):
            importance = np.abs(shap_values_list[i]).mean(axis=0)
            df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            segment_importance[name] = df.to_dict('records')

        return {
            'segment_column': segment_column,
            'segments': segment_names,
            'segment_sizes': segment_sizes,
            'comparison_plot': comparison_plot,
            'segment_importance': segment_importance
        }

    # ========================================================================
    # PHASE 2: Dependence and Interaction Analysis
    # ========================================================================

    def plot_beeswarm(self, top_n: Optional[int] = None, max_display: int = 1000, **kwargs):
        """
        Create beeswarm plot showing SHAP value distributions.

        Parameters:
            top_n: Number of top features (default: config.top_n_default)
            max_display: Maximum samples to display
            **kwargs: Additional arguments for plot_beeswarm()

        Returns:
            Plotly figure
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        if top_n is None:
            top_n = self.config.top_n_default

        return plot_beeswarm(
            self.shap_values,
            self.X,
            self.feature_names,
            top_n=top_n,
            max_display=max_display,
            **kwargs
        )

    def plot_dependence(self, feature_name: str, interaction_feature: Optional[str] = None, **kwargs):
        """
        Create dependence plot for a feature.

        Shows how a feature's value affects its SHAP value (prediction impact).

        Plot axes:
        - X-axis: Feature value (specified by feature_name)
        - Y-axis: SHAP value for that feature
        - Color: Values of interaction_feature (auto-detected if None)

        Parameters:
            feature_name: Feature to plot on X-axis
            interaction_feature: Feature to use for color coding (auto-detected if None)
                                Set to specific feature name to control color axis
            **kwargs: Additional arguments for plot_dependence()

        Returns:
            Plotly figure

        Examples:
            # Auto-detect interaction feature for color
            >>> interp.plot_dependence('discount_offered')

            # Specify color feature explicitly
            >>> interp.plot_dependence('discount_offered', interaction_feature='occupation')

            # No color coding
            >>> interp.plot_dependence('age', interaction_feature='')
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        return plot_dependence(
            self.shap_values,
            self.X,
            feature_name,
            self.feature_names,
            interaction_feature=interaction_feature,
            **kwargs
        )

    def plot_dependence_categorical(self, feature_name: str, value_labels: Optional[dict] = None, **kwargs):
        """
        Create dependence plot for categorical (label-encoded) feature.

        Parameters:
            feature_name: Categorical feature name
            value_labels: Optional dict mapping encoded values to labels
            **kwargs: Additional arguments for plot_dependence_categorical()

        Returns:
            Plotly figure
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        return plot_dependence_categorical(
            self.shap_values,
            self.X,
            feature_name,
            self.feature_names,
            value_labels=value_labels,
            **kwargs
        )

    def detect_interactions(self, top_n: int = 10, method: str = 'shap_variance') -> pd.DataFrame:
        """
        Detect feature interactions.

        Parameters:
            top_n: Number of top interactions to return
            method: 'shap_variance' or 'correlation'

        Returns:
            DataFrame with interaction strengths
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        return detect_interactions(
            self.shap_values,
            self.X,
            self.feature_names,
            top_n=top_n,
            method=method
        )

    def analyze_interaction(self, feature_1: str, feature_2: str) -> Dict:
        """
        Analyze specific interaction between two features.

        Parameters:
            feature_1: First feature name
            feature_2: Second feature name

        Returns:
            Dictionary with interaction analysis
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        return analyze_interaction(
            self.shap_values,
            self.X,
            self.feature_names,
            feature_1,
            feature_2
        )

    def plot_interaction_scatter(self, feature_1: str, feature_2: str, **kwargs):
        """
        Create 2D scatter plot showing interaction between two features.

        Parameters:
            feature_1: First feature name
            feature_2: Second feature name
            **kwargs: Additional arguments for plot_interaction_scatter()

        Returns:
            Plotly figure
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        return plot_interaction_scatter(
            self.shap_values,
            self.X,
            self.feature_names,
            feature_1,
            feature_2,
            **kwargs
        )

    def plot_interaction_heatmap(self, feature_1: str, feature_2: str, bins: int = 10, **kwargs):
        """
        Create heatmap showing average SHAP values across feature combinations.

        Parameters:
            feature_1: First feature name
            feature_2: Second feature name
            bins: Number of bins for continuous features
            **kwargs: Additional arguments for plot_interaction_heatmap()

        Returns:
            Plotly figure
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        return plot_interaction_heatmap(
            self.shap_values,
            self.X,
            self.feature_names,
            feature_1,
            feature_2,
            bins=bins,
            **kwargs
        )

    def plot_interaction_matrix(self, method: str = 'correlation', **kwargs):
        """
        Plot heatmap of feature interaction matrix.

        Parameters:
            method: 'correlation' or 'mutual_info'
            **kwargs: Additional arguments for plot_interaction_matrix()

        Returns:
            Plotly figure
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        matrix = compute_interaction_matrix(
            self.shap_values,
            self.feature_names,
            method=method
        )

        return plot_interaction_matrix(matrix, **kwargs)

    def plot_interaction_contour(
        self,
        feature_1: str,
        feature_2: str,
        n_grid: int = 50,
        other_features_percentile: float = 0.5,
        value_labels_1: Optional[dict] = None,
        value_labels_2: Optional[dict] = None,
        **kwargs
    ):
        """
        Create blocky heatmap showing predicted probability surface across two features.

        For features with <10 unique values (e.g., discount_offered with [0,10,20,30]),
        creates discrete blocks. Categorical features can be labeled with actual names.

        This visualization shows how different combinations of two features affect
        the model's predicted probability (Y) directly, making it easier to understand
        their joint effect on predictions.

        Parameters:
            feature_1: First feature name (X-axis)
            feature_2: Second feature name (Y-axis)
            n_grid: Grid resolution for continuous features (higher = smoother but slower)
            other_features_percentile: Value (0-1) to use for other features
            value_labels_1: Optional dict mapping feature_1 values to labels (e.g., {0: 'student', 1: 'professional'})
            value_labels_2: Optional dict mapping feature_2 values to labels
            **kwargs: Additional arguments (title, height)

        Returns:
            Plotly figure

        Examples:
            # Discrete features show blocky heatmap
            >>> interp.plot_interaction_contour('discount_offered', 'engagement_score')

            # With categorical labels
            >>> occupation_labels = {0: 'professional', 1: 'retired', 2: 'student', 3: 'unemployed'}
            >>> interp.plot_interaction_contour('discount_offered', 'occupation',
            ...                                   value_labels_2=occupation_labels)

            # Continuous features with higher resolution
            >>> interp.plot_interaction_contour('age', 'income', n_grid=100,
            ...                                   other_features_percentile=0.25)
        """
        return plot_interaction_contour(
            self.model,
            self.X,
            feature_1,
            feature_2,
            n_grid=n_grid,
            other_features_percentile=other_features_percentile,
            value_labels_1=value_labels_1,
            value_labels_2=value_labels_2,
            **kwargs
        )

    def plot_interaction_surface_3d(
        self,
        feature_1: str,
        feature_2: str,
        n_grid: int = 30,
        other_features_percentile: float = 0.5,
        value_labels_1: Optional[dict] = None,
        value_labels_2: Optional[dict] = None,
        **kwargs
    ):
        """
        Create 3D surface plot showing predicted probability across two features.

        Similar to contour plot but in 3D. Z-axis represents predicted probability,
        making peaks and valleys in the prediction surface visually obvious.

        For features with <10 unique values, uses discrete grid points.
        Categorical features can be labeled with actual names.

        Parameters:
            feature_1: First feature name (X-axis)
            feature_2: Second feature name (Y-axis)
            n_grid: Grid resolution for continuous features (lower for 3D to maintain performance)
            other_features_percentile: Value (0-1) to use for other features
            value_labels_1: Optional dict mapping feature_1 values to labels
            value_labels_2: Optional dict mapping feature_2 values to labels
            **kwargs: Additional arguments (title, height)

        Returns:
            Plotly figure

        Examples:
            # 3D view of how discount and occupation affect conversion
            >>> interp.plot_interaction_surface_3d('discount_offered', 'occupation')

            # With categorical labels
            >>> occupation_labels = {0: 'professional', 1: 'retired', 2: 'student', 3: 'unemployed'}
            >>> interp.plot_interaction_surface_3d('discount_offered', 'occupation',
            ...                                     value_labels_2=occupation_labels)
        """
        return plot_interaction_surface_3d(
            self.model,
            self.X,
            feature_1,
            feature_2,
            n_grid=n_grid,
            other_features_percentile=other_features_percentile,
            value_labels_1=value_labels_1,
            value_labels_2=value_labels_2,
            **kwargs
        )

    def plot_conditional_dependence(
        self,
        feature_name: str,
        condition_feature: str,
        n_bins: int = 4,
        value_labels: Optional[dict] = None,
        **kwargs
    ):
        """
        Show how a feature's effect varies depending on another feature's value.

        Creates separate dependence lines for different bins of the conditioning
        feature. If lines are parallel, there's no interaction. If they diverge
        significantly, there's a strong interaction.

        For discrete conditioning features (<10 unique values), uses actual values with optional labels.
        For continuous features, bins into quantiles.

        Parameters:
            feature_name: Feature to show dependence for (X-axis)
            condition_feature: Feature to condition on (creates separate lines)
            n_bins: Number of bins for continuous condition_feature (ignored if discrete)
            value_labels: Optional dict mapping condition_feature values to labels (e.g., {0: 'student', 1: 'professional'})
            **kwargs: Additional arguments (title, height)

        Returns:
            Plotly figure

        Examples:
            # Does discount effect vary by occupation?
            >>> interp.plot_conditional_dependence('discount_offered', 'occupation')

            # With categorical labels
            >>> occupation_labels = {0: 'professional', 1: 'retired', 2: 'student', 3: 'unemployed'}
            >>> interp.plot_conditional_dependence('discount_offered', 'occupation',
            ...                                     value_labels=occupation_labels)

            # Does engagement effect vary by previous course count?
            >>> interp.plot_conditional_dependence('engagement_score', 'previous_courses', n_bins=3)
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap() first.")

        return plot_conditional_dependence(
            self.shap_values,
            self.X,
            self.feature_names,
            feature_name,
            condition_feature,
            n_bins=n_bins,
            value_labels=value_labels,
            **kwargs
        )
