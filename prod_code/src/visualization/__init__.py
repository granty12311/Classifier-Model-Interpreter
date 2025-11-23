"""
Visualization modules for model interpretation.
"""

from .global_plots import (
    plot_global_importance,
    plot_feature_distribution,
    plot_importance_comparison,
    plot_categorical_breakdown
)
from .performance_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    calculate_metrics,
    plot_metrics_summary,
    plot_calibration_curve
)
from .dependence_plots import (
    plot_beeswarm,
    plot_dependence,
    plot_dependence_categorical
)
from .interaction_plots import (
    plot_interaction_scatter,
    plot_interaction_heatmap,
    plot_interaction_matrix
)

__all__ = [
    'plot_global_importance',
    'plot_feature_distribution',
    'plot_importance_comparison',
    'plot_categorical_breakdown',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'calculate_metrics',
    'plot_metrics_summary',
    'plot_calibration_curve',
    'plot_beeswarm',
    'plot_dependence',
    'plot_dependence_categorical',
    'plot_interaction_scatter',
    'plot_interaction_heatmap',
    'plot_interaction_matrix'
]
