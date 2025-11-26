"""
Visualization modules for model interpretation.
"""

from .global_plots import plot_global_importance
from .performance_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    calculate_metrics,
    plot_metrics_summary
)
from .dependence_plots import (
    plot_beeswarm,
    plot_dependence,
    plot_dependence_categorical
)
from .interaction_plots import (
    plot_interaction_contour,
    plot_interaction_surface_3d
)

__all__ = [
    'plot_global_importance',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'calculate_metrics',
    'plot_metrics_summary',
    'plot_beeswarm',
    'plot_dependence',
    'plot_dependence_categorical',
    'plot_interaction_contour',
    'plot_interaction_surface_3d'
]
