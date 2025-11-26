"""
Analysis modules for interaction detection and advanced interpretation.
"""

from .interaction_detection import (
    detect_interactions,
    analyze_interaction,
    compute_interaction_matrix
)
from .threshold_detection import (
    detect_thresholds,
    detect_all_thresholds
)
from .segment_discovery import (
    discover_segments,
    plot_segment_profiles,
    plot_segment_comparison,
    get_segment_summary
)

__all__ = [
    'detect_interactions',
    'analyze_interaction',
    'compute_interaction_matrix',
    'detect_thresholds',
    'detect_all_thresholds',
    'discover_segments',
    'plot_segment_profiles',
    'plot_segment_comparison',
    'get_segment_summary'
]
