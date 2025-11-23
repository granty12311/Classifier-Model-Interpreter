"""
Analysis modules for interaction detection and advanced interpretation.
"""

from .interaction_detection import (
    detect_interactions,
    analyze_interaction,
    compute_interaction_matrix
)

__all__ = [
    'detect_interactions',
    'analyze_interaction',
    'compute_interaction_matrix'
]
