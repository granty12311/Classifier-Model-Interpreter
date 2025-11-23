"""
Configuration classes and presets for model interpretation.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Config:
    """Global configuration for ModelInterpreter."""

    # SHAP settings
    sample_size: int = 2000
    shap_algorithm: str = 'auto'  # 'auto', 'tree', 'kernel'

    # Categorical handling
    categorical_features: Optional[List[str]] = None
    auto_detect_categorical: bool = True
    aggregation_method: str = 'sum'  # 'sum', 'mean'

    # Visualization
    top_n_default: int = 10
    plot_backend: str = 'plotly'  # 'plotly', 'matplotlib'
    figure_size: Tuple[int, int] = (10, 6)

    # Validation
    warn_on_issues: bool = True
    min_samples_per_segment: int = 30

    # Performance
    n_jobs: int = -1
    verbose: bool = False


# Preset configurations
PRESETS = {
    'quick_summary': Config(
        sample_size=1000,
        top_n_default=5,
        shap_algorithm='tree',
        verbose=False
    ),

    'detailed_analysis': Config(
        sample_size=2000,
        top_n_default=15,
        auto_detect_categorical=True,
        verbose=False
    ),

    'segment_focus': Config(
        sample_size=500,  # Per segment
        min_samples_per_segment=50,
        top_n_default=10,
        verbose=False
    )
}
