# Classifier Model Interpreter

A Python package for interpreting and explaining machine learning classification models with interactive visualizations and business-friendly insights.

## Overview

This package provides advanced model interpretation capabilities built on SHAP (SHapley Additive exPlanations) with a focus on:
- **Interactive Plotly visualizations** instead of static matplotlib
- **Prediction surface analysis** (2D heatmaps & 3D surfaces) - NOT available in native SHAP
- **Smart discrete/continuous feature handling** with auto-formatted axes
- **Conditional dependence plots** that make interactions visually obvious
- **Business-friendly outputs** showing probabilities and actionable insights

## Key Features

### Unique Capabilities (Not in Native SHAP)

1. **Blocky Heatmaps** - Prediction probability surfaces
   - Auto-detects discrete features (<10 unique values)
   - Shows only actual values on axes (e.g., 0,10,20,30 not 0,5,10,15...)
   - Categorical labels (e.g., "student" vs "0")
   - Continuous features use clean auto-formatted ticks
   - No interpolation smoothing for clearer interpretation

2. **3D Surface Plots** - Interactive prediction landscapes
   - Rotate, zoom, hover capabilities
   - Great for presentations
   - Intuitive for non-technical audiences

3. **Conditional Dependence Plots** - Shows how feature effects vary
   - Makes interactions visually obvious
   - Answers "does X effect depend on Y?" questions directly
   - Better than scatter for categorical conditioning

4. **Categorical Box Plots** - Specialized categorical visualizations
   - Clear distributions vs messy scatter plots

### Better Implementation Than Native SHAP

- **Interactive Plotly** vs static matplotlib (hover, zoom, export)
- **Simple API** - One Interpreter class vs multiple imports
- **Preset configs** - Quick setup vs manual parameters
- **Clean outputs** - DataFrames vs complex objects

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- shap
- plotly
- scipy
- matplotlib

## Quick Start

```python
from src.core import Interpreter
import pandas as pd
from xgboost import XGBClassifier

# Train your model
model = XGBClassifier()
model.fit(X_train, y_train)

# Initialize interpreter
interp = Interpreter(model, X_test, y_test, config='detailed_analysis')

# Global feature importance
fig = interp.plot_global_importance(top_n=10)
fig.show()

# Beeswarm plot
fig = interp.plot_beeswarm(top_n=15)
fig.show()

# Feature dependence
fig = interp.plot_dependence('engagement_score')
fig.show()

# Detect interactions
interactions = interp.detect_interactions(top_n=10)
print(interactions)

# Prediction heatmap (discrete × continuous)
fig = interp.plot_interaction_contour('discount_offered', 'engagement_score')
fig.show()

# Categorical labels
occupation_labels = {0: 'professional', 1: 'retired', 2: 'student', 3: 'unemployed'}
fig = interp.plot_interaction_contour(
    'discount_offered',
    'occupation',
    value_labels_2=occupation_labels
)
fig.show()

# 3D surface
fig = interp.plot_interaction_surface_3d('discount_offered', 'engagement_score')
fig.show()

# Conditional dependence
fig = interp.plot_conditional_dependence('discount_offered', 'occupation')
fig.show()
```

## Demo Notebook

See `demo_all_visualizations.ipynb` for a comprehensive demonstration of all visualization capabilities with detailed explanations and comparisons to native SHAP.

The demo includes:
- Global feature importance
- Beeswarm plots
- Feature dependence (numeric & categorical)
- Interaction detection & analysis
- Traditional interaction visualizations
- **NEW:** Blocky heatmaps with smart axis handling
- 3D surface plots
- Conditional dependence plots
- Summary comparison vs native SHAP

## API Reference

### Interpreter Class

```python
from src.core import Interpreter

interp = Interpreter(
    model,              # Trained model with predict_proba
    X,                  # Feature matrix (pandas DataFrame)
    y,                  # Target labels
    feature_names=None, # Optional: list of feature names
    config='detailed_analysis'  # Preset config
)
```

### Global Visualizations

```python
# Feature importance bar chart
interp.plot_global_importance(top_n=15)

# Beeswarm plot (SHAP value distributions)
interp.plot_beeswarm(top_n=15)
```

### Dependence Plots

```python
# Numeric feature dependence
interp.plot_dependence('feature_name', interaction_feature='other_feature')

# Categorical feature box plots
interp.plot_dependence_categorical('category_feature', value_labels={0: 'A', 1: 'B'})
```

### Interaction Analysis

```python
# Detect top interactions
interactions = interp.detect_interactions(top_n=15, method='shap_variance')

# Analyze specific interaction
analysis = interp.analyze_interaction('feature_1', 'feature_2')

# Scatter plot
interp.plot_interaction_scatter('feature_1', 'feature_2')

# Heatmap
interp.plot_interaction_heatmap('feature_1', 'feature_2', bins=10)
```

### Prediction Surfaces (NEW!)

```python
# 2D blocky heatmap
interp.plot_interaction_contour(
    'feature_1',
    'feature_2',
    n_grid=50,  # Grid resolution for continuous features
    value_labels_1={0: 'A', 1: 'B'},  # Optional categorical labels
    value_labels_2={0: 'X', 1: 'Y'}
)

# 3D surface
interp.plot_interaction_surface_3d(
    'feature_1',
    'feature_2',
    n_grid=30,
    value_labels_1=None,
    value_labels_2=None
)

# Conditional dependence
interp.plot_conditional_dependence(
    'feature_name',
    'condition_feature',
    n_bins=4,  # Number of bins for continuous conditioning feature
    value_labels={0: 'A', 1: 'B'}  # Labels for conditioning feature
)
```

## Smart Axis Handling

The package automatically detects feature types and handles axes appropriately:

**Discrete Features** (<10 unique values):
- Shows ONLY actual values on axis (e.g., 0, 10, 20, 30 for discounts)
- No meaningless intermediate ticks (no 5, 15, 25)
- Can display categorical labels instead of encoded numbers

**Continuous Features**:
- Uses n_grid points for smooth prediction surface (default 50 for heatmaps, 30 for 3D)
- Auto-formatted, clean tick marks for interpretability
- No cluttered axes

**All Features**:
- Blocky heatmap (no smoothing interpolation)
- Consistent visual style
- Clear cell-by-cell interpretation

## Business Value

- Shows **predictions** (probabilities) not just SHAP values
- Visualizations non-technical stakeholders understand
- Directly answers business questions like "What discount level maximizes conversion for students?"
- Great for presentations and reports
- Interactive dashboards
- Clear discrete feature visualization

## Project Structure

```
prod_code/
├── README.md                           # This file
├── requirements.txt                    # Package dependencies
├── demo_all_visualizations.ipynb      # Comprehensive demo notebook
└── src/
    ├── __init__.py
    ├── core/
    │   ├── __init__.py
    │   ├── interpreter.py              # Main Interpreter class
    │   └── config.py                   # Configuration presets
    ├── explainers/
    │   ├── __init__.py
    │   ├── shap_explainer.py          # SHAP computation
    │   └── categorical_handler.py      # Categorical feature handling
    ├── visualization/
    │   ├── __init__.py
    │   ├── global_plots.py            # Global importance & beeswarm
    │   ├── dependence_plots.py        # Feature dependence plots
    │   ├── interaction_plots.py       # Interaction visualizations
    │   └── performance_plots.py       # Performance metrics
    ├── analysis/
    │   ├── __init__.py
    │   └── interaction_detection.py   # Interaction detection methods
    └── utils/
        ├── __init__.py
        ├── data_utils.py              # Data manipulation utilities
        └── model_utils.py             # Model helper functions
```

## When to Use This Package

✅ Business presentations
✅ Finding optimal feature combinations
✅ Understanding interactions clearly
✅ Interactive dashboards
✅ Stakeholder communication
✅ Rapid model interpretation
✅ Working with categorical or discrete features

## When to Use Native SHAP

✅ Academic papers (matplotlib standard)
✅ Force plots for single predictions
✅ Advanced TreeSHAP features
✅ Non-tree models (we focus on tree models)

## Best Approach

Use **both**! This package for business insights and interactive exploration, native SHAP for advanced technical features.

## Examples from Demo

### Discrete × Continuous Heatmap
```python
# Shows discount values 0,10,20,30 only on X-axis
# Engagement score gets clean auto-formatted ticks on Y-axis
fig = interp.plot_interaction_contour('discount_offered', 'engagement_score')
```

### Categorical Labels
```python
# Y-axis shows "professional", "student", etc. instead of 0, 1, 2, 3
occupation_labels = {0: 'professional', 1: 'retired', 2: 'student', 3: 'unemployed'}
fig = interp.plot_interaction_contour(
    'discount_offered',
    'occupation',
    value_labels_2=occupation_labels
)
```

### Conditional Dependence
```python
# Does discount effectiveness vary by occupation?
# Diverging lines = YES, parallel lines = NO
fig = interp.plot_conditional_dependence(
    'discount_offered',
    'occupation',
    value_labels=occupation_labels
)
```

## License

This package is provided for educational and business use. Please ensure you have appropriate licenses for all dependencies (SHAP, Plotly, XGBoost, etc.).

## Contributing

For questions, issues, or contributions, please contact the repository maintainer.

## Acknowledgments

Built on top of:
- **SHAP** by Scott Lundberg et al.
- **Plotly** for interactive visualizations
- **XGBoost** for tree-based models
- **scikit-learn** for ML utilities

---

**Generated with Claude Code** 🤖
