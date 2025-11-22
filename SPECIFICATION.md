# Classifier Model Interpreter - Technical Specification v2.0

**Version**: 2.0 (Streamlined)
**Date**: 2025-11-22
**Status**: Ready for Implementation

---

## Executive Summary

A **focused Model Interpretation Package** that provides clear, global-scale insights into classifier behavior through SHAP analysis. Designed for exploratory analysis and segment-level understanding, not production deployment or instance-by-instance vetting.

### Core Philosophy

1. **Global Over Local**: Understand model patterns, not individual predictions
2. **Segments Over Instances**: Analyze customer groups, not one-offs
3. **Simplicity Over Features**: Reusable functions, not complex systems
4. **Warnings Over Errors**: Alert users to issues, don't block analysis
5. **Notebooks Over Dashboards**: Interactive exploration, not static reports

---

## Table of Contents

1. [Scope & Focus](#1-scope--focus)
2. [Core Capabilities](#2-core-capabilities)
3. [Package Dependencies](#3-package-dependencies)
4. [Module Structure](#4-module-structure)
5. [Key Features](#5-key-features)
6. [Categorical Handling](#6-categorical-handling)
7. [Segment Analysis](#7-segment-analysis)
8. [Implementation Plan](#8-implementation-plan)
9. [Usage Examples](#9-usage-examples)
10. [Testing Strategy](#10-testing-strategy)

---

## 1. Scope & Focus

### In Scope ✅

**Global Model Understanding**
- Overall feature importance across all predictions
- Segment-level patterns (e.g., professionals vs students)
- Feature interactions and dependencies
- Model performance context

**Visualization Focus**
- In-line plots in Jupyter notebooks
- Matplotlib and Plotly figures
- Interactive exploration (not static dashboards)
- Easy to save/export individual charts

**Data Handling**
- Categorical variable aggregation (sum SHAP values, no one-hot)
- Sampling for large datasets (1-2k instances)
- Warnings for data issues (not errors)
- Works with clean, prepared data

**Model Support**
- Sklearn-compatible classifiers
- Tree-based models (XGBoost, LightGBM, CatBoost)
- Linear models (LogisticRegression)
- Reusable across model types

### Out of Scope ❌

**Not Building**
- Web dashboards or applications
- HTML/PDF report generation
- API integrations or microservices
- Real-time prediction explanations
- Automated validation pipelines
- LIME or alternative explainers
- Deep instance-level analysis
- Production deployment features

**Assumptions**
- Data is generally clean and correct
- Users will validate inputs themselves
- Focus on exploration, not production
- Testing via notebooks, not unit tests
- Users understand basic ML concepts

---

## 2. Core Capabilities

### Capability 1: Global Feature Importance

**What**: Understand which features drive model predictions overall

**Outputs**:
- Bar chart of mean absolute SHAP values
- Beeswarm plot showing value distributions
- Feature importance table with statistics
- Model performance metrics for context

**Key Insight**: "Engagement (pages_viewed, time_on_site) drives 65% of model decisions"

---

### Capability 2: Segment Analysis

**What**: Compare how model behaves for different customer groups

**Outputs**:
- Feature importance by segment (occupation, country, etc.)
- Segment prediction distributions
- Comparison visualizations
- Segment-specific insights

**Key Insight**: "Discounts are 3x more important for students than professionals"

---

### Capability 3: Feature Dependence

**What**: Show how changing a feature affects predictions

**Outputs**:
- Scatter plots: feature value vs SHAP value
- Interaction detection (2-way dependencies)
- Trend lines and patterns
- Distribution overlays

**Key Insight**: "Discount effectiveness increases with engagement level (interaction)"

---

### Capability 4: Categorical Intelligence

**What**: Properly aggregate and display categorical variables

**Outputs**:
- Grouped SHAP values for categories
- Category-level importance rankings
- Breakdown by category value
- No dummy variable confusion

**Key Insight**: "Occupation contributes +8pp overall (professionals +6pp, students +2pp)"

---

### Capability 5: Model Performance Context

**What**: Show model quality alongside explanations

**Outputs**:
- Accuracy, AUC, precision, recall
- Confusion matrix
- Calibration information
- Confidence in explanations

**Key Insight**: "Model achieves 85% accuracy (AUC: 0.92) - explanations are reliable"

---

## 3. Package Dependencies

### Core Requirements

```python
# requirements.txt

# Core
shap>=0.42.0           # Primary interpretation engine
scikit-learn>=1.3.0    # Model interface
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical operations

# Visualization
matplotlib>=3.7.0      # Base plotting
plotly>=5.14.0         # Interactive plots
seaborn>=0.12.0        # Statistical visualization

# Optional (for specific models)
xgboost>=1.7.0         # If using XGBoost
lightgbm>=4.0.0        # If using LightGBM
catboost>=1.2.0        # If using CatBoost

# Jupyter
ipywidgets>=8.0.0      # Interactive widgets
```

### Why SHAP Only?

**Advantages**:
- ✅ Mathematically rigorous (Shapley values)
- ✅ Handles interactions naturally
- ✅ TreeExplainer is extremely fast
- ✅ Industry standard for interpretability
- ✅ Works with sampling for large datasets

**Decision**: Focus 100% on SHAP, skip LIME and other methods

---

## 4. Module Structure

```
classifier_model_interpreter/
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── interpreter.py          # Main API class
│   │   └── config.py               # Simple configuration
│   │
│   ├── explainers/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py       # SHAP computation
│   │   └── categorical_handler.py  # Category aggregation
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── global_plots.py         # Feature importance plots
│   │   ├── segment_plots.py        # Segment comparison plots
│   │   ├── dependence_plots.py     # Feature dependence
│   │   └── performance_plots.py    # Model performance viz
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── segment_analyzer.py     # Segment-level analysis
│   │   ├── interaction_detector.py # Feature interactions
│   │   └── pattern_finder.py       # Global pattern detection
│   │
│   └── utils/
│       ├── __init__.py
│       ├── data_utils.py           # Sampling, preprocessing
│       ├── model_utils.py          # Model wrapper
│       └── validation_utils.py     # Soft validation (warnings only)
│
├── notebooks/
│   ├── 01_global_interpretation.ipynb
│   ├── 02_segment_analysis.ipynb
│   ├── 03_categorical_handling.ipynb
│   ├── 04_feature_interactions.ipynb
│   └── 05_full_workflow_example.ipynb
│
├── data/
│   └── customer_conversion.csv
│
├── requirements.txt
├── SPECIFICATION_v2.md            # This file
└── README.md
```

**Design Principles**:
- Simple, flat structure
- Reusable functions (not classes where possible)
- Each module does one thing well
- Easy to understand and extend

---

## 5. Key Features

### 5.1 Global Feature Importance

**Function Signature**:
```python
def plot_global_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray = None,
    categorical_features: List[str] = None,
    top_n: int = 10,
    sample_size: int = 2000,
    plot_type: str = 'bar',  # 'bar', 'beeswarm'
    show_performance: bool = True
) -> plotly.graph_objects.Figure:
    """
    Display global feature importance using SHAP values.

    Parameters:
        model: Trained sklearn-compatible classifier
        X: Feature matrix
        y: Target variable (optional, for performance metrics)
        categorical_features: List of categorical column names
        top_n: Number of top features to show
        sample_size: Max samples for SHAP (use random sample if X is larger)
        plot_type: 'bar' for mean |SHAP|, 'beeswarm' for distribution
        show_performance: Include model performance metrics

    Returns:
        Plotly figure (can be displayed with fig.show())

    Notes:
        - Automatically aggregates categorical SHAP values
        - Warns if data issues detected (doesn't error)
        - Uses TreeExplainer for tree models, KernelExplainer otherwise
    """
```

**Output Example**:
```
Global Feature Importance
Model Performance: Accuracy=85%, AUC=0.92

Feature                    Mean |SHAP Value|
────────────────────────────────────────────
pages_viewed               0.42 ████████████
time_on_site_mins          0.35 ██████████
discount_offered           0.28 ████████
occupation                 0.18 █████
videos_watched             0.15 ████
previous_courses           0.12 ███
referral_source            0.08 ██
age                        0.05 █
device_type                0.03 █
```

---

### 5.2 Feature Distribution Context

**Function Signature**:
```python
def plot_feature_distribution(
    feature_name: str,
    X: pd.DataFrame,
    highlight_value: Any = None,
    show_quantiles: bool = True
) -> plotly.graph_objects.Figure:
    """
    Show distribution of feature values with optional highlight.

    Parameters:
        feature_name: Name of feature to plot
        X: Feature matrix
        highlight_value: Optional value to highlight (e.g., for comparison)
        show_quantiles: Show 25th, 50th, 75th percentile lines

    Returns:
        Plotly histogram/KDE plot

    Example Use:
        # Show where a customer stands
        plot_feature_distribution('pages_viewed', X, highlight_value=52)
        # → "Your value (52) is at 90th percentile"
    """
```

---

### 5.3 Segment Analysis

**Function Signature**:
```python
def compare_segments(
    model,
    X: pd.DataFrame,
    segment_column: str,
    categorical_features: List[str] = None,
    sample_size: int = 2000,
    segments_to_compare: List[Any] = None  # None = all segments
) -> Dict:
    """
    Compare model behavior across segments.

    Parameters:
        model: Trained classifier
        X: Feature matrix (must include segment_column)
        segment_column: Column to segment by (e.g., 'occupation')
        categorical_features: List of categorical columns
        sample_size: Max samples per segment
        segments_to_compare: Specific segments (None = all)

    Returns:
        {
            'segment_sizes': {segment: count},
            'feature_importance_by_segment': {segment: {feature: importance}},
            'prediction_distributions': {segment: [predictions]},
            'key_differences': {feature: {segment1: val, segment2: val}}
        }

    Example Use:
        results = compare_segments(model, X, 'occupation')
        # → Shows how discount matters more to students than professionals
    """
```

**Visualization Function**:
```python
def plot_segment_comparison(
    segment_results: Dict,
    feature_to_compare: str = None,  # None = all features
    comparison_type: str = 'importance'  # 'importance', 'predictions'
) -> plotly.graph_objects.Figure:
    """
    Visualize segment comparison results.

    Parameters:
        segment_results: Output from compare_segments()
        feature_to_compare: Specific feature (None = top features)
        comparison_type: What to compare

    Returns:
        Grouped bar chart or heatmap
    """
```

---

### 5.4 Feature Dependence with Interactions

**Function Signature**:
```python
def plot_feature_dependence(
    feature_name: str,
    model,
    X: pd.DataFrame,
    interaction_feature: str = 'auto',  # 'auto' detects strongest
    categorical_features: List[str] = None,
    sample_size: int = 2000
) -> plotly.graph_objects.Figure:
    """
    Show how feature affects predictions (with optional interaction).

    Parameters:
        feature_name: Primary feature to analyze
        model: Trained classifier
        X: Feature matrix
        interaction_feature: Feature to color by ('auto' finds strongest)
        categorical_features: List of categorical columns
        sample_size: Max samples to use

    Returns:
        Scatter plot: feature value vs SHAP value
        Color dimension: interaction feature (if provided)

    Example Use:
        plot_feature_dependence('discount_offered', model, X)
        # → Might auto-detect that occupation interacts with discount
        # → Shows discount is more effective for students
    """
```

---

### 5.5 Model Performance Dashboard

**Function Signature**:
```python
def plot_performance_summary(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    show_calibration: bool = True
) -> plotly.graph_objects.Figure:
    """
    Create 2x2 performance overview.

    Panels:
    1. Confusion Matrix
    2. ROC Curve
    3. Precision-Recall Curve
    4. Calibration Plot (predicted vs actual)

    Parameters:
        model: Trained classifier
        X: Feature matrix
        y: True labels
        show_calibration: Include calibration analysis

    Returns:
        Plotly figure with subplots
    """
```

---

## 6. Categorical Handling

### Philosophy: Aggregate, Don't Separate

**Problem**: One-hot encoding creates confusing explanations
```
# Bad (one-hot approach)
occupation_student        SHAP: +0.02
occupation_professional   SHAP: +0.06
occupation_self_employed  SHAP: +0.01
# → User sees 3 separate features, doesn't understand "occupation" impact
```

**Solution**: Sum SHAP values by category
```
# Good (aggregated approach)
occupation                SHAP: +0.09
  └─ professional:        +0.06
  └─ student:             +0.02
  └─ self_employed:       +0.01
# → User sees occupation's total impact, can drill down if needed
```

### Implementation

**Function Signature**:
```python
def aggregate_categorical_shap(
    shap_values: np.ndarray,
    feature_names: List[str],
    categorical_features: List[str],
    aggregation: str = 'sum'  # 'sum', 'mean', 'max'
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Aggregate SHAP values for categorical features.

    Detection Logic:
    1. User-specified categorical_features list
    2. Auto-detect one-hot patterns (feature_A, feature_B, feature_C)
    3. Auto-detect by dtype (object, category)

    Parameters:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: Original feature names
        categorical_features: List of categorical column names
        aggregation: How to combine dummy variables

    Returns:
        aggregated_shap: SHAP values with categoricals combined
        aggregated_names: Feature names with categoricals grouped
        category_breakdown: Dict mapping category to component breakdown

    Example:
        # Input: color_red, color_blue, color_green
        # Output: color (with breakdown available)
    """
```

**Auto-Detection**:
```python
def detect_categorical_groups(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Automatically detect one-hot encoded categorical groups.

    Detection patterns:
    - feature_value1, feature_value2 → feature
    - is_value1, is_value2 → is_*
    - Common prefixes with different suffixes

    Returns:
        {
            'color': ['color_red', 'color_blue', 'color_green'],
            'occupation': ['occupation_student', 'occupation_professional']
        }
    """
```

---

## 7. Segment Analysis

### 7.1 Segment Comparison Workflow

**Step 1: Identify Segments**
```python
# Define segments to analyze
segments = {
    'occupation': ['professional', 'student', 'self_employed'],
    'country': ['USA', 'UK', 'Canada'],
    'has_discount': [True, False]  # Derived from discount_offered > 0
}
```

**Step 2: Compute Segment Statistics**
```python
def analyze_all_segments(
    model,
    X: pd.DataFrame,
    segment_columns: List[str],
    categorical_features: List[str] = None,
    sample_size: int = 2000
) -> Dict:
    """
    Analyze model behavior across multiple segmentations.

    Returns comprehensive statistics:
    - Segment sizes
    - Average predictions by segment
    - Feature importance by segment
    - Segment-specific patterns
    - Key differentiators between segments
    """
```

**Step 3: Visualize Differences**
```python
# Heatmap of feature importance by segment
plot_segment_heatmap(results, top_n=10)

# Side-by-side comparison
plot_segment_comparison(results, segments=['professional', 'student'])

# Interaction plots
plot_segment_interactions(results)
```

### 7.2 Key Insights to Extract

**Pattern 1: Segment-Specific Drivers**
```
Question: "What drives conversions for students vs professionals?"

Answer:
Students:
- discount_offered:     0.28 (1st most important)
- pages_viewed:         0.22 (2nd)
- referral_source:      0.15 (3rd)

Professionals:
- pages_viewed:         0.35 (1st most important)
- previous_courses:     0.25 (2nd)
- discount_offered:     0.09 (5th)

INSIGHT: Students are 3x more price-sensitive than professionals
```

**Pattern 2: Feature Interactions by Segment**
```
Question: "Does discount effectiveness vary by segment?"

Answer:
Discount Impact by Occupation:
- Students:       +0.28 per 10% discount
- Self-employed:  +0.15 per 10% discount
- Professionals:  +0.09 per 10% discount
- Retired:        +0.05 per 10% discount

INSIGHT: Target discounts to students for maximum ROI
```

**Pattern 3: Segment Size vs Impact**
```
Segment           Size    Conversion Rate    Top Driver
──────────────────────────────────────────────────────────
professional      2,205   39.8%              pages_viewed
student           1,284   34.6%              discount
self_employed       727   37.1%              time_on_site
retired             523   33.5%              previous_courses

INSIGHT: Professionals are largest segment with highest conversion
```

---

## 8. Implementation Plan

### Phase 1: Core Foundation (Week 1-2)

**Deliverables**:
- [x] Project structure setup
- [ ] SHAP explainer with sampling
- [ ] Categorical aggregation logic
- [ ] Global feature importance plot
- [ ] Feature distribution plots
- [ ] Model performance summary
- [ ] Basic configuration

**Components**:
```
src/
├── core/interpreter.py
├── explainers/shap_explainer.py
├── explainers/categorical_handler.py
├── visualization/global_plots.py
├── utils/data_utils.py (sampling, validation)
└── utils/model_utils.py
```

**Test Notebook**: `notebooks/01_global_interpretation.ipynb`

**Success Criteria**:
- Can compute SHAP for customer_conversion.csv
- Can plot global importance with aggregated categoricals
- Can show model performance metrics
- Handles 5k+ dataset with sampling
- Warnings work (no errors thrown)

---

### Phase 2: Segment Analysis (Week 3)

**Deliverables**:
- [ ] Segment analyzer
- [ ] Segment comparison plots
- [ ] Heatmap visualizations
- [ ] Statistical comparison functions
- [ ] Segment-specific insights

**Components**:
```
src/
├── analysis/segment_analyzer.py
└── visualization/segment_plots.py
```

**Test Notebook**: `notebooks/02_segment_analysis.ipynb`

**Success Criteria**:
- Can compare occupation segments
- Can show feature importance by segment
- Can identify key differentiators
- Visualizations are clear and actionable

---

### Phase 3: Feature Dependencies (Week 4)

**Deliverables**:
- [ ] Feature dependence plots
- [ ] Interaction detection
- [ ] Pattern finder
- [ ] 2-way interaction visualizations

**Components**:
```
src/
├── analysis/interaction_detector.py
├── analysis/pattern_finder.py
└── visualization/dependence_plots.py
```

**Test Notebook**: `notebooks/04_feature_interactions.ipynb`

**Success Criteria**:
- Can plot feature vs SHAP value
- Can auto-detect strongest interactions
- Can visualize 2-way dependencies
- Discovers discount × occupation interaction

---

### Phase 4: Polish & Documentation (Week 5)

**Deliverables**:
- [ ] Configuration presets
- [ ] Helper functions
- [ ] Full workflow example
- [ ] Documentation
- [ ] Code cleanup

**Components**:
```
src/core/config.py (presets)
notebooks/05_full_workflow_example.ipynb
README.md (usage guide)
```

**Success Criteria**:
- Presets make common tasks 1-liners
- Full workflow notebook demonstrates all features
- Documentation is clear
- Code is well-commented

---

**Total Timeline: 5 weeks**

---

## 9. Usage Examples

### Example 1: Quick Global Analysis

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from classifier_model_interpreter import ModelInterpreter

# Load data
df = pd.read_csv('customer_conversion.csv')
X = df.drop(['customer_id', 'converted'], axis=1)
y = df['converted']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create interpreter with categorical specification
interpreter = ModelInterpreter(
    model=model,
    X=X,
    y=y,
    categorical_features=['country', 'occupation', 'referral_source', 'device_type']
)

# Global importance
fig = interpreter.plot_global_importance(top_n=10)
fig.show()

# With performance context
fig = interpreter.plot_global_importance(top_n=10, show_performance=True)
fig.show()
# → Shows: "Model Accuracy: 85%, AUC: 0.92"
```

---

### Example 2: Segment Comparison

```python
# Compare occupations
results = interpreter.compare_segments(
    segment_column='occupation',
    segments_to_compare=['professional', 'student', 'self_employed']
)

# Visualize
fig = interpreter.plot_segment_comparison(
    segment_results=results,
    comparison_type='importance'
)
fig.show()

# Heatmap view
fig = interpreter.plot_segment_heatmap(
    segment_results=results,
    top_n=10
)
fig.show()

# Get insights
insights = interpreter.get_segment_insights(results)
print(insights)
# Output:
# "Students are 3.1x more sensitive to discount than professionals"
# "Professionals prioritize engagement (pages_viewed) over price"
```

---

### Example 3: Feature Dependence with Interaction

```python
# Analyze discount effect
fig = interpreter.plot_feature_dependence(
    feature_name='discount_offered',
    interaction_feature='auto'  # Auto-detect strongest interaction
)
fig.show()

# Might show that occupation interacts with discount
# Color-coded by occupation showing different slopes

# Explicit interaction
fig = interpreter.plot_feature_dependence(
    feature_name='discount_offered',
    interaction_feature='occupation'
)
fig.show()
```

---

### Example 4: Feature Distribution Context

```python
# Understand what "52 pages viewed" means
fig = interpreter.plot_feature_distribution(
    feature_name='pages_viewed',
    highlight_value=52
)
fig.show()
# → Shows: "Your value (52) is at 90th percentile"

# Compare segment distributions
fig = interpreter.plot_feature_distribution_by_segment(
    feature_name='pages_viewed',
    segment_column='occupation'
)
fig.show()
# → Shows: Professionals average 18 pages, Students average 9
```

---

### Example 5: Configuration Presets

```python
# Quick summary (fast, high-level)
interpreter = ModelInterpreter.from_preset(
    'quick_summary',
    model=model,
    X=X,
    y=y
)
fig = interpreter.plot_global_importance()

# Detailed analysis (comprehensive)
interpreter = ModelInterpreter.from_preset(
    'detailed_analysis',
    model=model,
    X=X,
    y=y,
    categorical_features=['occupation', 'country']
)

# Custom configuration
from classifier_model_interpreter import Config

config = Config(
    sample_size=2000,
    categorical_features=['occupation', 'country', 'referral_source'],
    aggregation_method='sum',
    warn_on_issues=True,
    top_n_default=15
)

interpreter = ModelInterpreter(model, X, y, config=config)
```

---

## 10. Testing Strategy

### Philosophy: Notebook-Driven Testing

**Approach**: Create comprehensive Jupyter notebooks that demonstrate all functionality and serve as both tests and documentation.

### Test Notebooks

**Notebook 1: Global Interpretation** (`01_global_interpretation.ipynb`)
```markdown
# Global Model Interpretation

## Setup
- Load customer_conversion.csv
- Train RandomForestClassifier
- Create interpreter

## Tests
1. Global feature importance (bar chart)
   - Verify categoricals are aggregated
   - Check top 10 features make sense
   - Validate SHAP values sum correctly

2. Feature distributions
   - Plot pages_viewed distribution
   - Verify quantile markers
   - Check highlighting works

3. Model performance
   - Display confusion matrix
   - Show ROC curve
   - Validate AUC calculation

## Expected Results
- pages_viewed should be top feature
- Categorical features (occupation, referral_source) should be grouped
- Model accuracy should be ~85%
```

**Notebook 2: Segment Analysis** (`02_segment_analysis.ipynb`)
```markdown
# Segment-Level Analysis

## Tests
1. Occupation segment comparison
   - Compare professional vs student vs self_employed
   - Verify segment sizes are correct
   - Check feature importance differs by segment

2. Segment heatmap
   - Create heatmap of top 10 features × segments
   - Verify discount is more important for students

3. Country segment analysis
   - Compare USA vs UK vs Canada
   - Check for geographic patterns

## Expected Results
- Students should show higher discount sensitivity
- Professionals should prioritize engagement features
- Segment sizes should match data
```

**Notebook 3: Categorical Handling** (`03_categorical_handling.ipynb`)
```markdown
# Categorical Variable Handling

## Tests
1. One-hot detection
   - Create test data with one-hot encoded features
   - Verify auto-detection works
   - Check aggregation is correct

2. Manual specification
   - Specify categorical_features list
   - Verify aggregation matches manual

3. Category breakdown
   - Show occupation total: +0.09
   - Break down: professional +0.06, student +0.02, etc.

## Expected Results
- SHAP values for dummy variables should sum to category total
- No dummy variables should appear in plots
- Category labels should be clear
```

**Notebook 4: Feature Interactions** (`04_feature_interactions.ipynb`)
```markdown
# Feature Dependencies and Interactions

## Tests
1. Discount dependence
   - Plot discount vs SHAP
   - Check for non-linear effects

2. Auto-interaction detection
   - Run with interaction_feature='auto'
   - Verify occupation is detected as strongest interaction

3. Segment-specific interactions
   - Show discount effect by occupation
   - Verify students have steeper slope

## Expected Results
- Discount should show positive SHAP relationship
- Occupation should be detected as interacting feature
- Interaction plot should show different slopes by segment
```

**Notebook 5: Full Workflow** (`05_full_workflow_example.ipynb`)
```markdown
# Complete Analysis Workflow

## Scenario
Analyze customer_conversion.csv to understand:
1. What drives conversion overall?
2. How do different customer segments behave?
3. What features interact?
4. What are actionable insights?

## Workflow
1. Load data and train model
2. Global interpretation
3. Segment analysis (occupation, country)
4. Feature interactions
5. Generate insights

## Expected Insights
- Engagement is primary driver (pages, time, videos)
- Students are price-sensitive, professionals value content
- Discount effectiveness varies by occupation
- Referrals convert better than paid ads
```

---

### Validation Checklist

For each notebook, verify:

**Functionality**:
- [ ] All code cells run without errors
- [ ] Plots render correctly
- [ ] Results are interpretable
- [ ] Warnings appear where expected (not errors)

**Data Quality**:
- [ ] SHAP values are reasonable (no extreme outliers)
- [ ] Categorical aggregation is correct
- [ ] Sample sizes are appropriate
- [ ] Segment comparisons make sense

**Visualizations**:
- [ ] Labels are clear
- [ ] Colors are meaningful
- [ ] Titles are descriptive
- [ ] Legends are helpful

**Insights**:
- [ ] Findings align with known patterns in data
- [ ] Segment differences are explainable
- [ ] Interactions make logical sense
- [ ] Performance metrics are accurate

---

### User Verification

**Process**:
1. User runs each notebook sequentially
2. Reviews outputs visually
3. Validates insights against domain knowledge
4. Provides feedback on unclear results
5. Iterate on visualizations and explanations

**No Unit Tests**: Focus on end-to-end functionality through notebooks, not programmatic tests.

---

## Appendix A: Configuration Reference

### Default Configuration

```python
@dataclass
class Config:
    """Global configuration for interpreter."""

    # SHAP settings
    sample_size: int = 2000           # Max samples for SHAP
    shap_algorithm: str = 'auto'      # 'auto', 'tree', 'kernel'

    # Categorical handling
    categorical_features: List[str] = None
    auto_detect_categorical: bool = True
    aggregation_method: str = 'sum'   # 'sum', 'mean'

    # Visualization
    top_n_default: int = 10
    plot_backend: str = 'plotly'      # 'plotly', 'matplotlib'
    figure_size: Tuple[int, int] = (10, 6)

    # Validation
    warn_on_issues: bool = True       # Warn, don't error
    min_samples_per_segment: int = 30

    # Performance
    n_jobs: int = -1                  # Parallel processing
    verbose: bool = False
```

### Preset Configurations

```python
PRESETS = {
    'quick_summary': Config(
        sample_size=1000,
        top_n_default=5,
        shap_algorithm='tree'
    ),

    'detailed_analysis': Config(
        sample_size=2000,
        top_n_default=15,
        auto_detect_categorical=True
    ),

    'segment_focus': Config(
        sample_size=500,  # Per segment
        min_samples_per_segment=50,
        top_n_default=10
    )
}
```

---

## Appendix B: API Quick Reference

### Main Class

```python
class ModelInterpreter:
    """Main interpretation interface."""

    def __init__(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray = None,
        categorical_features: List[str] = None,
        config: Config = None
    ):
        """Initialize interpreter."""

    # Global analysis
    def plot_global_importance(self, top_n=10, show_performance=True) -> Figure
    def plot_feature_distribution(self, feature_name, highlight_value=None) -> Figure
    def plot_performance_summary(self) -> Figure

    # Segment analysis
    def compare_segments(self, segment_column, segments=None) -> Dict
    def plot_segment_comparison(self, results, feature=None) -> Figure
    def plot_segment_heatmap(self, results, top_n=10) -> Figure

    # Feature analysis
    def plot_feature_dependence(self, feature, interaction='auto') -> Figure
    def detect_interactions(self, feature, top_k=3) -> List[str]

    # Insights
    def get_segment_insights(self, segment_results) -> str
    def summarize_model(self) -> str

    @classmethod
    def from_preset(cls, preset_name, model, X, y=None) -> 'ModelInterpreter'
```

---

**END OF SPECIFICATION v2.0**

---

## Summary of Changes from v1.0

### Added ✅
- Feature distribution plots with context
- Prediction confidence indicators
- Model performance integration
- Enhanced segment analysis
- Categorical aggregation emphasis
- Configuration presets
- Notebook-driven testing

### Removed ❌
- Dashboard components
- HTML/PDF report generation
- Extensive validation and error handling
- LIME integration
- API/production features
- Instance-level deep dives
- Complex class hierarchies

### Simplified 🎯
- Focus on global over local
- Segments over instances
- Warnings over errors
- Functions over classes
- Notebooks over unit tests

**Result**: Leaner, more focused specification aligned with exploratory analysis goals.
