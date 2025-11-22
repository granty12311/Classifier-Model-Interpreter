# Classifier Model Interpreter - Technical Specification

**Version**: 1.0
**Date**: 2025-11-22
**Status**: Draft for Review

---

## Executive Summary

This specification defines a **Model Interpretation Package** that translates complex machine learning predictions into clear, actionable business insights. The system focuses on three core capabilities:

1. **Visual Clarity**: Intuitive visualizations showing how features drive predictions
2. **Categorical Handling**: Proper treatment of categorical variables in explanations
3. **Business Translation**: Converting SHAP outputs into actionable recommendations

**Target Use Case**: Customer conversion models, credit risk, churn prediction, and similar classification problems.

---

## Table of Contents

1. [Project Goals](#1-project-goals)
2. [Core Architecture](#2-core-architecture)
3. [Package Dependencies](#3-package-dependencies)
4. [Module Structure](#4-module-structure)
5. [Key Features](#5-key-features)
6. [Visualization Components](#6-visualization-components)
7. [Business Translation Layer](#7-business-translation-layer)
8. [Implementation Phases](#8-implementation-phases)
9. [Usage Examples](#9-usage-examples)
10. [Testing Strategy](#10-testing-strategy)
11. [Open Questions](#11-open-questions)

---

## 1. Project Goals

### Primary Objectives

**Goal 1: Clear Feature Impact Visualization**
- Show how each feature drives predictions (positive/negative direction)
- Display magnitude of effects (quantified impact)
- Handle both continuous and categorical variables seamlessly
- Support global (model-wide) and local (single prediction) explanations

**Goal 2: Categorical Variable Excellence**
- Automatically detect categorical features
- Group one-hot encoded variables back to original categories
- Visualize categories with meaningful labels (not dummy indicators)
- Show category-level importance and effects

**Goal 3: Actionable Business Insights**
- Translate SHAP values into plain English recommendations
- Generate "what-if" scenarios (e.g., "If we lower price by 10%, conversion increases by 5%")
- Identify levers (controllable features) vs. constraints (non-controllable features)
- Create executive summaries and detailed technical reports

### Success Criteria

- [ ] Non-technical stakeholders can understand model decisions without training
- [ ] Visualizations load in <5 seconds for datasets with 5,000 rows
- [ ] Categorical variables display as original categories, not dummy variables
- [ ] Business narratives accurately reflect SHAP value interpretations
- [ ] System works with sklearn, XGBoost, LightGBM, CatBoost models

### Non-Goals (Out of Scope)

- Deep learning / neural network interpretation (focus on tree-based and linear models)
- Real-time streaming explanations
- Automated retraining based on insights
- Causal inference (correlation ≠ causation)
- Multi-model ensemble interpretation

---

## 2. Core Architecture

### Design Principles

1. **Separation of Concerns**: Computation ↔ Visualization ↔ Business Logic
2. **Model Agnostic**: Works with any sklearn-compatible classifier
3. **Extensibility**: Easy to add new interpretation methods or visualizations
4. **Performance**: Cache expensive computations (SHAP values)
5. **Configurability**: Behavior driven by simple configuration objects

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  - Jupyter Notebooks                                         │
│  - Python Scripts                                            │
│  - (Future: Web Dashboard)                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│               Model Interpreter (Main API)                   │
│  - interpret_global()                                        │
│  - interpret_local()                                         │
│  - generate_report()                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
┌──────▼────┐   ┌──────▼────┐   ┌─────▼──────┐
│ Explainers │   │Visualizers│   │  Business  │
│  - SHAP    │   │  - Plotly │   │ Translator │
│  - LIME    │   │  - Mpl    │   │  - Narrative│
│  - Perm.   │   │  - Custom │   │  - Insights│
└────────────┘   └───────────┘   └────────────┘
       │               │               │
       └───────────────┼───────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Utility Layer                              │
│  - Data Processor (encoding, normalization)                 │
│  - Model Wrapper (unified interface)                        │
│  - Cache Manager (SHAP value storage)                       │
│  - Config Manager (settings)                                │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Input: Model + Training Data + Test Instance
  ↓
Data Processor: Encode categoricals, create baseline
  ↓
Explainer: Compute SHAP values (cached)
  ↓
Visualizer: Create plots (force, waterfall, summary)
  ↓
Business Translator: Generate narrative
  ↓
Output: Visualizations + Text Report + Recommendations
```

---

## 3. Package Dependencies

### Core Dependencies

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **shap** | >=0.42.0 | Primary interpretation engine | Industry standard, mathematically rigorous, comprehensive |
| **scikit-learn** | >=1.3.0 | Model interface, preprocessing | Standard ML library |
| **pandas** | >=2.0.0 | Data manipulation | DataFrame operations |
| **numpy** | >=1.24.0 | Numerical operations | Array computations |

### Visualization Dependencies

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **matplotlib** | >=3.7.0 | Base plotting | Foundation for many SHAP plots |
| **plotly** | >=5.14.0 | Interactive visualizations | Better for dashboards, business users |
| **seaborn** | >=0.12.0 | Statistical plots | Enhanced matplotlib aesthetics |

### Optional Dependencies

| Package | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **lime** | >=0.2.0 | Alternative explanations | Fast local explanations, validation |
| **xgboost** | >=1.7.0 | Model support | If using XGBoost models |
| **lightgbm** | >=4.0.0 | Model support | If using LightGBM models |
| **catboost** | >=1.2.0 | Model support | If using CatBoost models |

### Why SHAP as Primary Engine?

**Advantages:**
- ✅ Mathematically rigorous (Shapley values from game theory)
- ✅ Model-agnostic (works with any model)
- ✅ Handles interactions well
- ✅ Industry standard (regulatory acceptance)
- ✅ Rich visualization library
- ✅ TreeExplainer is extremely fast for tree models

**Trade-offs:**
- ⚠️ Can be computationally expensive (mitigated by caching)
- ⚠️ Requires baseline selection (mitigated by smart defaults)
- ⚠️ Learning curve for theory (mitigated by business translation layer)

**Decision**: Use SHAP as primary, with option to add LIME later for speed comparisons.

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
│   │   ├── base_explainer.py       # Abstract base for explainers
│   │   └── config.py               # Configuration dataclasses
│   │
│   ├── explainers/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py       # SHAP implementation
│   │   ├── feature_importance.py   # Global importance
│   │   └── interaction_analyzer.py # Feature interactions
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── base_plotter.py         # Abstract plotter
│   │   ├── shap_plots.py           # SHAP-specific plots
│   │   ├── plotly_dashboard.py     # Interactive dashboard
│   │   └── report_builder.py       # HTML/PDF report generation
│   │
│   ├── business/
│   │   ├── __init__.py
│   │   ├── narrative_generator.py  # Text explanations
│   │   ├── insight_extractor.py    # Actionable insights
│   │   └── recommendation_engine.py # What-if scenarios
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_processor.py       # Encoding, normalization
│   │   ├── categorical_handler.py  # Category grouping
│   │   ├── model_wrapper.py        # Unified model interface
│   │   ├── cache_manager.py        # SHAP value caching
│   │   └── validators.py           # Input validation
│   │
│   └── templates/
│       ├── report_template.html    # HTML report template
│       └── narrative_templates.py  # Text templates
│
├── notebooks/
│   ├── 01_basic_usage.ipynb
│   ├── 02_categorical_handling.ipynb
│   ├── 03_business_insights.ipynb
│   └── 04_full_workflow.ipynb
│
├── tests/
│   ├── test_explainers.py
│   ├── test_visualization.py
│   ├── test_business_translation.py
│   └── test_integration.py
│
├── data/
│   ├── customer_conversion.csv     # Test dataset
│   └── customer_conversion_README.md
│
├── requirements.txt
├── SPECIFICATION.md               # This file
└── README.md
```

---

## 5. Key Features

### Feature 1: Global Feature Importance

**Description**: Show which features matter most across all predictions.

**Methods**:
- Mean absolute SHAP values (primary)
- Permutation importance (validation)
- SHAP summary plot (beeswarm)

**Output**:
- Bar chart of top N features
- Beeswarm plot showing distribution of effects
- Table with numerical values

**Configuration**:
```python
config = GlobalImportanceConfig(
    top_n=10,                    # Show top 10 features
    plot_type='bar',             # 'bar' or 'beeswarm'
    categorical_grouping=True    # Group dummy variables
)
```

---

### Feature 2: Local Prediction Explanation

**Description**: Explain why a specific prediction was made.

**Methods**:
- SHAP force plot
- SHAP waterfall plot
- LIME (optional, for validation)

**Output**:
- Interactive force plot
- Waterfall chart showing feature stacking
- Text narrative explaining prediction

**Configuration**:
```python
config = LocalExplanationConfig(
    instance_idx=0,              # Which prediction to explain
    plot_types=['force', 'waterfall'],
    show_feature_values=True,    # Display actual values
    baseline_comparison=True     # Compare to average prediction
)
```

---

### Feature 3: Feature Dependence Analysis

**Description**: Show how a feature affects predictions across its range.

**Methods**:
- SHAP dependence plots
- Partial dependence plots
- ICE (Individual Conditional Expectation) plots

**Output**:
- Scatter plot: feature value vs SHAP value
- Optional color dimension for interactions
- Smoothed trend line

**Configuration**:
```python
config = DependenceConfig(
    feature='age',
    interaction_feature='income',  # Auto-detect if None
    plot_type='scatter',           # 'scatter' or 'violin'
    show_distribution=True         # Show marginal histogram
)
```

---

### Feature 4: Categorical Variable Handling

**Description**: Properly handle categorical features in all visualizations.

**Approach**:
1. **Detection**: Automatically identify categorical features
   - By dtype (object, category)
   - By one-hot encoding pattern (feature_A, feature_B, feature_C)
   - By user specification

2. **Grouping**: Aggregate SHAP values for dummy variables
   ```python
   # Original dummy variables
   color_red: +0.05
   color_blue: +0.03
   color_green: +0.00

   # Grouped representation
   color: +0.08 (primarily driven by red and blue)
   ```

3. **Visualization**: Show categories, not dummy indicators
   ```
   Bar chart:
   color_red    [████] 0.05     ❌ Bad
   color        [████] 0.08     ✅ Good

   With breakdown:
   color        [████████] 0.08
     └─ red     [█████]    0.05
     └─ blue    [███]      0.03
     └─ green   []         0.00
   ```

**Configuration**:
```python
config = CategoricalConfig(
    auto_detect=True,
    categorical_features=['country', 'occupation'],
    one_hot_groups={
        'color': ['color_red', 'color_blue', 'color_green']
    },
    aggregation='sum'  # 'sum', 'mean', 'max'
)
```

---

### Feature 5: Business Narrative Generation

**Description**: Convert SHAP outputs into plain English recommendations.

**Approach**:
1. **Template-based**: Predefined sentence structures
2. **Context-aware**: Different narratives for different domains
3. **Actionable**: Focus on controllable features

**Example Output**:
```
PREDICTION SUMMARY
==================
Customer #12345 has a 72% probability of converting (vs. 36% average).

TOP DRIVERS OF CONVERSION:
1. High Engagement (+28 percentage points)
   - Viewed 45 pages (vs. 12 average)
   - Watched 8 videos (vs. 3 average)
   → INSIGHT: Customer is highly engaged with content

2. Discount Offered (+12 percentage points)
   - 30% discount vs. 0% for average customer
   → RECOMMENDATION: Discount is working - customer is price-sensitive

3. Professional Occupation (+8 percentage points)
   - Professionals convert at 40% vs. 36% overall
   → INSIGHT: Aligns with target customer profile

MITIGATING FACTORS:
1. Recent Signup (-5 percentage points)
   - Only 3 days since signup vs. 10 average
   → RECOMMENDATION: Follow up in 1 week to increase conversion

RECOMMENDATION: HIGH likelihood of conversion. Consider:
- Send reminder email highlighting popular courses
- Extend trial by 3 days to allow more exploration time
- Emphasize professional development benefits in messaging
```

**Configuration**:
```python
config = NarrativeConfig(
    domain='customer_conversion',  # 'credit_risk', 'churn', etc.
    detail_level='summary',        # 'summary', 'detailed', 'technical'
    include_recommendations=True,
    focus_on_controllable=True,    # Emphasize actionable features
    top_n_features=3
)
```

---

### Feature 6: Interactive Dashboard

**Description**: Plotly-based dashboard for exploration.

**Components**:
- Global feature importance (interactive bar chart)
- Instance selector (dropdown)
- Local explanation (force plot + waterfall)
- Dependence plot (feature selector)
- Narrative panel (auto-generated text)

**Technology**: Plotly Dash or standalone HTML with plotly.js

**Configuration**:
```python
config = DashboardConfig(
    port=8050,
    show_feature_values=True,
    enable_what_if=True,      # Allow feature modification
    cache_explanations=True
)
```

---

## 6. Visualization Components

### 6.1 Global Importance - Bar Chart

**When to Use**: Show overall feature importance

**Implementation**:
```python
def plot_global_importance(
    shap_values: np.ndarray,
    feature_names: list,
    top_n: int = 10,
    categorical_groups: dict = None
) -> plotly.graph_objects.Figure:
    """
    Create bar chart of mean absolute SHAP values.

    Parameters:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: List of feature names
        top_n: Number of top features to show
        categorical_groups: Dict mapping category name to dummy vars

    Returns:
        Plotly Figure object
    """
```

**Visual Specifications**:
- Horizontal bars, sorted by importance (descending)
- Color: Single color for consistency
- Annotations: Numerical values on bars
- Title: "Top {n} Most Important Features"
- X-axis: "Mean |SHAP Value| (impact on prediction)"

---

### 6.2 Local Explanation - Waterfall Plot

**When to Use**: Explain individual predictions

**Implementation**:
```python
def plot_waterfall(
    base_value: float,
    shap_values: np.ndarray,
    feature_values: dict,
    feature_names: list,
    prediction: float
) -> plotly.graph_objects.Figure:
    """
    Create waterfall chart showing feature contributions.

    Parameters:
        base_value: Model's average prediction
        shap_values: SHAP values for this instance
        feature_values: Dict of feature_name: value
        feature_names: Ordered list of features to show
        prediction: Actual model prediction

    Returns:
        Plotly Figure object
    """
```

**Visual Specifications**:
- Start: Base value (average prediction)
- Bars: Features sorted by absolute SHAP value
- Colors: Red (positive) / Blue (negative)
- End: Actual prediction
- Annotations: Feature name + value + SHAP contribution

**Example**:
```
Base (36%) ─┬→ Engagement: +28% ─┬→ Discount: +12% ─┬→ ... ─┬→ Prediction (72%)
            │                    │                   │       │
         [Start]              [+28pp]             [+12pp]  [Final]
```

---

### 6.3 Dependence Plot - Scatter with Interaction

**When to Use**: Show how a feature affects predictions

**Implementation**:
```python
def plot_dependence(
    feature_name: str,
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    interaction_feature: str = None,
    interaction_values: np.ndarray = None
) -> plotly.graph_objects.Figure:
    """
    Create scatter plot of feature vs SHAP value.

    Parameters:
        feature_name: Name of primary feature
        feature_values: Values of primary feature
        shap_values: SHAP values for primary feature
        interaction_feature: Optional interacting feature name
        interaction_values: Optional interacting feature values

    Returns:
        Plotly Figure object
    """
```

**Visual Specifications**:
- X-axis: Feature value
- Y-axis: SHAP value (impact on prediction)
- Color: Interaction feature (if provided)
- Points: Semi-transparent for overlapping visibility
- Trend: Smoothed line (LOWESS)
- Zero line: Horizontal line at y=0

---

### 6.4 Summary Plot - Beeswarm

**When to Use**: Show feature importance + distribution of effects

**Implementation**: Use `shap.summary_plot()` with custom styling

**Visual Specifications**:
- Y-axis: Features (sorted by importance)
- X-axis: SHAP value
- Color: Feature value (low=blue, high=red)
- Points: One per sample, jittered vertically
- Interpretation: Shows not just importance, but directionality and spread

---

### 6.5 Force Plot - Interactive

**When to Use**: Interactive exploration of single prediction

**Implementation**: Use `shap.force_plot()` converted to Plotly

**Visual Specifications**:
- Base value at center
- Positive contributions push right (red)
- Negative contributions push left (blue)
- Interactive hover for feature details
- Can stack multiple instances for comparison

---

### 6.6 Categorical Breakdown - Grouped Bar

**When to Use**: Show contribution of each category

**Implementation**:
```python
def plot_categorical_breakdown(
    category_name: str,
    category_values: list,
    shap_values: dict,
    overall_shap: float
) -> plotly.graph_objects.Figure:
    """
    Show SHAP contribution broken down by category.

    Parameters:
        category_name: Name of categorical feature
        category_values: List of possible categories
        shap_values: Dict of category: SHAP value
        overall_shap: Total SHAP for this categorical feature

    Returns:
        Plotly Figure object
    """
```

**Visual Example**:
```
Occupation: +8pp total

professional  [████████] +6pp
student       [███]      +3pp
self_employed [█]        +1pp
retired       []         +0pp
unemployed    [▬▬]       -2pp
```

---

## 7. Business Translation Layer

### 7.1 Narrative Templates

**Structure**: Domain-specific templates with placeholders

**Example Template** (Customer Conversion):
```python
TEMPLATE = """
{PREDICTION_SUMMARY}

TOP DRIVERS OF {OUTCOME}:
{POSITIVE_DRIVERS}

{NEGATIVE_DRIVERS_SECTION}

RECOMMENDATIONS:
{RECOMMENDATIONS}
"""

POSITIVE_DRIVER_TEMPLATE = """
{rank}. {feature_name} (+{impact}pp)
   - {feature_description}
   → {insight}
"""
```

**Feature Descriptions** (Auto-generated):
```python
def describe_feature(feature_name, feature_value, percentile):
    """Generate natural language description of feature value."""
    if percentile > 75:
        return f"{feature_name} is {feature_value} (top 25%)"
    elif percentile > 50:
        return f"{feature_name} is {feature_value} (above average)"
    elif percentile > 25:
        return f"{feature_name} is {feature_value} (below average)"
    else:
        return f"{feature_name} is {feature_value} (bottom 25%)"
```

---

### 7.2 Insight Extraction Rules

**Pattern Matching**:
1. **High Impact + Controllable** → Primary recommendation
2. **High Impact + Non-controllable** → Context/explanation
3. **Interaction Effect Detected** → Highlight synergy
4. **Unexpected Pattern** → Flag for investigation

**Example Rules**:
```python
INSIGHT_RULES = {
    'high_engagement': {
        'pattern': lambda f: f.startswith('pages_viewed') or f.startswith('time_on'),
        'positive': "Customer is highly engaged → Strong conversion signal",
        'negative': "Low engagement → Risk of not converting"
    },
    'price_sensitivity': {
        'pattern': lambda f: 'discount' in f or 'price' in f,
        'positive': "Price discount is driving conversion → Price-sensitive customer",
        'negative': "No discount hurting conversion → Consider offering promotion"
    }
}
```

---

### 7.3 Recommendation Engine

**What-If Scenarios**:
```python
def generate_what_if(
    instance: pd.Series,
    feature_to_change: str,
    new_value: Any,
    model,
    explainer
) -> dict:
    """
    Calculate impact of changing feature value.

    Returns:
        {
            'original_prediction': 0.72,
            'new_prediction': 0.85,
            'delta': +0.13,
            'narrative': "If discount increased from 0% to 30%,
                         conversion probability increases by 13pp"
        }
    """
```

**Actionable Insights**:
```python
def extract_actionable_insights(
    shap_explanation,
    controllable_features: list
) -> list:
    """
    Extract top recommendations focusing on controllable features.

    Returns:
        [
            {
                'feature': 'discount_offered',
                'current_value': 0,
                'recommended_value': 20,
                'expected_impact': '+10pp conversion',
                'rationale': 'Customer is price-sensitive based on...'
            }
        ]
    """
```

---

## 8. Implementation Phases

### Phase 1: Foundation (MVP) - 2-3 weeks

**Deliverables**:
- [ ] Core architecture setup
- [ ] SHAP integration (TreeExplainer)
- [ ] Basic visualizations (bar chart, waterfall)
- [ ] Categorical variable grouping
- [ ] Simple narrative generation
- [ ] Test on customer_conversion.csv dataset

**Success Criteria**:
- Can explain a single prediction with waterfall chart
- Can show global feature importance
- Categorical variables display correctly
- Basic text summary generated

**Components**:
```
src/
├── core/interpreter.py          ✓ Main API
├── explainers/shap_explainer.py ✓ SHAP implementation
├── visualization/shap_plots.py   ✓ Waterfall + bar charts
├── utils/categorical_handler.py  ✓ Category grouping
├── business/narrative_generator.py ✓ Basic templates
```

---

### Phase 2: Enhanced Visualizations - 1-2 weeks

**Deliverables**:
- [ ] Dependence plots with interactions
- [ ] Beeswarm summary plots
- [ ] Force plots (interactive)
- [ ] Plotly-based interactive dashboard
- [ ] Multiple instance comparison

**Success Criteria**:
- All major SHAP plot types supported
- Interactive exploration works
- Dashboard loads in <5 seconds

**Components**:
```
src/
├── visualization/plotly_dashboard.py ✓ Interactive dashboard
├── explainers/interaction_analyzer.py ✓ Feature interactions
```

---

### Phase 3: Business Intelligence - 1-2 weeks

**Deliverables**:
- [ ] Advanced narrative generation
- [ ] What-if scenario calculator
- [ ] Actionable recommendations
- [ ] HTML/PDF report generation
- [ ] Domain-specific templates (credit, conversion, churn)

**Success Criteria**:
- Narratives read naturally to non-technical users
- What-if scenarios are accurate
- Reports are stakeholder-ready

**Components**:
```
src/
├── business/insight_extractor.py      ✓ Insight extraction
├── business/recommendation_engine.py  ✓ What-if scenarios
├── visualization/report_builder.py    ✓ Report generation
├── templates/                         ✓ Domain templates
```

---

### Phase 4: Optimization & Extension - 1 week

**Deliverables**:
- [ ] SHAP value caching
- [ ] Performance benchmarks
- [ ] LIME integration (optional)
- [ ] Batch processing support
- [ ] Model comparison tools

**Success Criteria**:
- 10x speedup from caching on repeated calls
- Can process 1,000 explanations in <1 minute
- Documentation complete

**Components**:
```
src/
├── utils/cache_manager.py        ✓ Caching layer
├── explainers/lime_explainer.py  ✓ LIME (optional)
```

---

## 9. Usage Examples

### Example 1: Basic Global Interpretation

```python
from classifier_model_interpreter import ModelInterpreter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('customer_conversion.csv')
X = df.drop(['customer_id', 'converted'], axis=1)
y = df['converted']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create interpreter
interpreter = ModelInterpreter(
    model=model,
    data=X,
    categorical_features=['country', 'occupation', 'device_type', 'referral_source']
)

# Global feature importance
fig = interpreter.plot_global_importance(top_n=10)
fig.show()

# Get narrative
narrative = interpreter.generate_global_narrative()
print(narrative)
```

**Expected Output**:
```
GLOBAL FEATURE IMPORTANCE
=========================

The model relies most heavily on these features when predicting conversion:

1. pages_viewed (Importance: 0.42)
   - Most important driver of predictions
   - Higher page views strongly associated with conversion

2. time_on_site_mins (Importance: 0.35)
   - Second most important
   - More time on site indicates higher engagement

3. discount_offered (Importance: 0.28)
   - Strong impact on conversion decisions
   - Discounts significantly increase conversion probability

[... continues ...]
```

---

### Example 2: Local Prediction Explanation

```python
# Explain specific prediction
instance_idx = 42
fig = interpreter.plot_local_explanation(
    instance_idx=instance_idx,
    plot_type='waterfall'
)
fig.show()

# Get detailed narrative
narrative = interpreter.generate_local_narrative(
    instance_idx=instance_idx,
    detail_level='detailed'
)
print(narrative)
```

**Expected Output**:
```
PREDICTION EXPLANATION - Customer #42
======================================

PREDICTION: 85% probability of converting (HIGH)
BASELINE: 36% average conversion rate
DIFFERENCE: +49 percentage points above average

WHY IS THIS PREDICTION SO HIGH?

TOP 3 DRIVERS:
1. Engagement Level (+35pp)
   - Viewed 52 pages (vs. 12 average) → +20pp
   - Spent 95 minutes on site (vs. 24 average) → +15pp
   This customer is HIGHLY engaged with the content.

2. Discount Offered (+18pp)
   - Received 30% discount (vs. 0% typical)
   This customer is benefiting from our promotion.

3. Previous Courses (+8pp)
   - Has completed 2 prior courses (vs. 0 typical)
   This customer has a history with our platform.

RECOMMENDATION: VERY HIGH conversion likelihood. This customer is:
- Highly engaged (top 5% in page views)
- Responding to discount promotion
- A repeat customer

ACTION: Send personalized course recommendations NOW to capitalize on high engagement.
```

---

### Example 3: Categorical Variable Analysis

```python
# Analyze categorical feature
fig = interpreter.plot_categorical_importance(
    feature='occupation',
    plot_type='grouped_bar'
)
fig.show()

# Get category-specific insights
insights = interpreter.analyze_categorical_feature('occupation')
print(insights)
```

**Expected Output**:
```
OCCUPATION ANALYSIS
===================

Overall Impact: +6pp on conversion (positive driver)

Breakdown by Category:
- professional:   +8pp (highest impact, 40% conversion rate)
- self_employed:  +4pp (above average, 37% conversion)
- student:        +2pp (slightly above, 35% conversion)
- retired:        +1pp (near average, 34% conversion)
- unemployed:     -5pp (lowest, 27% conversion)

INSIGHT: Professionals are the strongest converters. Consider:
- Tailoring messaging to professional development
- Highlighting career advancement benefits
- Offering corporate/team packages
```

---

### Example 4: What-If Scenario

```python
# Run what-if analysis
scenario = interpreter.what_if(
    instance_idx=42,
    changes={
        'discount_offered': 20,  # Change from 30% to 20%
        'pages_viewed': 40       # Change from 52 to 40
    }
)

print(scenario['narrative'])
```

**Expected Output**:
```
WHAT-IF SCENARIO ANALYSIS
=========================

ORIGINAL PREDICTION: 85% conversion probability

PROPOSED CHANGES:
- Reduce discount from 30% to 20% (-10pp)
- Reduce pages viewed from 52 to 40 (-12 pages)

NEW PREDICTION: 68% conversion probability

DELTA: -17 percentage points

INTERPRETATION:
- Reducing discount by 10pp decreases conversion by ~9pp
- Reducing engagement decreases conversion by ~8pp
- Combined effect is -17pp (slightly less than additive due to interactions)

BUSINESS DECISION:
If considering reducing discount to improve margins, expect ~9pp drop
in conversion rate. At current engagement level (52 pages), customer
would still likely convert even with lower discount.
```

---

### Example 5: Batch Report Generation

```python
# Generate reports for top 100 high-value customers
high_value_customers = [10, 25, 42, 67, 89, ...]  # Instance indices

report = interpreter.generate_batch_report(
    instance_indices=high_value_customers,
    output_format='html',
    include_what_if=True,
    output_path='reports/high_value_customers_analysis.html'
)

print(f"Report saved to: {report['path']}")
```

---

## 10. Testing Strategy

### Unit Tests

**Coverage Targets**: >80% for all modules

**Key Test Cases**:
```python
# test_explainers.py
def test_shap_explainer_returns_correct_shape():
    """SHAP values should match (n_samples, n_features)"""

def test_categorical_grouping():
    """Dummy variables should be grouped correctly"""

def test_baseline_calculation():
    """Baseline should equal mean prediction"""

# test_visualization.py
def test_waterfall_plot_reconciles():
    """Waterfall should sum to actual prediction"""

def test_force_plot_rendering():
    """Force plot should render without errors"""

# test_business_translation.py
def test_narrative_generation():
    """Narrative should include all required sections"""

def test_recommendation_accuracy():
    """Recommendations should be based on SHAP values"""
```

---

### Integration Tests

**Test Scenarios**:
1. **End-to-end workflow**: Load data → Train model → Generate explanations → Create report
2. **Multiple model types**: Test with RF, XGBoost, LightGBM, LogisticRegression
3. **Different data types**: Numerical only, categorical only, mixed
4. **Edge cases**: Single instance, single feature, perfectly balanced classes

---

### Validation Tests

**Domain Validation**:
```python
def test_shap_values_sum_to_prediction():
    """SHAP values must satisfy additivity property"""

def test_categorical_shap_matches_dummy_sum():
    """Grouped categorical SHAP should equal sum of dummies"""

def test_what_if_predictions_are_consistent():
    """What-if predictions should match model.predict()"""
```

---

### Performance Benchmarks

**Targets**:
- Global importance: <2 seconds for 5,000 samples
- Local explanation: <1 second per instance
- Batch explanations (100 instances): <30 seconds
- Dashboard load: <5 seconds

**Measurement**:
```python
import time

def benchmark_global_importance():
    start = time.time()
    interpreter.plot_global_importance()
    elapsed = time.time() - start
    assert elapsed < 2.0, f"Too slow: {elapsed:.2f}s"
```

---

## 11. Open Questions

### Questions for Review

**Q1: Model Support Scope**
- Should we support deep learning models (e.g., Keras, PyTorch)?
- **Recommendation**: Start with tree-based + linear, add DL in Phase 4 if needed

**Q2: Interaction Analysis Depth**
- How deep should we go with interaction effects?
- **Options**:
  - Basic: Show top interacting feature for each main feature
  - Advanced: Full interaction matrix (computationally expensive)
- **Recommendation**: Start with basic, add advanced as optional

**Q3: Categorical Handling - Encoding Strategy**
- Should we require one-hot encoding upfront, or handle internally?
- **Recommendation**: Accept both, auto-detect encoding style

**Q4: Business Narratives - Customization**
- Should users be able to customize narrative templates?
- **Recommendation**: Yes - provide default templates + allow custom templates

**Q5: Caching Strategy**
- Where to store cached SHAP values (memory vs. disk)?
- **Options**:
  - Memory only (fast, but lost on restart)
  - Disk cache (persistent, but slower)
  - Both (memory with disk fallback)
- **Recommendation**: Both - memory cache with optional disk persistence

**Q6: What-If Scenario Constraints**
- Should we validate feasibility of what-if changes?
  - E.g., prevent setting age=200 or discount=150%
- **Recommendation**: Yes - add soft warnings for extreme values

**Q7: Report Formats**
- Which output formats to support?
- **Options**: HTML, PDF, Markdown, PowerPoint
- **Recommendation**: HTML (Phase 3), PDF (Phase 4), others if requested

**Q8: Interactive Dashboard - Deployment**
- Should dashboard be embeddable in other applications?
- **Recommendation**: Start with standalone, add embedding support later

**Q9: Feature Engineering Integration**
- Should we support explaining engineered features (e.g., polynomial)?
- **Recommendation**: Yes, but user must provide feature definitions

**Q10: Comparison Mode**
- Should we support comparing multiple models' explanations?
- **Recommendation**: Add in Phase 4 as "model comparison" feature

---

## Appendix A: Configuration Schema

### Global Configuration

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class InterpreterConfig:
    """Global configuration for interpreter"""

    # Model settings
    model_type: str = 'auto'  # 'tree', 'linear', 'auto'

    # Categorical handling
    categorical_features: List[str] = None
    categorical_encoding: str = 'auto'  # 'auto', 'onehot', 'label'

    # SHAP settings
    shap_algorithm: str = 'auto'  # 'auto', 'tree', 'kernel', 'linear'
    shap_n_samples: int = 100  # For kernel explainer

    # Caching
    enable_cache: bool = True
    cache_dir: str = '.shap_cache'

    # Visualization
    default_plot_backend: str = 'plotly'  # 'plotly', 'matplotlib'
    plot_style: str = 'default'  # 'default', 'presentation', 'technical'

    # Business translation
    domain: str = 'generic'  # 'generic', 'credit', 'conversion', 'churn'
    narrative_detail: str = 'summary'  # 'summary', 'detailed', 'technical'

    # Performance
    n_jobs: int = -1  # Parallel processing
    verbose: bool = False
```

---

## Appendix B: API Reference

### Core API

```python
class ModelInterpreter:
    """Main interpreter class"""

    def __init__(
        self,
        model,
        data: pd.DataFrame,
        categorical_features: List[str] = None,
        config: InterpreterConfig = None
    ):
        """Initialize interpreter with model and data"""

    def plot_global_importance(
        self,
        top_n: int = 10,
        plot_type: str = 'bar'
    ) -> Figure:
        """Generate global feature importance plot"""

    def plot_local_explanation(
        self,
        instance_idx: int,
        plot_type: str = 'waterfall'
    ) -> Figure:
        """Generate local explanation for single instance"""

    def plot_dependence(
        self,
        feature: str,
        interaction_feature: str = None
    ) -> Figure:
        """Generate feature dependence plot"""

    def generate_global_narrative(self) -> str:
        """Generate text summary of global importance"""

    def generate_local_narrative(
        self,
        instance_idx: int,
        detail_level: str = 'summary'
    ) -> str:
        """Generate text explanation for single instance"""

    def what_if(
        self,
        instance_idx: int,
        changes: Dict[str, Any]
    ) -> Dict:
        """Calculate impact of changing feature values"""

    def generate_report(
        self,
        instance_indices: List[int],
        output_format: str = 'html',
        output_path: str = None
    ) -> Dict:
        """Generate comprehensive report"""
```

---

## Appendix C: Example Datasets

### Customer Conversion Dataset

**File**: `data/customer_conversion.csv`
**Rows**: 5,000
**Features**: 17 (+ target)
**Target**: Binary (converted: 0/1)
**Conversion Rate**: 36%

**Feature Types**:
- **Numeric** (10): pages_viewed, time_on_site_mins, videos_watched, email_opens, session_count, age, previous_courses, account_age_days, days_since_signup, signup_month
- **Categorical** (7): referral_source, country, occupation, device_type, discount_offered, signup_day_of_week

**Known Patterns**:
- Engagement drives conversion (strongest effect)
- Discount has non-linear effect
- Professional occupation converts best
- Age shows inverted U-shape

**Purpose**: Perfect for testing categorical handling and business translation

---

## Document Control

### Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-22 | Claude | Initial draft for review |

### Review Status

- [ ] Technical architecture reviewed
- [ ] Package dependencies approved
- [ ] Feature scope confirmed
- [ ] Implementation phases agreed
- [ ] Open questions resolved

### Next Steps

1. **Review this specification** - Add comments, questions, feedback
2. **Resolve open questions** - Make decisions on scope and approach
3. **Finalize architecture** - Lock down module structure
4. **Begin Phase 1** - Implement MVP foundation

---

## Your Feedback

**Please review and provide feedback on:**

1. **Overall Approach**: Does the architecture make sense?
2. **Feature Scope**: Are we building the right features? Any missing?
3. **Categorical Handling**: Is the grouping approach clear and correct?
4. **Business Translation**: Will the narrative generation meet your needs?
5. **Visualizations**: Are the proposed charts the right ones?
6. **Phasing**: Is the implementation timeline realistic?
7. **Open Questions**: Your answers to the questions in Section 11
8. **API Design**: Is the usage interface intuitive?

**Add your comments directly to this document or create a separate feedback file.**

---

**END OF SPECIFICATION**
