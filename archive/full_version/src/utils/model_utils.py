"""
Model utilities: type detection, wrapping, prediction interface.
"""

import numpy as np
from typing import Any


def get_model_type(model: Any) -> str:
    """
    Detect model type for optimal SHAP algorithm selection.

    Parameters:
        model: Trained model

    Returns:
        'tree', 'linear', or 'other'
    """
    model_class = model.__class__.__name__

    # Tree-based models
    tree_models = [
        'RandomForestClassifier', 'RandomForestRegressor',
        'GradientBoostingClassifier', 'GradientBoostingRegressor',
        'XGBClassifier', 'XGBRegressor',
        'LGBMClassifier', 'LGBMRegressor',
        'CatBoostClassifier', 'CatBoostRegressor',
        'DecisionTreeClassifier', 'DecisionTreeRegressor',
        'ExtraTreesClassifier', 'ExtraTreesRegressor'
    ]

    if model_class in tree_models:
        return 'tree'

    # Linear models
    linear_models = [
        'LogisticRegression', 'LinearRegression',
        'Ridge', 'Lasso', 'ElasticNet',
        'SGDClassifier', 'SGDRegressor'
    ]

    if model_class in linear_models:
        return 'linear'

    return 'other'


def wrap_model(model: Any) -> callable:
    """
    Create prediction function for SHAP.

    Parameters:
        model: Trained model

    Returns:
        Prediction function
    """
    # Check if model has predict_proba (classifier)
    if hasattr(model, 'predict_proba'):
        def predict_fn(X):
            proba = model.predict_proba(X)
            # Return probability of positive class for binary
            if proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        return predict_fn

    # Fallback to predict
    return model.predict


def check_model_fitted(model: Any) -> bool:
    """
    Check if model is fitted.

    Parameters:
        model: Model instance

    Returns:
        True if fitted, False otherwise
    """
    # sklearn convention
    if hasattr(model, 'classes_'):
        return True
    if hasattr(model, 'coef_'):
        return True

    # XGBoost
    if hasattr(model, 'get_booster'):
        try:
            model.get_booster()
            return True
        except:
            return False

    # LightGBM
    if hasattr(model, 'booster_'):
        return model.booster_ is not None

    # Assume fitted if we can't determine
    return True
