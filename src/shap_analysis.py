"""
SHAP analysis module for model interpretability
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def initialize_shap():
    """Initialize SHAP JavaScript for visualization"""
    shap.initjs()


def create_tree_explainer(model):
    """
    Create SHAP TreeExplainer for tree-based models
    
    Args:
        model: Trained tree-based model (RandomForest, XGBoost, LightGBM, CatBoost)
        
    Returns:
        TreeExplainer: SHAP explainer object
    """
    explainer = shap.TreeExplainer(model)
    return explainer


def create_kernel_explainer(model, X_background, **kwargs):
    """
    Create SHAP KernelExplainer for any model
    
    Args:
        model: Trained model with predict method
        X_background (array): Background dataset for explainer
        **kwargs: Additional arguments for KernelExplainer
        
    Returns:
        KernelExplainer: SHAP explainer object
    """
    explainer = shap.KernelExplainer(model.predict, X_background, **kwargs)
    return explainer


def calculate_shap_values(explainer, X):
    """
    Calculate SHAP values for given data
    
    Args:
        explainer: SHAP explainer object
        X (array or DataFrame): Data to explain
        
    Returns:
        array: SHAP values
    """
    shap_values = explainer.shap_values(X)
    return shap_values


def plot_shap_summary(shap_values, X, feature_names=None, max_display=10):
    """
    Plot SHAP summary plot
    
    Args:
        shap_values (array): SHAP values
        X (array or DataFrame): Feature data
        feature_names (list): Feature names
        max_display (int): Maximum number of features to display
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=max_display)
    plt.tight_layout()


def plot_shap_bar(shap_values, X, feature_names=None, max_display=10):
    """
    Plot SHAP bar plot (mean absolute SHAP values)
    
    Args:
        shap_values (array): SHAP values
        X (array or DataFrame): Feature data
        feature_names (list): Feature names
        max_display (int): Maximum number of features to display
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", feature_names=feature_names, max_display=max_display)
    plt.tight_layout()


def plot_shap_force(explainer, shap_values, X, instance_idx=0, feature_names=None):
    """
    Plot SHAP force plot for a single instance
    
    Args:
        explainer: SHAP explainer object
        shap_values (array): SHAP values
        X (array or DataFrame): Feature data
        instance_idx (int): Index of instance to explain
        feature_names (list): Feature names
    """
    if isinstance(X, pd.DataFrame):
        instance = X.iloc[instance_idx]
    else:
        instance = X[instance_idx]
    
    # Handle multi-class case
    if isinstance(shap_values, list):
        # For multi-class, use first class or specify which class
        class_idx = 0
        shap_values_instance = shap_values[class_idx][instance_idx]
        expected_value = explainer.expected_value[class_idx]
    else:
        shap_values_instance = shap_values[instance_idx]
        expected_value = explainer.expected_value
    
    shap.force_plot(
        expected_value,
        shap_values_instance,
        instance,
        feature_names=feature_names,
        matplotlib=True
    )
    plt.tight_layout()


def plot_shap_waterfall(explainer, shap_values, X, instance_idx=0):
    """
    Plot SHAP waterfall plot for a single instance
    
    Args:
        explainer: SHAP explainer object
        shap_values (array): SHAP values
        X (array or DataFrame): Feature data
        instance_idx (int): Index of instance to explain
    """
    # Handle multi-class case
    if isinstance(shap_values, list):
        class_idx = 0
        shap_explanation = shap.Explanation(
            values=shap_values[class_idx][instance_idx],
            base_values=explainer.expected_value[class_idx],
            data=X.iloc[instance_idx] if isinstance(X, pd.DataFrame) else X[instance_idx]
        )
    else:
        shap_explanation = shap.Explanation(
            values=shap_values[instance_idx],
            base_values=explainer.expected_value,
            data=X.iloc[instance_idx] if isinstance(X, pd.DataFrame) else X[instance_idx]
        )
    
    shap.waterfall_plot(shap_explanation)


def plot_shap_dependence(shap_values, X, feature_idx, feature_names=None, interaction_index="auto"):
    """
    Plot SHAP dependence plot for a feature
    
    Args:
        shap_values (array): SHAP values
        X (array or DataFrame): Feature data
        feature_idx (int or str): Feature index or name
        feature_names (list): Feature names
        interaction_index (str or int): Feature to use for coloring
    """
    plt.figure(figsize=(10, 6))
    
    # Handle multi-class case
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X,
        feature_names=feature_names,
        interaction_index=interaction_index
    )
    plt.tight_layout()


def analyze_model_with_shap(model, X_train, X_test, feature_names=None, 
                            model_type='tree', sample_size=100):
    """
    Comprehensive SHAP analysis for a model
    
    Args:
        model: Trained model
        X_train (array or DataFrame): Training data (for background)
        X_test (array or DataFrame): Test data to explain
        feature_names (list): Feature names
        model_type (str): Type of model ('tree' or 'kernel')
        sample_size (int): Sample size for test data analysis
        
    Returns:
        tuple: (explainer, shap_values)
    """
    print(f"Creating SHAP explainer ({model_type})...")
    
    # Create explainer
    if model_type == 'tree':
        explainer = create_tree_explainer(model)
    else:
        # Use a sample of training data as background
        background_size = min(100, len(X_train))
        if isinstance(X_train, pd.DataFrame):
            background = X_train.sample(n=background_size, random_state=42)
        else:
            indices = np.random.choice(len(X_train), background_size, replace=False)
            background = X_train[indices]
        explainer = create_kernel_explainer(model, background)
    
    # Calculate SHAP values on a sample of test data
    print("Calculating SHAP values...")
    if isinstance(X_test, pd.DataFrame):
        X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    else:
        sample_indices = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
        X_sample = X_test[sample_indices]
    
    shap_values = calculate_shap_values(explainer, X_sample)
    
    print("SHAP analysis complete!")
    
    return explainer, shap_values, X_sample


def get_feature_importance_from_shap(shap_values, feature_names):
    """
    Get feature importance based on mean absolute SHAP values
    
    Args:
        shap_values (array): SHAP values
        feature_names (list): Feature names
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    # Handle multi-class case
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df
