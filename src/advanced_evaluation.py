"""
Advanced model evaluation module with comprehensive metrics
"""
import numpy as np
import pandas as pd
import time
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, log_loss, classification_report
)


def evaluate_model_comprehensive(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Comprehensive evaluation of a model with all metrics
    
    Args:
        model: Trained model
        X_train (array): Training features
        X_test (array): Test features
        y_train (array): Training labels
        y_test (array): Test labels
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary of comprehensive evaluation metrics
    """
    # Measure training time (if not already trained)
    # Note: This assumes model is already trained, so we skip retraining
    
    # Measure prediction time on test set
    start_time = time.time()
    y_test_pred = model.predict(X_test)
    prediction_time = (time.time() - start_time) / len(X_test)
    
    # Get training predictions
    y_train_pred = model.predict(X_train)
    
    # Get prediction probabilities if available
    y_test_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_test_pred_proba = model.predict_proba(X_test)
    
    # Calculate all metrics
    metrics = {
        'model_name': model_name,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'train_recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'train_precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'test_precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'train_f1': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(y_test, y_test_pred),
        'prediction_time_ms_per_sample': prediction_time * 1000
    }
    
    # Add AUC-ROC and Log Loss if probabilities are available
    if y_test_pred_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr', average='weighted')
        except:
            metrics['auc_roc'] = np.nan
        
        try:
            metrics['log_loss'] = log_loss(y_test, y_test_pred_proba)
        except:
            metrics['log_loss'] = np.nan
    else:
        metrics['auc_roc'] = np.nan
        metrics['log_loss'] = np.nan
    
    return metrics


def evaluate_all_models_comprehensive(models, X_train, X_test, y_train, y_test):
    """
    Comprehensive evaluation of all models
    
    Args:
        models (dict): Dictionary of trained models
        X_train (array): Training features
        X_test (array): Test features
        y_train (array): Training labels
        y_test (array): Test labels
        
    Returns:
        pd.DataFrame: DataFrame with comprehensive metrics for all models
    """
    results = []
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model_comprehensive(model, X_train, X_test, y_train, y_test, model_name)
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('model_name')
    results_df = results_df.sort_values(by='test_accuracy', ascending=False)
    
    return results_df


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Train and comprehensively evaluate a single model
    
    Args:
        model: Untrained model
        X_train (array): Training features
        X_test (array): Test features
        y_train (array): Training labels
        y_test (array): Test labels
        model_name (str): Name of the model
        
    Returns:
        tuple: (trained_model, metrics_dict)
    """
    # Measure training time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate model
    metrics = evaluate_model_comprehensive(model, X_train, X_test, y_train, y_test, model_name)
    metrics['training_time_s'] = training_time
    
    return model, metrics


def train_and_evaluate_all_models(models, X_train, X_test, y_train, y_test):
    """
    Train and comprehensively evaluate all models
    
    Args:
        models (dict): Dictionary of untrained models
        X_train (array): Training features
        X_test (array): Test features
        y_train (array): Training labels
        y_test (array): Test labels
        
    Returns:
        tuple: (trained_models_dict, results_df)
    """
    results = []
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}...")
        trained_model, metrics = train_and_evaluate_model(
            model, X_train, X_test, y_train, y_test, model_name
        )
        trained_models[model_name] = trained_model
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('model_name')
    results_df = results_df.sort_values(by='test_accuracy', ascending=False)
    
    return trained_models, results_df


def get_classification_report(model, X_test, y_test, target_names=None):
    """
    Get detailed classification report
    
    Args:
        model: Trained model
        X_test (array): Test features
        y_test (array): Test labels
        target_names (list): List of target class names
        
    Returns:
        str: Classification report
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=target_names)
    return report


def compare_model_performance(results_df, metric='test_accuracy', top_n=5):
    """
    Compare and display top performing models
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        metric (str): Metric to compare
        top_n (int): Number of top models to display
        
    Returns:
        pd.DataFrame: Top N models
    """
    top_models = results_df.nlargest(top_n, metric)
    
    print(f"\nTop {top_n} models by {metric}:")
    print("=" * 60)
    for idx, (model_name, row) in enumerate(top_models.iterrows(), 1):
        print(f"{idx}. {model_name}: {row[metric]:.4f}")
    print("=" * 60)
    
    return top_models


def get_model_summary(results_df):
    """
    Get summary statistics of model performance
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        
    Returns:
        pd.DataFrame: Summary statistics
    """
    summary = results_df.describe()
    return summary
