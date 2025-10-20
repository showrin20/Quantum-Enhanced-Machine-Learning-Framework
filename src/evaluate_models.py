"""
Model evaluation module for fish classification
"""
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate a single model on training and test sets
    
    Args:
        model: Trained model
        X_train (array): Training features
        X_test (array): Test features
        y_train (array): Training labels
        y_test (array): Test labels
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_f1_score': f1_score(y_train, y_train_pred, average='weighted'),
        'test_f1_score': f1_score(y_test, y_test_pred, average='weighted'),
        'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
        'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
        'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
        'test_recall': recall_score(y_test, y_test_pred, average='weighted')
    }
    
    return metrics


def evaluate_all_models(models, X_train, X_test, y_train, y_test):
    """
    Evaluate all models
    
    Args:
        models (dict): Dictionary of trained models
        X_train (array): Training features
        X_test (array): Test features
        y_train (array): Training labels
        y_test (array): Test labels
        
    Returns:
        pd.DataFrame: DataFrame with all evaluation metrics
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[model_name] = metrics
    
    results_df = pd.DataFrame(results).T
    return results_df


def get_confusion_matrix(model, X_test, y_test):
    """
    Generate confusion matrix for a model
    
    Args:
        model: Trained model
        X_test (array): Test features
        y_test (array): Test labels
        
    Returns:
        array: Confusion matrix
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm


def get_roc_data(model, X_test, y_test, classes):
    """
    Calculate ROC curve data for multi-class classification
    
    Args:
        model: Trained model with predict_proba method
        X_test (array): Test features
        y_test (array): Test labels
        classes (array): Unique class labels
        
    Returns:
        tuple: fpr, tpr, roc_auc
    """
    # Binarize the labels
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # Get probabilities
    y_test_prob = model.predict_proba(X_test)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_test_prob.ravel())
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc


def get_best_model(results_df, metric='test_accuracy'):
    """
    Get the name of the best performing model
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        metric (str): Metric to use for comparison
        
    Returns:
        str: Name of the best model
    """
    best_model_name = results_df[metric].idxmax()
    return best_model_name
