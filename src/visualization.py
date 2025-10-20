"""
Visualization module for fish classification analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_correlation_matrix(df, columns=None, figsize=(8, 6)):
    """
    Plot correlation matrix heatmap
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): Columns to include in correlation
        figsize (tuple): Figure size
    """
    if columns is None:
        columns = ['ph', 'temperature', 'turbidity']
    
    correlation_matrix = df[columns].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap for Numerical Columns')
    plt.tight_layout()
    return plt.gcf()


def plot_feature_relationships(df, figsize=(12, 10)):
    """
    Plot scatter plots for feature relationships
    
    Args:
        df (pd.DataFrame): Input dataframe
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.subplot(2, 2, 1)
    sns.regplot(x='ph', y='temperature', data=df)
    plt.title('pH vs Temperature')
    
    plt.subplot(2, 2, 2)
    sns.regplot(x='ph', y='turbidity', data=df)
    plt.title('pH vs Turbidity')
    
    plt.subplot(2, 2, 3)
    sns.regplot(x='temperature', y='turbidity', data=df)
    plt.title('Temperature vs Turbidity')
    
    plt.tight_layout()
    return plt.gcf()


def plot_class_distribution(y_encoded, label_encoder=None, figsize=(10, 6)):
    """
    Plot class distribution
    
    Args:
        y_encoded (array): Encoded labels
        label_encoder (LabelEncoder): Label encoder for class names
        figsize (tuple): Figure size
    """
    class_distribution = pd.Series(y_encoded).value_counts()
    plt.figure(figsize=figsize)
    sns.barplot(x=class_distribution.index, y=class_distribution.values, palette="viridis")
    plt.title('Class Distribution (Label Encoded)')
    plt.xlabel('Encoded Class')
    plt.ylabel('Number of Samples')
    
    if label_encoder:
        plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=45)
    
    plt.tight_layout()
    return plt.gcf()


def plot_model_comparison(results_df, figsize=(12, 8)):
    """
    Plot model performance comparison
    
    Args:
        results_df (pd.DataFrame): Results dataframe with metrics
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    results_df[['train_accuracy', 'test_accuracy']].plot(kind='bar', ax=plt.gca(), color=['blue', 'orange'])
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.legend(['Train', 'Test'])
    
    # Plot F1 score
    plt.subplot(2, 2, 2)
    results_df[['train_f1_score', 'test_f1_score']].plot(kind='bar', ax=plt.gca(), color=['blue', 'orange'])
    plt.title('F1 Score Comparison')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(['Train', 'Test'])
    
    # Plot precision
    plt.subplot(2, 2, 3)
    results_df[['train_precision', 'test_precision']].plot(kind='bar', ax=plt.gca(), color=['blue', 'orange'])
    plt.title('Precision Comparison')
    plt.ylabel('Precision')
    plt.xticks(rotation=45, ha='right')
    plt.legend(['Train', 'Test'])
    
    # Plot recall
    plt.subplot(2, 2, 4)
    results_df[['train_recall', 'test_recall']].plot(kind='bar', ax=plt.gca(), color=['blue', 'orange'])
    plt.title('Recall Comparison')
    plt.ylabel('Recall')
    plt.xticks(rotation=45, ha='right')
    plt.legend(['Train', 'Test'])
    
    plt.tight_layout()
    return plt.gcf()


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', figsize=(8, 6)):
    """
    Plot confusion matrix heatmap
    
    Args:
        cm (array): Confusion matrix
        classes (array): Class labels
        title (str): Plot title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    return plt.gcf()


def plot_roc_curve(fpr, tpr, roc_auc, model_name='Model', figsize=(8, 6)):
    """
    Plot ROC curve
    
    Args:
        fpr (array): False positive rate
        tpr (array): True positive rate
        roc_auc (float): AUC score
        model_name (str): Name of the model
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title(f'ROC Curve for {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    return plt.gcf()


def plot_svm_metrics(train_metrics, test_metrics, figsize=(10, 6)):
    """
    Plot SVM metrics comparison
    
    Args:
        train_metrics (dict): Training metrics
        test_metrics (dict): Test metrics
        figsize (tuple): Figure size
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    train_scores = [train_metrics['accuracy'], train_metrics['precision'], 
                   train_metrics['recall'], train_metrics['f1']]
    test_scores = [test_metrics['accuracy'], test_metrics['precision'], 
                  test_metrics['recall'], test_metrics['f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=figsize)
    plt.bar(x - width/2, train_scores, width, label='Train', color='royalblue')
    plt.bar(x + width/2, test_scores, width, label='Test', color='seagreen')
    
    plt.title('Model Performance Comparison: Training vs Test Sets')
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    
    for i, (train, test) in enumerate(zip(train_scores, test_scores)):
        plt.text(i - width/2, train + 0.02, f'{train:.2f}', ha='center')
        plt.text(i + width/2, test + 0.02, f'{test:.2f}', ha='center')
    
    plt.tight_layout()
    return plt.gcf()
