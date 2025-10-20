"""
Enhanced visualization module for quantum-enhanced features
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_correlation_matrix_enhanced(df, columns=None, figsize=(12, 10)):
    """
    Plot enhanced correlation matrix with all features including quantum
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): Columns to include in correlation
        figsize (tuple): Figure size
    """
    if columns is None:
        # Include all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    correlation_matrix = df[columns].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Features (Including Quantum Features)')
    plt.tight_layout()
    return plt.gcf()


def plot_pca_scatter(df, pca_col1='latent_pca1', pca_col2='latent_pca2', 
                     target_col='fish_encoded', figsize=(10, 8)):
    """
    Plot PCA scatter plot colored by target
    
    Args:
        df (pd.DataFrame): Input dataframe
        pca_col1 (str): First PCA component column
        pca_col2 (str): Second PCA component column
        target_col (str): Target column for coloring
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x=pca_col1, y=pca_col2, hue=target_col, data=df, palette='viridis')
    plt.title('PCA of Latent Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Fish Species')
    plt.tight_layout()
    return plt.gcf()


def plot_quantum_features_distribution(df, figsize=(12, 5)):
    """
    Plot distribution of quantum features
    
    Args:
        df (pd.DataFrame): Input dataframe with quantum features
        figsize (tuple): Figure size
    """
    quantum_cols = [col for col in df.columns if col.startswith('quantum')]
    
    if not quantum_cols:
        print("No quantum features found in dataframe")
        return None
    
    fig, axes = plt.subplots(1, len(quantum_cols), figsize=figsize)
    
    if len(quantum_cols) == 1:
        axes = [axes]
    
    for idx, col in enumerate(quantum_cols):
        axes[idx].hist(df[col], bins=30, color='blue', alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{col} Distribution')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig


def plot_latent_features_distribution(df, figsize=(15, 10)):
    """
    Plot distribution of latent features
    
    Args:
        df (pd.DataFrame): Input dataframe with latent features
        figsize (tuple): Figure size
    """
    latent_cols = [col for col in df.columns if col.startswith('latent') and not col.startswith('latent_pca')]
    
    if not latent_cols:
        print("No latent features found in dataframe")
        return None
    
    n_features = len(latent_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, col in enumerate(latent_cols):
        axes[idx].hist(df[col], bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{col} Distribution')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    # Hide unused subplots
    for idx in range(len(latent_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_feature_importance_comparison(feature_importances_dict, figsize=(12, 8)):
    """
    Compare feature importance across multiple models
    
    Args:
        feature_importances_dict (dict): Dictionary of {model_name: importance_df}
        figsize (tuple): Figure size
    """
    n_models = len(feature_importances_dict)
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (model_name, importance_df) in enumerate(feature_importances_dict.items()):
        top_features = importance_df.head(10)
        axes[idx].barh(top_features['feature'], top_features['importance'])
        axes[idx].set_title(f'{model_name} - Top 10 Features')
        axes[idx].set_xlabel('Importance')
        axes[idx].invert_yaxis()
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_comprehensive_model_comparison(results_df, figsize=(16, 12)):
    """
    Plot comprehensive model performance comparison
    
    Args:
        results_df (pd.DataFrame): Results dataframe with all metrics
        figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.flatten()
    
    metrics = [
        ('test_accuracy', 'Test Accuracy'),
        ('balanced_accuracy', 'Balanced Accuracy'),
        ('test_f1', 'Test F1 Score'),
        ('test_precision', 'Test Precision'),
        ('test_recall', 'Test Recall'),
        ('mcc', 'Matthews Correlation Coefficient'),
        ('auc_roc', 'AUC-ROC'),
        ('log_loss', 'Log Loss (Lower is Better)'),
        ('prediction_time_ms_per_sample', 'Prediction Time (ms)')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        if metric in results_df.columns:
            results_df[metric].plot(kind='barh', ax=axes[idx], color='steelblue')
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Score' if metric != 'log_loss' else 'Loss')
            
            # Add value labels
            for i, v in enumerate(results_df[metric]):
                if not pd.isna(v):
                    axes[idx].text(v, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    return fig


def plot_train_test_comparison(results_df, figsize=(14, 10)):
    """
    Plot train vs test performance comparison
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        figsize (tuple): Figure size
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        train_col = f'train_{metric}'
        test_col = f'test_{metric}'
        
        if train_col in results_df.columns and test_col in results_df.columns:
            x = np.arange(len(results_df))
            width = 0.35
            
            axes[idx].bar(x - width/2, results_df[train_col], width, label='Train', alpha=0.8)
            axes[idx].bar(x + width/2, results_df[test_col], width, label='Test', alpha=0.8)
            
            axes[idx].set_xlabel('Models')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_title(f'{metric.capitalize()} Comparison')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(results_df.index, rotation=45, ha='right')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix_grid(confusion_matrices, model_names, class_names, 
                                figsize=(16, 12)):
    """
    Plot multiple confusion matrices in a grid
    
    Args:
        confusion_matrices (list): List of confusion matrices
        model_names (list): List of model names
        class_names (list): List of class names
        figsize (tuple): Figure size
    """
    n_models = len(confusion_matrices)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (cm, model_name) in enumerate(zip(confusion_matrices, model_names)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{model_name}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_class_distribution_comparison(df, target_col='fish_encoded', 
                                       label_encoder=None, figsize=(12, 6)):
    """
    Plot class distribution with actual class names
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column
        label_encoder (LabelEncoder): Label encoder for class names
        figsize (tuple): Figure size
    """
    class_distribution = df[target_col].value_counts().sort_index()
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(class_distribution)), class_distribution.values, 
                   color='viridis', alpha=0.8, edgecolor='black')
    
    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Fish Species', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    
    if label_encoder:
        plt.xticks(range(len(label_encoder.classes_)), 
                  label_encoder.classes_, rotation=45, ha='right')
    else:
        plt.xticks(range(len(class_distribution)), class_distribution.index)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return plt.gcf()


def save_all_plots(plots_dict, output_dir='results/plots'):
    """
    Save all plots to files
    
    Args:
        plots_dict (dict): Dictionary of {filename: figure}
        output_dir (str): Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, fig in plots_dict.items():
        if fig is not None:
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filepath}")
            plt.close(fig)
