"""
Feature engineering module for dimensionality reduction and feature combination
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def apply_pca_to_features(df, feature_columns, n_components=2, prefix='pca'):
    """
    Apply PCA to reduce feature dimensions
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_columns (list): List of feature columns to apply PCA
        n_components (int): Number of principal components
        prefix (str): Prefix for new PCA column names
        
    Returns:
        tuple: (enhanced_df, pca_model)
    """
    # Extract features
    features = df[feature_columns].values
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)
    
    # Add PCA features to dataframe
    enhanced_df = df.copy()
    for i in range(n_components):
        enhanced_df[f'{prefix}{i+1}'] = pca_features[:, i]
    
    # Print explained variance
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    return enhanced_df, pca


def apply_pca_to_latent_features(df, n_components=2):
    """
    Apply PCA specifically to latent features
    
    Args:
        df (pd.DataFrame): Input dataframe with latent features
        n_components (int): Number of principal components
        
    Returns:
        tuple: (enhanced_df, pca_model)
    """
    # Identify latent feature columns
    latent_columns = [col for col in df.columns if col.startswith('latent')]
    
    if not latent_columns:
        raise ValueError("No latent features found in dataframe")
    
    print(f"Applying PCA to {len(latent_columns)} latent features...")
    enhanced_df, pca = apply_pca_to_features(df, latent_columns, n_components, prefix='latent_pca')
    
    return enhanced_df, pca


def drop_original_latent_features(df):
    """
    Drop original latent features after PCA
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe without original latent features
    """
    latent_columns = [col for col in df.columns if col.startswith('latent') and not col.startswith('latent_pca')]
    df_reduced = df.drop(columns=latent_columns)
    print(f"Dropped {len(latent_columns)} original latent features")
    return df_reduced


def get_feature_columns(df, include_quantum=True, include_latent_pca=False, include_latent=False):
    """
    Get list of feature columns based on options
    
    Args:
        df (pd.DataFrame): Input dataframe
        include_quantum (bool): Include quantum features
        include_latent_pca (bool): Include latent PCA features
        include_latent (bool): Include original latent features
        
    Returns:
        list: List of feature column names
    """
    # Base features
    base_features = ['ph', 'temperature', 'turbidity']
    feature_cols = base_features.copy()
    
    # Add latent features
    if include_latent:
        latent_cols = [col for col in df.columns if col.startswith('latent') and not col.startswith('latent_pca')]
        feature_cols.extend(latent_cols)
    
    # Add latent PCA features
    if include_latent_pca:
        latent_pca_cols = [col for col in df.columns if col.startswith('latent_pca')]
        feature_cols.extend(latent_pca_cols)
    
    # Add quantum features
    if include_quantum:
        quantum_cols = [col for col in df.columns if col.startswith('quantum')]
        feature_cols.extend(quantum_cols)
    
    return feature_cols


def create_feature_sets(df, target_column='fish_encoded'):
    """
    Create multiple feature sets for comparison
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Target column name
        
    Returns:
        dict: Dictionary of feature sets
    """
    feature_sets = {
        'base': ['ph', 'temperature', 'turbidity'],
        'base_latent': get_feature_columns(df, include_quantum=False, include_latent_pca=False, include_latent=True),
        'base_quantum': get_feature_columns(df, include_quantum=True, include_latent_pca=False, include_latent=False),
        'base_latent_quantum': get_feature_columns(df, include_quantum=True, include_latent_pca=False, include_latent=True),
        'base_latent_pca_quantum': get_feature_columns(df, include_quantum=True, include_latent_pca=True, include_latent=False)
    }
    
    # Print feature set information
    for name, features in feature_sets.items():
        print(f"{name}: {len(features)} features")
    
    return feature_sets


def standardize_features(X_train, X_test):
    """
    Standardize feature sets
    
    Args:
        X_train (array): Training features
        X_test (array): Test features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def get_correlation_matrix(df, feature_columns=None):
    """
    Calculate correlation matrix for features
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_columns (list): List of feature columns (None for all numeric)
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    if feature_columns is None:
        # Get all numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[feature_columns]
    
    correlation_matrix = numeric_df.corr()
    return correlation_matrix


def select_features_by_correlation(df, target_column, threshold=0.1):
    """
    Select features based on correlation with target
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Target column name
        threshold (float): Minimum correlation threshold
        
    Returns:
        list: Selected feature columns
    """
    # Calculate correlation with target
    correlations = df.corr()[target_column].abs()
    
    # Select features above threshold (excluding target itself)
    selected_features = correlations[
        (correlations >= threshold) & (correlations.index != target_column)
    ].index.tolist()
    
    print(f"Selected {len(selected_features)} features with correlation >= {threshold}")
    for feature in selected_features:
        print(f"  {feature}: {correlations[feature]:.4f}")
    
    return selected_features
