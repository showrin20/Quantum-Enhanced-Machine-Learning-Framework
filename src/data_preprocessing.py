"""
Data preprocessing module for fish classification
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(file_path):
    """
    Load the fish dataset from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df, target_column='fish', feature_columns=None, test_size=0.2, random_state=42):
    """
    Preprocess the dataset: encode labels, split data, and scale features
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        feature_columns (list): List of feature column names
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler
    """
    if feature_columns is None:
        feature_columns = ['ph', 'temperature', 'turbidity']
    
    # Extract features and target
    X = df[feature_columns]
    y = df[target_column]
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to balance the training dataset
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        random_state (int): Random seed
        
    Returns:
        tuple: X_resampled, y_resampled
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def encode_target(df, target_column='fish'):
    """
    Encode target column and add to dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of target column
        
    Returns:
        tuple: Updated dataframe, label_encoder
    """
    label_encoder = LabelEncoder()
    df['fish_encoded'] = label_encoder.fit_transform(df[target_column])
    return df, label_encoder
