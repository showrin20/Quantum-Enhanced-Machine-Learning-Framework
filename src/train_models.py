"""
Model training module for fish classification
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib


def get_models():
    """
    Get dictionary of all classification models
    
    Returns:
        dict: Dictionary mapping model names to model instances
    """
    models = {
        'Artificial Neural Networks': MLPClassifier(max_iter=300, random_state=42),
        'k-Nearest Neighbour': KNeighborsClassifier(n_neighbors=4),
        'Random Forest': RandomForestClassifier(random_state=100),
        'Decision Tree': DecisionTreeClassifier(random_state=100, max_depth=5),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
        'SVM': SVC(kernel='rbf', gamma='scale', probability=True, random_state=42)
    }
    return models


def train_model(model, X_train, y_train):
    """
    Train a single model
    
    Args:
        model: Scikit-learn compatible model
        X_train (array): Training features
        y_train (array): Training labels
        
    Returns:
        Trained model
    """
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train):
    """
    Train all models
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        
    Returns:
        dict: Dictionary of trained models
    """
    models = get_models()
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        trained_model = train_model(model, X_train, y_train)
        trained_models[model_name] = trained_model
    
    return trained_models


def save_model(model, filepath):
    """
    Save a trained model to disk
    
    Args:
        model: Trained model
        filepath (str): Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load a trained model from disk
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
