"""
Enhanced model definitions with support for quantum features
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


def get_base_models():
    """
    Get dictionary of base classification models (without SMOTE)
    
    Returns:
        dict: Dictionary mapping model names to model instances
    """
    models = {
        'Artificial Neural Network': MLPClassifier(
            hidden_layer_sizes=(64, 32), 
            max_iter=300, 
            random_state=42
        ),
        'k-Nearest Neighbour': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(
            n_estimators=1000, 
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'XGBoost': XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            random_state=42
        ),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
    }
    return models


def get_optimized_models_with_smote():
    """
    Get dictionary of optimized classification models (with SMOTE)
    
    Returns:
        dict: Dictionary mapping model names to model instances
    """
    models = {
        'Artificial Neural Network': MLPClassifier(
            hidden_layer_sizes=(128, 64), 
            max_iter=500, 
            activation='relu', 
            random_state=42
        ),
        'k-Nearest Neighbour': KNeighborsClassifier(
            n_neighbors=10, 
            metric='minkowski'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=1500, 
            max_depth=None, 
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, 
            criterion='entropy', 
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            learning_rate=0.1, 
            max_depth=6, 
            n_estimators=1000, 
            random_state=42
        ),
        'LightGBM': LGBMClassifier(
            learning_rate=0.1, 
            max_depth=10, 
            n_estimators=1000, 
            random_state=42,
            verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            depth=8, 
            iterations=1000, 
            learning_rate=0.1, 
            verbose=0, 
            random_state=42
        )
    }
    return models


def get_svm_model(use_smote=True, kernel='rbf'):
    """
    Get SVM model configuration
    
    Args:
        use_smote (bool): Whether this is for use with SMOTE
        kernel (str): Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
        
    Returns:
        SVC: SVM model instance
    """
    return SVC(
        kernel=kernel, 
        gamma='scale', 
        random_state=42, 
        probability=True
    )


def get_all_models(use_smote=False):
    """
    Get all models including SVM
    
    Args:
        use_smote (bool): Whether to use optimized parameters for SMOTE
        
    Returns:
        dict: Dictionary of all models
    """
    if use_smote:
        models = get_optimized_models_with_smote()
    else:
        models = get_base_models()
    
    models['SVM'] = get_svm_model(use_smote=use_smote)
    
    return models


def get_tree_based_models(use_smote=False):
    """
    Get only tree-based models (for SHAP analysis)
    
    Args:
        use_smote (bool): Whether to use optimized parameters for SMOTE
        
    Returns:
        dict: Dictionary of tree-based models
    """
    if use_smote:
        all_models = get_optimized_models_with_smote()
    else:
        all_models = get_base_models()
    
    tree_models = {
        'Random Forest': all_models['Random Forest'],
        'Decision Tree': all_models['Decision Tree'],
        'XGBoost': all_models['XGBoost'],
        'LightGBM': all_models['LightGBM'],
        'CatBoost': all_models['CatBoost']
    }
    
    return tree_models


def get_model_by_name(model_name, use_smote=False):
    """
    Get a specific model by name
    
    Args:
        model_name (str): Name of the model
        use_smote (bool): Whether to use optimized parameters for SMOTE
        
    Returns:
        Model instance
    """
    all_models = get_all_models(use_smote=use_smote)
    
    if model_name not in all_models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(all_models.keys())}")
    
    return all_models[model_name]
