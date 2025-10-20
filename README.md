# Quantum-Enhanced Machine Learning Framework for Predicting Fish Species Suitability in Aquaculture Environments

A hybrid quantum-classical machine learning framework for predicting fish species pond suitability based on real-time water quality parameters in aquaculture environments.

## 🎯 Overview

This project addresses the critical need for effective water quality management in aquaculture by introducing a comprehensive machine learning framework that combines classical and quantum computing approaches. Using a real-time water quality dataset of **40,280 records** collected from five ponds in Bangladesh, the system predicts optimal fish species for specific environmental conditions.

The framework leverages:

- **Neural Autoencoders** for latent feature extraction
- **Quantum-Derived Features** through parameterized quantum circuits
- **Advanced ML Algorithms** (Random Forest, XGBoost, SVM)
- **Hybrid Feature Ensemble** combining original data with quantum and latent features
- **SHAP Analysis** for model interpretability and decision-making insights

This research demonstrates how quantum-enhanced methodologies can improve predictive analysis tools for sustainable aquaculture systems and environmental management strategies.

## ✨ Features

### Dataset

- **40,280 Records** from 5 ponds in Bangladesh
- **11 Fish Species**: Sing, Silver Carp, Katla, Prawn, Shrimp, Rui, Pangas, Tilapia, Koi, and more
- **3 Environmental Parameters**: pH, Temperature, Turbidity
- Real-time water quality monitoring data

### Quantum-Enhanced Framework

#### Neural Autoencoders
- **Architecture**: Input → Dense(32) → LeakyReLU → Dense(latent_dim=5) → Decoder
- **Purpose**: Extract non-linear latent representations from environmental data
- **Training**: 200 epochs with MSE loss optimization
- **Output**: 5 latent features capturing complex environmental patterns

#### Parameterized Quantum Circuits (PennyLane)
- **Quantum Device**: `default.qubit` simulator with 2 qubits
- **Circuit Architecture**:
  - RX rotation gate on qubit 0 (controlled by latent feature 1)
  - RY rotation gate on qubit 1 (controlled by latent feature 2)
  - RZ rotation gate on qubit 0 (controlled by latent feature 3)
  - CNOT gate between qubits 0 and 1 (entanglement)
- **Measurement**: Pauli-Z expectation values from each qubit
- **Output**: 2 quantum-derived features per sample

#### Hybrid Feature Ensemble
The final feature set combines:
- **Original Features** (3): pH, temperature, turbidity
- **Latent Features** (5): Autoencoder-extracted representations
- **Quantum Features** (2): Quantum circuit expectation values
- **Total**: 10 features for enhanced prediction

### Machine Learning Pipeline

- **Advanced ML Algorithms**: 
  - Ensemble Methods: Random Forest, XGBoost, LightGBM, CatBoost
  - Neural Networks: Artificial Neural Network (MLP)
  - Instance-based: k-Nearest Neighbor
  - Support Vector Machines: SVM with RBF kernel
  - Decision Trees
- **Dimensionality Reduction**: PCA for optimal feature selection
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Model Interpretability**: SHAP (SHapley Additive exPlanations) analysis

### Evaluation Metrics

Comprehensive performance assessment using:
- **Accuracy**: Overall prediction correctness
- **Balanced Accuracy**: Performance across imbalanced classes
- **Precision & Recall**: Class-specific performance
- **F1-Score**: Harmonic mean of precision and recall
- **Matthews Correlation Coefficient (MCC)**: Comprehensive quality measure
- **AUC-ROC**: Area under ROC curve
- **Log Loss**: Probabilistic prediction quality
- **Confusion Matrices**: Detailed error analysis
- **Training & Prediction Time**: Computational efficiency metrics

## � Workflow Overview

The quantum-enhanced pipeline consists of six main stages:

1. **Data Loading**: 40,280 records with pH, temperature, turbidity
2. **Quantum Feature Extraction**: Autoencoder (5 latent) + Quantum Circuit (2 quantum features)
3. **Feature Engineering**: PCA reduction and correlation analysis
4. **Model Training**: 8 models with/without SMOTE for class balancing
5. **Interpretability**: SHAP analysis for feature importance
6. **Visualization**: Comprehensive performance plots and metrics

## �🚀 Quick Start

### Option 1: Run the Complete Quantum Pipeline

Execute the entire quantum-enhanced workflow with a single command:

```bash
python run_quantum_pipeline.py
```

This will:
1. Generate quantum features using autoencoders and PennyLane circuits
2. Apply PCA for dimensionality reduction
3. Train and evaluate models (with and without SMOTE)
4. Perform SHAP analysis for interpretability
5. Generate comprehensive visualizations
6. Save all results to the `results/` directory

### Option 2: Use Jupyter Notebooks

#### Basic Classification (Fish.ipynb)
```bash
jupyter notebook Fish.ipynb
```

#### Quantum-Enhanced Features (Quantum_Features.ipynb)
```bash
jupyter notebook Quantum_Features.ipynb
```

### Option 3: Use Modular Python Scripts

Import and use individual modules for custom workflows:

```python
from src.quantum_feature_extraction import generate_enhanced_dataset
from src.quantum_models import get_all_models
from src.advanced_evaluation import train_and_evaluate_all_models

# Generate quantum features
enhanced_df, scaler, encoder = generate_enhanced_dataset(df)

# Train models
models = get_all_models(use_smote=True)
trained_models, results = train_and_evaluate_all_models(models, X_train, X_test, y_train, y_test)
```

### Option 4: Run Example Scripts

See `examples.py` for detailed usage examples:

```bash
# Run quick model comparison
python examples.py

# Or import specific examples
from examples import example_quantum_features, example_shap_analysis
example_quantum_features()  # Generate quantum features
example_shap_analysis()      # Perform SHAP analysis
```

## 📁 Project Structure

```
Quantum-Enhanced-Machine-Learning-Framework/
├── data/
│   ├── realfishdataset.csv                    # Original dataset
│   ├── enhanced_data_with_quantum_features.csv # Quantum-enhanced dataset
│   └── README.md                               # Data documentation
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py                  # Data loading and preprocessing
│   ├── quantum_feature_extraction.py          # Autoencoder & quantum circuits
│   ├── feature_engineering.py                 # PCA & feature combination
│   ├── quantum_models.py                      # Model definitions (base & optimized)
│   ├── train_models.py                        # Model training functions
│   ├── evaluate_models.py                     # Basic model evaluation
│   ├── advanced_evaluation.py                 # Comprehensive metrics
│   ├── visualization.py                       # Base plotting functions
│   ├── quantum_visualization.py               # Enhanced visualizations
│   └── shap_analysis.py                       # SHAP explainability
├── notebooks/
│   ├── Fish.ipynb                             # Basic fish classification
│   └── Quantum_Features.ipynb                 # Quantum-enhanced workflow
├── results/
│   ├── plots/                                 # Generated visualizations
│   ├── results_without_smote.csv              # Model performance (no SMOTE)
│   └── results_with_smote.csv                 # Model performance (with SMOTE)
├── requirements.txt                           # Python dependencies
├── run_quantum_pipeline.py                    # Main quantum pipeline script
├── LICENSE
└── README.md
```

## 📦 Module Documentation

### Core Modules

#### `quantum_feature_extraction.py`
Quantum feature generation using PennyLane and TensorFlow autoencoders:
- `build_autoencoder()` - Create encoder-decoder architecture
- `train_autoencoder()` - Train on environmental data
- `extract_latent_features()` - Get latent representations
- `create_quantum_circuit()` - Define parameterized quantum circuit
- `extract_quantum_features()` - Apply quantum transformations
- `generate_enhanced_dataset()` - Complete quantum feature pipeline

#### `feature_engineering.py`
Feature manipulation and dimensionality reduction:
- `apply_pca_to_features()` - Apply PCA to any feature set
- `apply_pca_to_latent_features()` - PCA specifically for latent features
- `get_feature_columns()` - Select feature combinations
- `create_feature_sets()` - Generate multiple feature configurations
- `get_correlation_matrix()` - Compute feature correlations

#### `quantum_models.py`
Model definitions optimized for quantum features:
- `get_base_models()` - Standard models (no SMOTE)
- `get_optimized_models_with_smote()` - Optimized for balanced data
- `get_svm_model()` - SVM with RBF kernel
- `get_tree_based_models()` - Models compatible with SHAP
- `get_all_models()` - Complete model suite

#### `advanced_evaluation.py`
Comprehensive model evaluation:
- `evaluate_model_comprehensive()` - All metrics for single model
- `train_and_evaluate_all_models()` - Train and assess multiple models
- `compare_model_performance()` - Rank models by metrics
- `get_classification_report()` - Detailed per-class performance

#### `shap_analysis.py`
Model interpretability and explainability:
- `create_tree_explainer()` - SHAP for tree-based models
- `analyze_model_with_shap()` - Complete SHAP workflow
- `plot_shap_summary()` - Feature importance visualization
- `plot_shap_force()` - Instance-level explanations
- `get_feature_importance_from_shap()` - Extract importance rankings

#### `quantum_visualization.py`
Enhanced visualization suite:
- `plot_correlation_matrix_enhanced()` - Heatmap with all features
- `plot_pca_scatter()` - PCA component visualization
- `plot_quantum_features_distribution()` - Quantum feature histograms
- `plot_comprehensive_model_comparison()` - Multi-metric comparison
- `plot_train_test_comparison()` - Overfitting detection
- `save_all_plots()` - Batch save visualizations

### Legacy Modules

#### `data_preprocessing.py`
Basic data operations (used in Fish.ipynb):
- `load_data()` - Load CSV datasets
- `preprocess_data()` - Scale and encode
- `apply_smote()` - Balance training data
- `encode_target()` - Label encoding

#### `train_models.py`, `evaluate_models.py`, `visualization.py`
Original modules for basic fish classification workflow

## 🤖 Models

### Core Machine Learning Algorithms

The framework implements and compares three primary ML algorithms optimized for aquaculture prediction:

1. **Random Forest**

   - Ensemble learning method for robust predictions
   - Handles non-linear relationships in environmental data
   - Feature importance for water quality parameters
2. **XGBoost (Extreme Gradient Boosting)**

   - Advanced gradient boosting framework
   - Optimized for speed and performance
   - Handles missing data and prevents overfitting
3. **Support Vector Machine (SVM)**

   - RBF kernel for non-linear decision boundaries
   - Enhanced with SMOTE for class balance
   - High-dimensional feature space mapping

### Quantum-Enhanced Features

- **Neural Autoencoder**: Extracts latent representations of environmental conditions
- **Parameterized Quantum Circuits**: Generates quantum-derived features for enhanced prediction
- **Hybrid Feature Space**: Combines classical, latent, and quantum features

### Additional Models (Comparative Analysis)

4. **Artificial Neural Networks (ANN)** - MLPClassifier
5. **k-Nearest Neighbour (KNN)** - k=4
6. **Decision Tree** - max_depth=5
7. **LightGBM** - Fast gradient boosting
8. **CatBoost** - Categorical boosting

## 📈 Results

### Performance Metrics

The quantum-enhanced models are evaluated using comprehensive metrics:

- **Accuracy**: Overall correct predictions across all fish species
- **F1-Score**: Harmonic mean of precision and recall, crucial for imbalanced datasets
- **Balanced Accuracy**: Accounts for class imbalance in fish species distribution
- **Matthews Correlation Coefficient (MCC)**: Comprehensive quality measure for binary/multiclass classification

### Best Performing Model

The quantum-enhanced methodology demonstrates strong performance across all metrics. The best model is automatically selected based on a combination of:

- Test accuracy
- Balanced accuracy
- MCC score
- F1-score

Results are visualized in comparison plots showing train vs test performance across all metrics.

### Key Findings

1. **Quantum Enhancement**: Quantum-derived features significantly improve prediction accuracy
2. **Hybrid Approach**: Combining classical, latent, and quantum features outperforms single-source models
3. **SMOTE Effectiveness**: Class balancing techniques improve minority class prediction
4. **PCA Benefits**: Dimensionality reduction maintains performance while reducing computational complexity

## 🔍 SHAP Analysis

SHAP (SHapley Additive exPlanations) is employed to evaluate how each feature influences the models' decision-making process, providing critical insights into:

- **Feature Importance**: Which environmental parameters (pH, temperature, turbidity) have the most impact
- **Decision Transparency**: Understanding how quantum-enhanced features contribute to predictions
- **Species-Specific Patterns**: How different fish species respond to environmental conditions
- **Model Interpretability**: Making black-box models explainable for aquaculture practitioners

### Models Analyzed with SHAP:

- Random Forest (with quantum features)
- XGBoost (with quantum features)
- LightGBM
- CatBoost
- KNN
- ANN
- SVM

This interpretability is crucial for:

- Validating model decisions against aquaculture domain knowledge
- Building trust in quantum-enhanced predictions
- Identifying optimal water quality ranges for each fish species
- Supporting sustainable aquaculture management decisions

## 📝 Dataset Information

### Real-Time Water Quality Dataset

**Dataset Source**: [A Real-Time Dataset of Pond Water for Fish Farming using IoT devices](https://doi.org/10.17632/hxd382z2fg.2)
**Authors**: Md Monirul Islam, Mohammod Abul Kashem
**Published**: 29 September 2023 (Version 2)
**DOI**: 10.17632/hxd382z2fg.2

The dataset comprises **40,280 records** (40,280 rows × 4 columns) collected from **5 ponds** in Bangladesh using an IoT framework for real-time aquatic environment monitoring. Three sensors (pH, Temperature, and Turbidity) along with an Arduino controller were used to monitor water quality across the ponds.

#### Environmental Parameters (Features):

- **pH**: Water acidity/alkalinity level (critical for fish health)
- **Temperature**: Water temperature in °C (affects metabolism and oxygen levels)
- **Turbidity**: Water clarity measurement in NTU (impacts light penetration and feeding)

## 👥 Authors

### Research Team

- **Sowad Rahman**¹ - Co-Lead Researcher
- **Showrin Rahman**¹ - Co-Lead Researcher & Implementation
- **Adity Khisa**² - Research Contributor
- **Soumitra Paul**² - Research Contributor
- **Jia Uddin**³ - Principal Investigator & Corresponding Author

### Corresponding Author

For research inquiries and collaboration:

- **Dr. Jia Uddin** (Principal Investigator)
- - **Email**: [Contact for correspondence]

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{quantum_fish_classification_2024,
  title={Quantum-Enhanced Machine Learning Framework for Predicting Fish Species Suitability in Aquaculture Environments},
  author={Rahman, Sowad and Rahman, Showrin and Khisa, Adity and Paul, Soumitra and Uddin, Jia},
  year={2024},
  publisher={GitHub},
  url={https://github.com/showrin20/Quantum-Enhanced-Machine-Learning-Framework}
}
```

## 📚 Additional Documentation

- **[MODULES_SUMMARY.md](MODULES_SUMMARY.md)** - Detailed module documentation and data flow
- **[requirements.txt](requirements.txt)** - Complete list of dependencies
- **[examples.py](examples.py)** - Usage examples for all modules

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- Dataset provided by Md Monirul Islam and Mohammod Abul Kashem
- PennyLane team for quantum computing framework
- scikit-learn, TensorFlow, and XGBoost communities
