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

- **Neural Autoencoders**: Extract latent features from environmental data
- **Parameterized Quantum Circuits**: Generate quantum-derived features
- **Hybrid Feature Ensemble**: Combines original, latent, and quantum features

### Machine Learning Pipeline

- **Advanced ML Algorithms**: Random Forest, XGBoost, Support Vector Machines (SVM)
- **Dimensionality Reduction**: PCA for optimal feature selection
- **Class Imbalance Handling**: SMOTE oversampling techniques
- **Model Interpretability**: SHAP (SHapley Additive exPlanations) analysis

### Evaluation Metrics

- **Accuracy**: Overall prediction correctness
- **F1-Score**: Harmonic mean of precision and recall
- **Balanced Accuracy**: Performance across imbalanced classes
- **Matthews Correlation Coefficient (MCC)**: Comprehensive quality measure
- **Confusion Matrices & ROC Curves**: Visual performance assessment

## Use Jupyter Notebook

```bash
jupyter notebook Fish.ipynb
```

## 📁 Project Structure

```
fish-classification/
├── data/
│   ├── realfishdataset.csv    # Your dataset
│   └── README.md               # Data documentation
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── model_training.py      # Model training functions
│   ├── evaluation.py          # Model evaluation metrics
│   ├── visualization.py       # Plotting functions
│   └── shap_analysis.py       # SHAP explainability
├── notebooks/
│   └── Fish.ipynb             # Original Jupyter notebook
├── results/
│   ├── plots/                 # Generated visualizations
│   └── metrics/               # Performance metrics
├── models/
│   └── saved_models/          # Trained models (saved)
├── requirements.txt           # Python dependencies
├── run_pipeline.py            # Main execution script
├── .gitignore
└── README.md
```

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
- Email: [Contact for correspondence]
