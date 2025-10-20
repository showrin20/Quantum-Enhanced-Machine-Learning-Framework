

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

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fish-classification.git
cd fish-classification
```

### Step 2: Create Virtual Environment (Recommended)

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 📊 Usage

### Option 1: Run the Complete Pipeline

```bash
python run_pipeline.py
```

This will:

1. Load and preprocess the data
2. Train all models
3. Generate evaluation metrics
4. Create visualizations
5. Perform SHAP analysis

### Option 2: Run Individual Components

**Data Analysis Only:**

```bash
python -c "from src.data_loader import load_and_explore_data; load_and_explore_data('data/realfishdataset.csv')"
```

**Train Specific Model:**

```bash
python -c "from src.model_training import train_single_model; train_single_model('Random Forest')"
```

**Generate Visualizations:**

```bash
python -c "from src.visualization import generate_all_plots; generate_all_plots()"
```

### Option 3: Use Jupyter Notebook

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

The dataset comprises **40,280 records** collected from **5 ponds** in Bangladesh, representing diverse aquaculture environments.

#### Environmental Parameters (Features):
- **pH**: Water acidity/alkalinity level (critical for fish health)
- **Temperature**: Water temperature in °C (affects metabolism and oxygen levels)
- **Turbidity**: Water clarity measurement in NTU (impacts light penetration and feeding)

#### Fish Species (Target):
The dataset includes 11 common aquaculture species in Bangladesh:
1. **Sing** (Stinging Catfish)
2. **Silver Carp**
3. **Katla** (Major Carp)
4. **Prawn**
5. **Shrimp**
6. **Rui** (Rohu)
7. **Pangas** (Pangasius)
8. **Tilapia**
9. **Koi** (Climbing Perch)
10. Additional species...

#### Data Collection:
- Real-time monitoring from active fish ponds
- Balanced representation across different environmental conditions
- Quality-controlled measurements ensuring data reliability

#### Example Data Format:

```csv
ph,temperature,turbidity,fish
7.2,28.5,12.3,Rui
6.8,26.1,8.7,Tilapia
7.5,29.0,10.5,Katla
8.1,27.5,15.2,Silver Carp
```

### Dataset Significance

This dataset is crucial for:
- Optimizing aquatic environmental conditions
- Improving fish farming productivity
- Enhancing aquaculture sustainability in Bangladesh
- Supporting data-driven aquaculture management decisions

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors:**

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**2. Memory Issues with Large Dataset:**

```python
# In run_pipeline.py, reduce batch size or use sampling
df = df.sample(frac=0.5, random_state=42)
```

**3. SHAP Visualization Not Showing:**

```bash
# Install additional dependencies
pip install ipython ipywidgets
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

### Research Team

- **Sowad Rahman**¹ - Co-Lead Researcher
- **Showrin Rahman**¹ - Co-Lead Researcher & Implementation
- **Adity Khisa**² - Research Contributor
- **Soumitra Paul**² - Research Contributor  
- **Jia Uddin**³ - Principal Investigator & Corresponding Author

¹ Affiliation 1  
² Affiliation 2  
³ Corresponding Author

### Corresponding Author
For research inquiries and collaboration:
- **Dr. Jia Uddin** (Principal Investigator)
- Email: [Contact for correspondence]

## 🙏 Acknowledgments

- **Dataset**: Real-time water quality data from 5 aquaculture ponds in Bangladesh (40,280 records)
- **Fish Species**: 11 common aquaculture species including Sing, Silver Carp, Katla, Prawn, Shrimp, Rui, Pangas, Tilapia, and Koi
- **Technologies**: 
  - Classical ML: scikit-learn, XGBoost, LightGBM, CatBoost
  - Quantum Computing: Parameterized quantum circuits
  - Interpretability: SHAP (SHapley Additive exPlanations)
  - Deep Learning: Neural autoencoders for feature extraction
- **Application Domain**: Sustainable aquaculture and environmental management in Bangladesh
- **Research Focus**: Quantum-enhanced machine learning for real-world environmental applications

### Impact

This research contributes to:
- **Sustainable Aquaculture**: Optimizing fish farming practices in Bangladesh
- **Food Security**: Supporting the emergent aquaculture industry
- **Environmental Management**: Data-driven water quality monitoring
- **Quantum ML**: Demonstrating practical applications of quantum-enhanced machine learning
- **Aquatic Productivity**: Improving fish cultivation efficiency through predictive analytics

## 📧 Contact

For questions, collaboration, or correspondence:

- **Email**: [Corresponding Author Email]
- **Research Inquiries**: Contact Dr. Jia Uddin (Principal Investigator)
- **GitHub**: [@showrinrahman](https://github.com/showrinrahman)

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{rahman2025quantum,
  title={Quantum-Enhanced Machine Learning Framework for Predicting Fish Species Suitability in Aquaculture Environments},
  author={Rahman, Sowad and Rahman, Showrin and Khisa, Adity and Paul, Soumitra and Uddin, Jia},
  journal={[Journal Name]},
  year={2025},
  note={Submitted for publication}
}
```

---

**Research Context**: This project addresses the critical need for effective water quality management in Bangladesh's emergent aquaculture industry. By combining quantum computing features with classical machine learning, we demonstrate how advanced computational methods can support sustainable fish farming and environmental management strategies.

**Keywords**: Quantum Machine Learning, Aquaculture, Fish Species Prediction, Water Quality Management, SHAP Interpretability, Neural Autoencoders, Parameterized Quantum Circuits, Sustainable Agriculture, Bangladesh
