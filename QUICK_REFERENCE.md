# Quick Reference Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Pipeline
```bash
python run_quantum_pipeline.py
```

### Step 3: View Results
Results are saved in the `results/` directory:
- `results_without_smote.csv` - Model metrics without SMOTE
- `results_with_smote.csv` - Model metrics with SMOTE
- `plots/` - All visualization files

---

## 📂 File Structure Quick Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `run_quantum_pipeline.py` | Complete automated pipeline | First time setup, full analysis |
| `examples.py` | Individual component examples | Learning, testing specific features |
| `Quantum_Features.ipynb` | Interactive quantum workflow | Step-by-step exploration |
| `Fish.ipynb` | Basic classification | Baseline comparison |

---

## 🔧 Common Tasks

### Generate Quantum Features
```python
from src.quantum_feature_extraction import generate_enhanced_dataset
from src.data_preprocessing import load_data

df = load_data('data/realfishdataset.csv')
enhanced_df, scaler, encoder = generate_enhanced_dataset(df)
```

### Train a Single Model
```python
from src.quantum_models import get_model_by_name
from sklearn.model_selection import train_test_split

model = get_model_by_name('Random Forest', use_smote=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Compare All Models
```python
from src.quantum_models import get_all_models
from src.advanced_evaluation import train_and_evaluate_all_models

models = get_all_models(use_smote=True)
trained_models, results = train_and_evaluate_all_models(
    models, X_train, X_test, y_train, y_test
)
print(results[['test_accuracy', 'test_f1', 'mcc']])
```

### SHAP Analysis
```python
from src.shap_analysis import analyze_model_with_shap, plot_shap_summary

explainer, shap_values, X_sample = analyze_model_with_shap(
    model, X_train, X_test, 
    feature_names=feature_cols,
    model_type='tree'
)
plot_shap_summary(shap_values, X_sample, feature_names=feature_cols)
```

### Create Visualizations
```python
from src.quantum_visualization import (
    plot_correlation_matrix_enhanced,
    plot_comprehensive_model_comparison
)

# Correlation heatmap
plot_correlation_matrix_enhanced(enhanced_df)

# Model comparison
plot_comprehensive_model_comparison(results_df)
```

---

## 🎯 Module Import Guide

```python
# Data Handling
from src.data_preprocessing import load_data, preprocess_data, apply_smote

# Quantum Features
from src.quantum_feature_extraction import generate_enhanced_dataset

# Feature Engineering
from src.feature_engineering import (
    apply_pca_to_latent_features, 
    get_feature_columns
)

# Models
from src.quantum_models import (
    get_all_models, 
    get_tree_based_models,
    get_model_by_name
)

# Evaluation
from src.advanced_evaluation import (
    train_and_evaluate_all_models,
    compare_model_performance
)

# Interpretability
from src.shap_analysis import (
    analyze_model_with_shap,
    plot_shap_summary
)

# Visualization
from src.quantum_visualization import (
    plot_correlation_matrix_enhanced,
    plot_comprehensive_model_comparison,
    plot_quantum_features_distribution
)
```

---

## 📊 Feature Sets

| Feature Set | Features Included | Use Case |
|-------------|-------------------|----------|
| Base | pH, temperature, turbidity (3) | Baseline comparison |
| Base + Latent | Base + latent1-5 (8) | Autoencoder enhancement |
| Base + Quantum | Base + quantum1-2 (5) | Quantum enhancement only |
| Full | Base + latent1-5 + quantum1-2 (10) | Maximum information |

---

## 🤖 Model Configuration

### Without SMOTE (Imbalanced Data)
```python
models = get_all_models(use_smote=False)
# Uses standard hyperparameters
```

### With SMOTE (Balanced Data)
```python
models = get_all_models(use_smote=True)
# Uses optimized hyperparameters
```

### Available Models
- Artificial Neural Network (MLP)
- k-Nearest Neighbor
- Random Forest
- Decision Tree
- XGBoost
- LightGBM
- CatBoost
- SVM (RBF kernel)

---

## 📈 Evaluation Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| Accuracy | Overall correctness | > 0.90 |
| Balanced Accuracy | Accounts for class imbalance | > 0.85 |
| F1-Score | Harmonic mean of precision/recall | > 0.88 |
| MCC | Correlation coefficient | > 0.80 |
| AUC-ROC | Area under ROC curve | > 0.95 |
| Log Loss | Probabilistic loss | < 0.30 |

---

## 🐛 Troubleshooting

### Issue: Import errors
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: TensorFlow warnings
**Solution**: These are usually harmless. Suppress with:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

### Issue: Memory errors with SHAP
**Solution**: Reduce sample size
```python
analyze_model_with_shap(model, X_train, X_test, sample_size=50)
```

### Issue: Long training time
**Solution**: Use fewer models or reduce estimators
```python
from src.quantum_models import get_model_by_name
model = get_model_by_name('Random Forest', use_smote=False)
# Train just this one model
```

---

## 💡 Best Practices

1. **Always standardize features** before quantum circuit application
2. **Use SMOTE** when dealing with imbalanced classes
3. **Apply PCA** after generating latent features to reduce dimensionality
4. **Compare feature sets** to understand quantum feature contribution
5. **Use SHAP** on tree-based models for best interpretability
6. **Save intermediate results** (enhanced dataset, trained models)
7. **Visualize results** to communicate findings effectively

---

## 📞 Need Help?

1. Check **MODULES_SUMMARY.md** for detailed module documentation
2. Run **examples.py** to see working code samples
3. Explore **Quantum_Features.ipynb** for interactive walkthrough
4. Review function docstrings with `help(function_name)`

---

## ⚡ Performance Tips

- **Parallel Processing**: Use `n_jobs=-1` in tree-based models
- **GPU Acceleration**: TensorFlow will auto-detect GPUs for autoencoder
- **Batch Processing**: Process large datasets in chunks
- **Caching**: Save enhanced dataset to avoid regenerating quantum features

```python
# Save enhanced dataset
enhanced_df.to_csv('data/enhanced_cached.csv', index=False)

# Load later
df = pd.read_csv('data/enhanced_cached.csv')
```

---

**Last Updated**: 2024
**Version**: 1.0
