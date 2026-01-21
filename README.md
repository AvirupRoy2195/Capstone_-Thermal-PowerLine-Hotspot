# 🔥 AI-Based Thermal Powerline Hotspot Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-SOTA-green.svg)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian-orange.svg)](https://optuna.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **State-of-the-Art Physics-Informed Machine Learning (PIML) pipeline for thermal anomaly detection in power transmission infrastructure using drone-based thermal inspection data.**

---

## 📋 Project Overview

This capstone project implements a **production-ready, statistically rigorous** anomaly detection system for identifying thermal hotspots in power lines and transmission towers. The system combines:

- **Physics-Informed Feature Engineering** (Joule's Law, Thermodynamics)
- **Bayesian Hyperparameter Optimization** (Optuna)
- **Advanced Feature Selection** (VIF, Correlation Analysis)
- **Statistical Reliability** (Bootstrap CI, Cross-Validation)
- **Explainable AI** (Permutation Importance)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION                               │
│  Thermal Powerline Dataset.xlsx (6000 tiles, 9 features)        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHYSICS-INFORMED FEATURE ENGINEERING               │
│  • delta_T = T_max - T_ambient                                  │
│  • load_norm_severity = ΔT / (load_factor² + ε)  [Joule's Law]  │
│  • thermal_gradient_intensity = edge_gradient × temp_std        │
│  • neighbor_zscore = Z-score of spatial anomaly                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE SELECTION                             │
│  1. VIF Analysis (remove multicollinearity, threshold=10)       │
│  2. Correlation Matrix (identify redundant features)            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│            BAYESIAN HYPERPARAMETER OPTIMIZATION                 │
│  • Optuna (50 trials, TPE sampler)                              │
│  • Optimized: n_estimators, max_depth, learning_rate,           │
│    subsample, colsample, min_child_weight, reg_alpha/lambda     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                COST-SENSITIVE XGBOOST                           │
│  • scale_pos_weight for class imbalance                         │
│  • F2-Score optimization (Recall-focused)                       │
│  • Stratified 5-Fold Cross-Validation                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              THRESHOLD & CALIBRATION                            │
│  • F2-Optimal threshold selection                               │
│  • Probability calibration (Platt scaling)                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT                                     │
│  • Thermal Risk Heatmap (Anomaly Corridors)                     │
│  • Maintenance Recommendations                                  │
│  • Bootstrap 95% Confidence Intervals                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Methodology

### 1. Physics-Informed Features
| Feature | Formula | Physical Meaning |
|---------|---------|------------------|
| `delta_T` | T_max - T_ambient | Temperature rise above ambient |
| `load_norm_severity` | ΔT / (I² + ε) | Joule's Law proxy (resistance faults) |
| `thermal_gradient_intensity` | edge_gradient × temp_std | Heat gradient intensity |
| `neighbor_zscore` | Z-score(delta_to_neighbors) | Spatial anomaly score |

### 2. Why F2-Score Over Accuracy?
> In safety-critical infrastructure, **missing a hotspot (False Negative) = fire risk**. F2-Score weights Recall 2× higher than Precision, ensuring minimal missed detections.

### 3. Hyperparameter Optimization
- **Method**: Optuna Bayesian (TPE Sampler)
- **Trials**: 50
- **Search Space**: 9 parameters including regularization

---

## 📁 Project Structure

```
Capstone_ThermalPowerline/
├── PIML_Thermal_Powerline_SOTA.ipynb   # Main notebook (all code + outputs)
├── Thermal Powerline Dataset.xlsx       # Input dataset
├── README.md                             # This file
└── requirements.txt                      # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna statsmodels openpyxl
```

### Run the Notebook
```bash
jupyter notebook PIML_Thermal_Powerline_SOTA.ipynb
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Recall** | See notebook |
| **F2-Score** | See notebook |
| **ROC-AUC** | See notebook |
| **Optimal Threshold** | F2-optimized |

---

## 📈 Visualizations

The notebook generates:
1. **Correlation Heatmap** - Feature relationships
2. **Optuna Optimization History** - Trial convergence
3. **Calibration Curves** - Probability reliability
4. **Permutation Importance** - Feature ranking
5. **Thermal Risk Heatmap** - Spatial anomaly corridors
6. **Confusion Matrix** - Classification performance

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **XGBoost** - Gradient Boosting
- **Optuna** - Bayesian Hyperparameter Optimization
- **Scikit-learn** - ML utilities
- **Statsmodels** - VIF calculation
- **Seaborn/Matplotlib** - Visualization

---

## 📝 License

This project is for educational purposes (Capstone Project).

---

## 👤 Author

AI-Based Thermal Powerline Hotspot Capstone Project
