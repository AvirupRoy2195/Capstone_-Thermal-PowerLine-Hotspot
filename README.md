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
- **Stacking Ensemble Model** (XGBoost, RandomForest, GradientBoosting)
- **Bayesian Hyperparameter Optimization** (Optuna)
- **Advanced Feature Selection** (VIF, Correlation Analysis)
- **Statistical Reliability** (Bootstrap CI, Cross-Validation)
- **Explainable AI** (LIME, SHAP)
- **Spatial Risk Analysis** (Corridor Aggregation & Drone Planning)

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
│  • Core: delta_T, load_norm_severity, thermal_gradient_intensity│
│  • Advanced: relative_hotspot, neighbor_zscore, combined_severity│
│  • 12+ Physics-derived features created                         │
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
│  • Optuna (50 trials, TPE sampler) for XGBoost                  │
│  • Optimized: n_estimators, max_depth, learning_rate, etc.      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                STACKING ENSEMBLE MODEL                          │
│  • Base Learners: XGBoost, RandomForest, GradientBoosting       │
│  • Meta Learner: LogisticRegression                             │
│  • SMOTE for Class Imbalance Handling                           │
│  • Stratified 5-Fold Cross-Validation                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              THRESHOLD & CALIBRATION                            │
│  • F2-Optimal threshold selection (0.1324)                      │
│  • Probability calibration                                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT                                     │
│  • Thermal Risk Heatmap (Anomaly Corridors)                     │
│  • Maintenance Recommendations (Severity Levels)                │
│  • Drone Flight Sequence Plan                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Methodology

### 1. Physics-Informed Features
The model uses 12+ engineered features based on thermal physics:

| Feature | Formula | Physical Meaning |
|---------|---------|------------------|
| `delta_T` | T_max - T_ambient | Temperature rise above ambient |
| `load_norm_severity` | ΔT / (I² + ε) | Joule's Law proxy (resistance faults) |
| `thermal_gradient_intensity` | edge_gradient × temp_std | Heat gradient intensity |
| `neighbor_zscore` | Z-score(delta_to_neighbors) | Spatial anomaly score |
| `relative_hotspot` | hotspot_fraction * T_max / T_mean | Normalized hotspot severity |
| `combined_severity` | load_norm_severity * neighbor_zscore | Multi-factor risk index |

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
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook PIML_Thermal_Powerline_SOTA.ipynb
```

---

## 📊 Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Recall** | **0.8842** | Priority metric (Fault Detection Rate) |
| **F2-Score** | **0.8026** | Optimized objective |
| **Precision** | 0.5862 | Trade-off for high recall |
| **ROC-AUC** | 0.8709 | Overall discriminative power |
| **Optimal Threshold** | 0.1324 | Selected to minimize False Negatives |

---

## 📈 Visualizations

The notebook generates comprehensive insights:
1.  **Correlation Heatmap** - Feature relationships
2.  **Optuna Optimization History** - Trial convergence
3.  **LIME Explanations** - Local feature importance for individual tiles
4.  **SHAP Summary** - Global feature impact
5.  **Spatial Risk Heatmaps** - Tile-level and Zone-level risk aggregation
6.  **Hotspot Clusters** - Identification of connected fault regions
7.  **Drone Flight Plan** - Prioritized inspection sequence

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **XGBoost, GradientBoosting, RandomForest** - Ensemble Learning
- **Optuna** - Bayesian Hyperparameter Optimization
- **Imbalanced-learn (SMOTE)** - Handling Class Imbalance
- **LIME & SHAP** - Explainable AI (XAI)
- **Scikit-learn** - ML utilities
- **Statsmodels** - VIF calculation
- **Seaborn/Matplotlib** - Visualization

---

## 📝 License

This project is for educational purposes (Capstone Project).

---

## 👤 Author

AI-Based Thermal Powerline Hotspot Capstone Project
