# Binary Classification Using Serial and Parallel Systems

## Team Members

* Muhammad Qasim (22I-1994)
* Ayaan Khan (22I-2066)
* Abu Bakr Nadeem (22I-2003)

---

## Overview

This project demonstrates the comparative analysis of optimized binary classification pipelines using XGBoost and PyTorch models across serial CPU, parallel CPU, and GPU compute environments. The focus is on evaluating performance improvements from parallelization and acceleration techniques while handling a noisy, imbalanced dataset.

### Key Goals:

* Preprocess a real-world tabular dataset (handling missing values, categorical encoding, normalization)
* Train XGBoost and PyTorch models in serial/parallel/GPU settings
* Compare models on speed and performance (accuracy, F1-score)
* Achieve at least 70% time reduction through hardware acceleration

---

## Dataset

**File:** `pdc_dataset_with_target.csv`

### Features:

* 4 Numerical: `feature_1`, `feature_2`, `feature_4`, `feature_7`
* 3 Categorical: `feature_3`, `feature_5`, `feature_6`
* Binary Target Label

### Stats:

* \~40,100 samples
* Moderate class imbalance

---

## Environment Setup

* **Language:** Python 3.12
* **Libraries:** pandas, scikit-learn, imbalanced-learn, XGBoost 1.7, PyTorch 2.0
* **Hardware:**

  * CPU: Intel Core i7 12th Gen (8 logical cores)
  * GPU: NVIDIA Tesla T4 (Google Colab)

---

## Data Preprocessing

* **Missing Value Imputation:**

  * Mean for symmetric features (1 & 2)
  * Median for skewed features (4 & 7)
* **Encoding:**

  * One-hot encoding for categorical features
* **Normalization:**

  * Quantile transformation for numerical features
* **Class Imbalance:**

  * SMOTE applied only on training set

### Final Shapes:

* Pre-SMOTE: (30750, 18)
* Post-SMOTE: (36990, 18)

---

## Model Architectures

### XGBoost Classifier

* **Serial:** `n_jobs=1`
* **Parallel:** `n_jobs=-1`
* **GPU:** `tree_method='gpu_hist'`, `predictor='gpu_predictor'`

### PyTorch Neural Network

* Layers: `[Input → 32 → 64 → 16 → 1]`
* Activations: ReLU + Sigmoid (output)
* Optimizer: Adam (lr = 0.001)
* Loss: BCEWithLogitsLoss
* Epochs: 50
* Batch Sizes: 64 (CPU), 1024 (GPU)
* GPU AMP: Enabled with GradScaler

---

## Results Summary

| Model Variant    | Accuracy | F1 Score | Training Time | Time Reduction |
| ---------------- | -------- | -------- | ------------- | -------------- |
| XGBoost Serial   | 0.5245   | 0.4327   | 0.61 sec      | —              |
| XGBoost Parallel | 0.5245   | 0.4327   | 0.38 sec      | 37.7%          |
| XGBoost GPU      | 0.5197   | 0.4321   | 0.57 sec      | 6.55%          |
| PyTorch CPU      | 0.5084   | 0.4513   | 119.37 sec    | —              |
| PyTorch GPU      | 0.4898   | 0.4592   | 37.34 sec     | 71.8%          |

### Key Takeaways

* **XGBoost Parallel CPU** gave best speed-performance balance.
* **PyTorch GPU** improved speed drastically (vs CPU) but slower than XGBoost overall.
* **PyTorch F1-scores** were slightly better, suggesting improved minority class handling.

---

## Performance Analysis

* **GPU Inefficiency in XGBoost**: Dataset too small (\~40k) for meaningful GPU speed-up due to data transfer and kernel launch overhead.
* **PyTorch Limitations**: Neural networks are resource-intensive, slower, and more sensitive to hyperparameters on tabular data.
* **Dataset Issues**: High noise, outliers, low semantic value of categories, class imbalance – all led to capped performance (\~52% accuracy).

---

## Comparative Insights

* **Best Time Reduction**: PyTorch GPU (71.8%)
* **Best Accuracy**: XGBoost Serial/Parallel (0.5245)
* **Best F1-Score**: PyTorch GPU (0.4592)
* **Worst Performer**: PyTorch CPU (slow and suboptimal accuracy)

---

## Future Directions

* Feature engineering improvements
* Ensemble methods combining trees and deep nets
* Larger datasets to leverage GPU benefits
* Advanced architectures (e.g. TabNet, Transformers for tabular data)
* Better hyperparameter tuning frameworks (Optuna, Ray Tune)

---

## References

* Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. SIGKDD.
* Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.

---

## Repository Structure Suggestion

```
parallel-tabular-modeling/
├── data/
│   └── pdc_dataset_with_target.csv
├── notebooks/
│   ├── xgboost_analysis.ipynb
│   └── pytorch_training.ipynb
├── models/
│   ├── xgb_model.pkl
│   └── pytorch_model.pth
├── src/
│   ├── preprocessing.py
│   ├── xgboost_runner.py
│   ├── pytorch_runner.py
├── README.md
└── requirements.txt
```

---

## How to Run

1. Clone the repo: `git clone https://github.com/yourusername/parallel-tabular-modeling`
2. Install dependencies: `pip install -r requirements.txt`
3. Run `notebooks/xgboost_analysis.ipynb` or `pytorch_training.ipynb`

---


