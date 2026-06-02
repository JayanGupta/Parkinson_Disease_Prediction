<div align="center">
  
  # Parkinson’s Disease Prediction using Machine Learning

  [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![XGBoost](https://img.shields.io/badge/XGBoost-AA4A44?style=for-the-badge)](https://xgboost.readthedocs.io)
  [![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)]()
  [![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)]()

  *A machine learning project for early prediction of Parkinson’s Disease using biomedical voice measurements.*
</div>

---

## Background

Parkinson's Disease (PD) is a progressive neurological disorder affecting motor control. Early-stage diagnosis remains clinically challenging. Research has shown that sustained phonation and connected speech yield measurable acoustic irregularities in PD patients — enabling non-invasive, data-driven screening.

This project applies classical machine learning to a dataset of biomedical voice measurements, with the goal of building a reliable binary classifier that distinguishes PD patients from healthy controls.

---

## Dataset

| Property | Detail |
|----------|--------|
| File | `parkinson_disease.csv` |
| Domain | Biomedical voice measurements |
| Task | Binary classification (PD vs. healthy) |
| Features | Acoustic measures: jitter, shimmer, HNR, RPDE, DFA, spread, PPE |
| Samples | Multiple recordings per subject |

Features are derived from sustained phonation recordings and capture vocal tremor, noise-to-harmonics ratio, and nonlinear dynamical complexity — all known biomarkers for PD.

---

## Project Structure

```
.
├── Parkinson_Disease_Prediction_using_Machine_Learning.ipynb   # Main analysis notebook
├── parkinson_disease.csv                                        # Raw dataset
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.8+

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn tqdm
```

All experiments are self-contained within the Jupyter notebook. No additional configuration is required.

---

## Methodology

The pipeline follows a structured ML workflow:

**1. Data Preprocessing**
- Null value audit and type validation
- Removal of non-informative identifiers (subject name, recording index)
- Class distribution analysis and SMOTE-based oversampling to address imbalance

**2. Exploratory Data Analysis**
- Correlation heatmap to identify feature redundancy
- Distribution plots across PD vs. healthy cohorts
- Pairwise feature relationships for key biomarkers

**3. Feature Engineering & Scaling**
- StandardScaler normalization applied to all continuous features
- Feature importance ranking via XGBoost's built-in scorer
- Removal of low-variance and highly correlated features

**4. Model Training**
- Stratified train/test split (80/20) to preserve class ratios
- Cross-validated hyperparameter selection

**5. Evaluation**
- Accuracy, precision, recall, F1-score
- Confusion matrix for error analysis
- Side-by-side model comparison

---

## Models & Results

Three classifiers were trained and evaluated:

| Model | Notes |
|-------|-------|
| Support Vector Machine (SVM) | RBF kernel; performs well on high-dimensional acoustic feature spaces |
| XGBoost Classifier | Gradient-boosted trees; provides native feature importance scores |
| Logistic Regression | Linear baseline; interpretable coefficient-level insights |

> Detailed accuracy scores, confusion matrices, and per-class metrics are reported in the notebook.

---

## Usage

1. Clone the repository and install dependencies (see [Installation](#installation)).
2. Open `Parkinson_Disease_Prediction_using_Machine_Learning.ipynb` in Jupyter.
3. Run all cells sequentially — each section is annotated with its purpose and outputs.

The notebook is structured to be reproduced end-to-end without manual intervention between steps.

---

## Future Work

- **Hyperparameter optimisation** — systematic grid/random search for SVM and XGBoost
- **Deep learning baselines** — 1D CNN or LSTM over raw MFCC features as a comparison point
- **Model deployment** — REST API via FastAPI or an interactive demo via Streamlit
- **Expanded dataset** — integration with mPower or PC-GITA for cross-corpus generalisation
- **Explainability** — SHAP value analysis to surface clinically interpretable feature contributions

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request. Ensure any new code is accompanied by clear inline documentation.

---

## License

This project is released for academic and research purposes. See `LICENSE` for details.
