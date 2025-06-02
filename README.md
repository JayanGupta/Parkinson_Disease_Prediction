# Parkinson's Disease Prediction using Machine Learning

This repository contains a Jupyter Notebook that demonstrates the process of predicting Parkinson’s Disease using machine learning techniques. The notebook walks through data preprocessing, feature selection, model training, and evaluation using Python.

## Overview

Parkinson’s Disease is a progressive nervous system disorder. Early diagnosis can help manage symptoms and improve quality of life. This notebook applies supervised learning models to predict the presence of Parkinson’s Disease based on various biomedical voice measurements.

Key tasks performed in this notebook include:

- Data loading and cleaning
- Exploratory data analysis
- Feature scaling and selection
- Model training and validation using:
  - Support Vector Machine (SVM)
  - XGBoost Classifier
  - Logistic Regression
- Evaluation using metrics such as accuracy and confusion matrix

## Dataset

The dataset used is `parkinson_disease.csv`, which includes multiple recordings per subject. Key features include a range of vocal measurements extracted from speech signals.

## Requirements

This project requires the following Python libraries:

- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  
- xgboost  
- imbalanced-learn  
- tqdm

Install the required packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn tqdm
