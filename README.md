# Breast Cancer Classification

## a. Problem Statement

The objective is to classify a breast tumor as **malignant** or **benign** and compare five machine learning classification models.

## b. Dataset Description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository, accessed through scikit-learn
- **Instances:** 569
- **Features:** 30 numeric features
- **Classes:** Malignant (0) and Benign (1)
- **Train-test split:** 80% training and 20% testing

## c. GitHub Repository Link

https://github.com/shiv-kobgapu/ml-breast-cancer-classification

## d. Models Used and Results

The following models were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9234 | 0.9211 | 0.9216 | 0.8341 |
| kNN | 0.9561 | 0.9788 | 0.9561 | 0.9561 | 0.9560 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9298 | 0.9298 | 0.9298 | 0.8492 |
| Random Forest | 0.9561 | 0.9931 | 0.9561 | 0.9561 | 0.9560 | 0.9054 |

## Model Performance Observations

| ML Model Name | Observation |
|---|---|
| Logistic Regression | It achieved the best overall results. |
| Decision Tree | It was easy to understand but had the lowest performance. |
| kNN | It performed well after scaling the features. |
| Naive Bayes | It was fast and produced a strong AUC score. |
| Random Forest | It performed better than the single Decision Tree. |
| Overall Winner | **Logistic Regression** was the winner with 0.9825 accuracy. |

## Streamlit App Link


