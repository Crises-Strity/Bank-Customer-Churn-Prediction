# Bank Customer Churn Prediction

This project aims to predict whether a bank customer will **churn (leave the bank)** based on their demographic, financial, and account activity features.  
It applies **machine learning classification models** — Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest — and performs **hyperparameter optimization** using `GridSearchCV`.

---

## 📘 Table of Contents

1. [Overview](#overview)  
2. [Dataset Description](#dataset-description)  
3. [Project Pipeline](#project-pipeline)  
4. [Technologies Used](#technologies-used)  
5. [Data Preprocessing](#data-preprocessing)  
6. [Modeling and Hyperparameter Tuning](#modeling-and-hyperparameter-tuning)  
7. [Model Evaluation](#model-evaluation)  
8. [Results](#results)  
9. [How to Run](#how-to-run)  
10. [Future Improvements](#future-improvements)  
11. [License](#license)

---

## 🧭 Overview

Customer churn is a major problem in the banking sector, directly affecting profitability and long-term growth.  
The goal of this project is to identify customers who are most likely to **exit** based on their characteristics and transaction history.

The pipeline includes:
- Exploratory data analysis (EDA)
- Feature encoding (One-Hot, Ordinal)
- Feature scaling (StandardScaler)
- Model training and hyperparameter tuning
- Model evaluation with ROC-AUC, accuracy, and other metrics

---

## 📊 Dataset Description

The dataset `bank.data.csv` contains approximately **10,000 customers** and includes the following features:

| Feature | Description |
|----------|--------------|
| `CreditScore` | Customer credit score |
| `Geography` | Country (France, Spain, Germany) |
| `Gender` | Male/Female |
| `Age` | Customer age |
| `Tenure` | Years of relationship with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Whether the customer has a credit card |
| `IsActiveMember` | Whether the customer is an active member |
| `EstimatedSalary` | Estimated yearly salary |
| `Exited` | Target variable — 1 if customer churned, 0 otherwise |

---

## ⚙️ Project Pipeline

1. **Data Loading**  
   Read dataset using `pandas`:
   ```python
   initial_data = pd.read_csv("bank.data.csv")
