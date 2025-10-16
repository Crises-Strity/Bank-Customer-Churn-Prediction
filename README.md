# Bank Customer Churn Prediction

A machine learning project for predicting bank customer churn using classification models (Logistic Regression, K-Nearest Neighbors, Random Forest), with hyperparameter tuning (GridSearchCV) to optimize model performance.  

This repository implements a quantitative framework to train, evaluate, and compare models on a financial dataset of ~10,000 records.  

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Installation & Setup](#installation--setup)  
5. [Usage](#usage)  
6. [Model Training & Hyperparameter Tuning](#model-training--hyperparameter-tuning)  
7. [Evaluation & Metrics](#evaluation--metrics)  
8. [Results & Findings](#results--findings)  
9. [Contributing](#contributing)  
10. [License](#license)  
11. [Acknowledgements](#acknowledgements)  

---

## Project Overview

Churn prediction is critical for banks to identify customers at risk of leaving (churning) and take proactive retention actions. This project:

- Ingests a cleaned financial dataset of 10,000 bank customers  
- Performs exploratory data analysis (EDA) and preprocessing  
- Trains three classification models:
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  
  - Random Forest  
- Utilizes **GridSearchCV** to optimize hyperparameters  
- Compares model performance and selects the best model  
- (Optional) Saves the best model for inference

The goal is to build a robust pipeline from data to model evaluation and selection.

---

## Features

- Data loading and cleaning  
- Feature engineering and encoding  
- Train / test splitting  
- Hyperparameter tuning with cross-validation  
- Model evaluation (accuracy, precision, recall, F1, ROC-AUC)  
- Visualization of results (e.g. ROC curves, confusion matrices)  
- Modular design in scripts or notebooks  

---

## Project Structure

Here’s a typical project layout (adapted to this repo):

Bank-Customer-Churn-Prediction/
├── data/
│ ├── raw/ # raw / original data (if included)
│ ├── processed/ # cleaned / processed data
│ └── …
├── notebooks/ # Jupyter notebooks for exploration and prototyping
│ └── churn_analysis.ipynb
├── src/ # source code modules / scripts
│ └── project_code/
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── evaluation.py
│ └── utils.py
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md


- `data/` – datasets (raw and processed)  
- `notebooks/` – for EDA, visualization, prototyping  
- `src/project_code/` – the production or structured Python scripts  
- `requirements.txt` – Python dependencies  
- `LICENSE` – license file  
- `README.md` – project introduction and instructions  

---

## Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Crises-Strity/Bank-Customer-Churn-Prediction.git
   cd Bank-Customer-Churn-Prediction
(Recommended) Create a virtual environment


python3 -m venv venv
source venv/bin/activate         # Linux / macOS
# or venv\Scripts\activate       # Windows
Install dependencies

bash
复制代码
pip install -r requirements.txt
Note: The requirements.txt is derived from import statements; if you find missing packages, add them and re-freeze.

Ensure data is placed correctly

Place the dataset(s) under data/raw/ or data/processed/ as expected by scripts

Adjust file paths in configuration or scripts if necessary

Usage
Here is a sample usage flow:

Preprocessing / Feature Engineering

bash
复制代码
python src/project_code/data_preprocessing.py
Training & Hyperparameter Tuning

bash
复制代码
python src/project_code/model_training.py
Evaluation & Visualization

bash
复制代码
python src/project_code/evaluation.py
You may also run the Jupyter notebook for exploratory analysis:

bash
复制代码
jupyter notebook notebooks/churn_analysis.ipynb
Model Training & Hyperparameter Tuning
Train/Test Split: split the cleaned dataset into training and testing sets, e.g. 70/30 or 80/20

Cross-Validation: use cross validation (e.g. 5-fold) within GridSearchCV

Models & Hyperparameters

Model	Hyperparameters to tune
Logistic Regression	penalty (l1, l2), C (regularization strength)
K-Nearest Neighbors	n_neighbors, weights, metric
Random Forest	n_estimators, max_depth, min_samples_split, min_samples_leaf

GridSearchCV: for each model, define a grid of hyperparameter values and search for the best set using cross-validation

Best Model Selection: pick the model (and hyperparameters) achieving highest performance (e.g. via ROC-AUC or F1) on validation sets

Evaluation & Metrics
To evaluate models, common metrics used include:

Accuracy

Precision / Recall / F1-score

ROC AUC

Confusion Matrix

ROC Curve / Precision-Recall Curve plots

Visualization and numeric summaries help compare models side by side.

Results & Findings
(You should fill this section based on your actual experiments and findings.)

Which model performed best (e.g. Random Forest with tuned parameters)

Key metrics (e.g. test set accuracy, AUC, recall, etc.)

Feature importance (e.g. top features influencing churn)

Observations from misclassifications

Suggestions or limitations (e.g. class imbalance, data quality, overfitting risk)

You may also include sample plots (ROC curves, feature importance bar charts, confusion matrix) in the notebook or embed here via links.

Contributing
Contributions are welcome! Here are some suggested ways:

Add more classifiers (e.g. XGBoost, LightGBM, SVM)

Improve feature engineering (e.g. feature interactions, dimensionality reduction)

Deal with class imbalance (e.g. SMOTE, undersampling)

Add model serialization / API for inference

Improve documentation, testing, or modularization

If you’d like to contribute, please fork the repository, create a feature branch, and open a pull request.

License
This project is licensed under the MIT License — see the LICENSE file for details. 
GitHub

Acknowledgements
Inspiration and structure from common churn-prediction tutorials

Libraries such as scikit-learn, pandas, numpy, matplotlib / seaborn

Any data sources or benchmarks you used

