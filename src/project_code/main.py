# Auto-generated from notebook: Project 1.ipynb
# Exported on 2025-10-09T14:20:12.057673Z

# Original language: python

# --- Cell 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn import metrics

# --- Cell 2 ---
# Introduce the used data
initial_data = pd.read_csv('bank.data.csv')
initial_data.head()

# --- Cell 3 ---
# Check the information of data
initial_data.info()

# --- Cell 4 ---
# Chech the unique value for each category(column) in the data
initial_data.nunique()

# --- Cell 5 ---
# Here choose the 'Exited' as the target variable, therefore, this is try to predict value of 'Exited' with other variables under given value of 'Exited'
y = initial_data['Exited']
y

# --- Cell 6 ---
# Try to find if there are some missing values
initial_data.isnull().sum()

# --- Cell 7 ---
# Find numerical features
# They are 'CreditScore', 'Age', 'Tenure', 'NumberOfProducts', 'Balance', 'EstimatedSalary'
initial_data[['CreditScore', 'Age', 'Tenure', 'NumOfProducts','Balance', 'EstimatedSalary']].describe()

# --- Cell 8 ---
# Then use boxplot to find the difference of distribution of exited and non-exited bank customers towards different features such as 'CreditScore'

fig, axss = plt.subplots(2,3, figsize=[20,10])

sns.boxplot(x='Exited', y ='CreditScore', data=initial_data, ax=axss[0][0])
sns.boxplot(x='Exited', y ='Age', data=initial_data, ax=axss[0][1])
sns.boxplot(x='Exited', y ='Tenure', data=initial_data, ax=axss[0][2])
sns.boxplot(x='Exited', y ='NumOfProducts', data=initial_data, ax=axss[1][0])
sns.boxplot(x='Exited', y ='Balance', data=initial_data, ax=axss[1][1])
sns.boxplot(x='Exited', y ='EstimatedSalary', data=initial_data, ax=axss[1][2])

numerical_categories_name = ['CreditScore', 'Age', 'Tenure', 'NumberOfProducts', 'Balance', 'EstimatedSalary']
for i, ax in enumerate(axss.flat):
    ax.set_title(list(numerical_categories_name)[i])
plt.tight_layout()
plt.show()

# --- Cell 9 ---
# Find categorical features
# They are 'Geography', 'Gender', 'HasCrCard', 'IsActiveMember'
initial_data[['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']]

# --- Cell 10 ---
# Then use boxplot to find the difference of distribution of exited and non-exited bank customers towards different categorical features such as 'RowNumber'
fig, axss = plt.subplots(2,2, figsize=[20,10])

sns.countplot(x='Exited', hue='Geography', data=initial_data, ax=axss[0][0])
sns.countplot(x='Exited', hue='Gender', data=initial_data, ax=axss[0][1])
sns.countplot(x='Exited', hue='HasCrCard', data=initial_data, ax=axss[1][0])
sns.countplot(x='Exited', hue='IsActiveMember', data=initial_data, ax=axss[1][1])

categorical_feature_name = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
for j, ax in enumerate(axss.flat):
    ax.set_title(list(categorical_feature_name)[j])
plt.tight_layout()
plt.show()

# --- Cell 11 ---
# Drop useless features such as customers' id and the target variable
to_drop = ['RowNumber','CustomerId','Surname','Exited']
X = initial_data.drop(to_drop, axis = 1)
X.head()

# --- Cell 12 ---
X.dtypes

# --- Cell 13 ---
# Divide the data space into 2 parts: numerical and categorical
cat_cols = X.columns[X.dtypes == 'object']
num_cols = X.columns[(X.dtypes == 'float64') | (X.dtypes == 'int64')]
print(cat_cols)
print(num_cols)

# --- Cell 14 ---
# Here split the data into training data(75%) and testing data(25%) 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, stratify = y, random_state = 1)

print('training data has ' + str(X_train.shape[0]) + ' observation with ' + str(X_train.shape[1]) + ' features')
print('test data has ' + str(X_test.shape[0]) + ' observation with ' + str(X_test.shape[1]) + ' features')

# --- Cell 15 ---
X_train.head()

# --- Cell 16 ---
# Here use one hot encoding to transform the 'Geography' into 3 different features with value of only 1 or 0 (numerical)
# Under 'Geography' feature, all customers have 3 countries Spain, Germany, and France, therefore 3 features of respective 3 countries
# In detail, the Geography of one customer is shown with 1 under this feature indicating this country and 0 indicating not this country
def OneHotEncoding(df, enc, categories):
  transformed = pd.DataFrame(enc.transform(df[categories]).toarray(), columns = enc.get_feature_names_out(categories))
  return pd.concat([df.reset_index(drop=True), transformed], axis=1).drop(categories, axis=1)

categories = ['Geography']
enc_ohe = OneHotEncoder()
enc_ohe.fit(X_train[categories])

X_train = OneHotEncoding(X_train, enc_ohe, categories)
X_test = OneHotEncoding(X_test, enc_ohe, categories)

# --- Cell 17 ---
# Here use ordinal encoding to transform the 'Gender' into numerical feature
# Because there are only 2 type of gender(male or female) in the data, therefore could use 1 to represent one and 0 for another
categories = ['Gender']
enc_oe = OrdinalEncoder()
enc_oe.fit(X_train[categories])

X_train[categories] = enc_oe.transform(X_train[categories])
X_test[categories] = enc_oe.transform(X_test[categories])

# --- Cell 18 ---
# Finally, there are only numerical data in the whole data space
X_train.head()

# --- Cell 19 ---
# Here standardize the data to avoid the effect of magnitudes and units
scaler = StandardScaler()
scaler.fit(X_train[num_cols])

X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

X_train.head()

# --- Cell 20 ---
classifier_logistic = LogisticRegression()
classifier_logistic.fit(X_train, y_train)
classifier_logistic.predict(X_test)
classifier_logistic.score(X_test, y_test)

# --- Cell 21 ---
classifier_KNN = KNeighborsClassifier()
classifier_KNN.fit(X_train, y_train)
classifier_KNN.predict(X_test)
classifier_KNN.score(X_test, y_test)

# --- Cell 22 ---
classifier_RF = RandomForestClassifier()
classifier_RF.fit(X_train, y_train)
classifier_RF.predict(X_test)
classifier_RF.score(X_test, y_test)

# --- Cell 23 ---
# Define the helper function for printing out grid search results
def print_grid_search_metrics(gs):
    print ("Best score: " + str(gs.best_score_))
    print ("Best parameters set:")
    best_parameters = gs.best_params_
    for param_name in sorted(best_parameters.keys()):
        print(param_name + ':' + str(best_parameters[param_name]))

# --- Cell 24 ---
# Possible hyperparamter options for Logistic Regression Regularization are: 
# Penalty (L1 or L2) and C is the 1/lambda value(weight)
parameters = {
    'penalty':('l2','l1'),
    'C':(0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1)
}

Grid_LR = GridSearchCV(LogisticRegression(solver='liblinear'),parameters, cv = 5)
Grid_LR.fit(X_train, y_train)

print_grid_search_metrics(Grid_LR)

# --- Cell 25 ---
best_LR_model = Grid_LR.best_estimator_
best_LR_model.predict(X_test)
best_LR_model.score(X_test, y_test)

# --- Cell 26 ---
# Use thermodynamic diagram to find the change of score with the change of parameter penalty and C
LR_models = pd.DataFrame(Grid_LR.cv_results_)
LR_models['param_C'] = pd.to_numeric(LR_models['param_C'], errors='coerce')
res = (LR_models.pivot(index='param_penalty', columns='param_C', values='mean_test_score'))
_ = sns.heatmap(res, cmap='viridis')

# --- Cell 27 ---
# Possible hyperparamter options for KNN is only k
parameters = {
    'n_neighbors':[1,3,5,7,9]
}

Grid_KNN = GridSearchCV(KNeighborsClassifier(),parameters, cv=5)
Grid_KNN.fit(X_train, y_train)

print_grid_search_metrics(Grid_KNN)

# --- Cell 28 ---
best_KNN_model = Grid_KNN.best_estimator_
best_KNN_model.predict(X_test)
best_KNN_model.score(X_test, y_test)

# --- Cell 29 ---
# Possible hyperparamter options for Random Forest are number and depth
# Choose the number of trees and the maximum depth
parameters = {
    'n_estimators' : [60,70,80,90,100],
    'max_depth': [1,5,10]
}

Grid_RF = GridSearchCV(RandomForestClassifier(),parameters, cv=5)
Grid_RF.fit(X_train, y_train)

print_grid_search_metrics(Grid_RF)

# --- Cell 30 ---
best_RF_model = Grid_RF.best_estimator_
best_RF_model.predict(X_test)
best_RF_model.score(X_test, y_test)

# --- Cell 31 ---
# Use thermodynamic diagram to find the change of score with the change of parameter number and depth
LR_models = pd.DataFrame(Grid_RF.cv_results_)
LR_models['param_n_estimators'] = pd.to_numeric(LR_models['param_n_estimators'], errors='coerce')
LR_models['param_max_depth'] = pd.to_numeric(LR_models['param_max_depth'], errors='coerce')
res = (LR_models.pivot(index='param_max_depth', columns='param_n_estimators', values='mean_test_score'))
_ = sns.heatmap(res, cmap='viridis')

# --- Cell 32 ---
# Define function to calculate the accuracy, precision, and recall of model
def cal_evaluation(classifier, cm):
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 0.0)
    precision = tp / (tp + fp + 0.0)
    recall = tp / (tp + fn + 0.0)
    print (classifier)
    print ("Accuracy is: " + str(accuracy))
    print ("precision is: " + str(precision))
    print ("recall is: " + str(recall))
    print ()

# Define function to print out confusion matrices
def draw_confusion_matrices(confusion_matricies):
    class_names = ['Not','Churn']
    for cm in confusion_matricies:
        classifier, cm = cm[0], cm[1]
        cal_evaluation(classifier, cm)

# --- Cell 33 ---
# Show the confusion matrix, accuracy, precison and recall for random forest, logistic regression, and k nearest neighbor
confusion_matrices = [
    ("Random Forest", confusion_matrix(y_test,best_RF_model.predict(X_test))),
    ("Logistic Regression", confusion_matrix(y_test,best_LR_model.predict(X_test))),
    ("K nearest neighbor", confusion_matrix(y_test, best_KNN_model.predict(X_test)))
]

draw_confusion_matrices(confusion_matrices)

# --- Cell 34 ---
# Use predict_proba to get the probability results of Logistic Regression
y_pred_lr = best_LR_model.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, thresh = roc_curve(y_test, y_pred_lr)
best_LR_model.predict_proba(X_test)

# --- Cell 35 ---
# ROC Curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_lr, tpr_lr, label='LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - LR Model')
plt.legend(loc='best')
plt.show()

# --- Cell 36 ---
# AUC score
metrics.auc(fpr_lr,tpr_lr)

# --- Cell 37 ---
# Use predict_proba to get the probability results of K Nearest Neighbors
y_pred_knn = best_KNN_model.predict_proba(X_test)[:, 1]
fpr_knn, tpr_knn, thresh = roc_curve(y_test, y_pred_knn)
best_KNN_model.predict_proba(X_test)

# --- Cell 38 ---
# ROC Curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_knn, tpr_knn, label='KNN')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - LR Model')
plt.legend(loc='best')
plt.show()

# --- Cell 39 ---
# AUC score
metrics.auc(fpr_knn,tpr_knn)

# --- Cell 40 ---
# Use predict_proba to get the probability results of Random Forest
y_pred_rf = best_RF_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
best_RF_model.predict_proba(X_test)

# --- Cell 41 ---
# ROC curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - RF model')
plt.legend(loc='best')
plt.show()

# --- Cell 42 ---
# AUC score
metrics.auc(fpr_rf,tpr_rf)

# --- Cell 43 ---
X_with_corr = X.copy()

X_with_corr = OneHotEncoding(X_with_corr, enc_ohe, ['Geography'])
X_with_corr['Gender'] = enc_oe.transform(X_with_corr[['Gender']])
X_with_corr['SalaryInRMB'] = X_with_corr['EstimatedSalary'] * 6.4
X_with_corr.head()

# --- Cell 44 ---
# add L2 regularization to logistic regression
# check the coef for feature selection
np.random.seed()
scaler = StandardScaler()
X_l2 = scaler.fit_transform(X_with_corr)
LRmodel_l2 = LogisticRegression(penalty="l2", C = 0.1, solver='liblinear', random_state=42)
LRmodel_l2.fit(X_l2, y)
LRmodel_l2.coef_[0]

indices = np.argsort(abs(LRmodel_l2.coef_[0]))[::-1]

print ("Logistic Regression (L2) Coefficients")
for ind in range(X_with_corr.shape[1]):
  print ("{0} : {1}".format(X_with_corr.columns[indices[ind]],round(LRmodel_l2.coef_[0][indices[ind]], 4)))

# --- Cell 45 ---
X_RF = X.copy()

X_RF = OneHotEncoding(X_RF, enc_ohe, ['Geography'])
X_RF['Gender'] = enc_oe.transform(X_RF[['Gender']])

X_RF.head()

# --- Cell 46 ---
# check feature importance of random forest for feature selection
forest = RandomForestClassifier()
forest.fit(X_RF, y)

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature importance ranking by Random Forest Model:")
for ind in range(X.shape[1]):
  print ("{0} : {1}".format(X_RF.columns[indices[ind]],round(importances[indices[ind]], 4)))

