# # Assignment No:- 2
# # Predict people suffering from heart issues
# # Name:- Raj Verma
# Roll No:-210029010017

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

nhanes_df = pd.read_csv('your_dataset.csv')

# Data preprocessing
# Implement data cleaning, feature selection, and encoding here

# Extract relevant features and target variable
features = ['Age', 'Gender', 'OtherFeatures']
target = 'HeartCondition'

# Convert categorical variables to numerical
nhanes_df['Gender'] = nhanes_df['Gender'].map({'Male': 0, 'Female': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(nhanes_df[features], nhanes_df[target], test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model selection and tuning
# Logistic Regression
logreg = LogisticRegression()
param_grid_logreg = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_logreg = GridSearchCV(logreg, param_grid_logreg, cv=5)
grid_logreg.fit(X_train_scaled, y_train)

# Support Vector Machine (SVM)
svm = SVC(probability=True)
param_grid_svm = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5)
grid_svm.fit(X_train_scaled, y_train)

# Decision Tree
dt = DecisionTreeClassifier()
param_grid_dt = {'max_depth': [None, 10, 20, 30, 40, 50], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid_dt = GridSearchCV(dt, param_grid_dt, cv=5)
grid_dt.fit(X_train_scaled, y_train)

# K-Nearest Neighbors (K-NN)
knn = KNeighborsClassifier()
param_grid_knn = {'n_neighbors': [3, 5, 7, 9, 11]}
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5)
grid_knn.fit(X_train_scaled, y_train)

# Evaluation
models = {'Logistic Regression': grid_logreg, 'SVM': grid_svm, 'Decision Tree': grid_dt, 'K-NN': grid_knn}

for model_name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(f'\n{model_name} Performance:')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f'Recall: {recall_score(y_test, y_pred)}')
    print(f'F1 Score: {f1_score(y_test, y_pred)}')
    print(f'AUC: {roc_auc_score(y_test, y_proba)}')