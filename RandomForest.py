# Import libraries
import kagglehub
import numpy as np
import pandas as pd  # reading/managing dataset
import matplotlib.pyplot as plt  # plotting

# Functions from sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import column_or_1d


# Load data
# Download latest version
path = kagglehub.dataset_download("oktayrdeki/heart-disease")

# Get the data into python
data = pd.read_csv(path + '\heart_disease.csv')
df = pd.DataFrame(data)
print(df.head(5))

# There are 8000 Nos and 2000 Yes'

# Reduce the dataset to the last 4000 rows because there are only 2000 people with heart disease
# Trees work better when the dataset is balanced so we want there to be the same number of people
# with and without heart disease
df = df[6000:10000]

## Now, we want to remove any rows with missing values

# Remove rows with missing values (NaN)
df = df.dropna()

# Check how many rows are left
print('Rows after removing missing values:', df.shape[0])  # only returns nrow

# Turn words into numbers :)
df['Gender'] = df['Gender'].replace('Male',1)
df['Gender'] = df['Gender'].replace('Female',2)
df['Exercise Habits'] = df['Exercise Habits'].replace('Low',1)
df['Exercise Habits'] = df['Exercise Habits'].replace('Medium',2)
df['Exercise Habits'] = df['Exercise Habits'].replace('High',3)
df['Smoking'] = df['Smoking'].replace('No',0)
df['Smoking'] = df['Smoking'].replace('Yes',1)
df['Family Heart Disease'] = df['Family Heart Disease'].replace('No',0)
df['Family Heart Disease'] = df['Family Heart Disease'].replace('Yes',1)
df['Diabetes'] = df['Diabetes'].replace('No',0)
df['Diabetes'] = df['Diabetes'].replace('Yes',1)
df['High Blood Pressure'] = df['High Blood Pressure'].replace('No',0)
df['High Blood Pressure'] = df['High Blood Pressure'].replace('Yes',1)
df['Low HDL Cholesterol'] = df['Low HDL Cholesterol'].replace('No',0)
df['Low HDL Cholesterol'] = df['Low HDL Cholesterol'].replace('Yes',1)
df['High LDL Cholesterol'] = df['High LDL Cholesterol'].replace('No',0)
df['High LDL Cholesterol'] = df['High LDL Cholesterol'].replace('Yes',1)
df['Alcohol Consumption'] = df['Alcohol Consumption'].replace('None',0)
df['Alcohol Consumption'] = df['Alcohol Consumption'].replace('Low',1)
df['Alcohol Consumption'] = df['Alcohol Consumption'].replace('Medium',2)
df['Alcohol Consumption'] = df['Alcohol Consumption'].replace('High',3)
df['Stress Level'] = df['Stress Level'].replace('Low',1)
df['Stress Level'] = df['Stress Level'].replace('Medium',2)
df['Stress Level'] = df['Stress Level'].replace('High',3)
df['Sugar Consumption'] = df['Sugar Consumption'].replace('Low',1)
df['Sugar Consumption'] = df['Sugar Consumption'].replace('Medium',2)
df['Sugar Consumption'] = df['Sugar Consumption'].replace('High',3)
df['Heart Disease Status'] = df['Heart Disease Status'].replace('No',0)
df['Heart Disease Status'] = df['Heart Disease Status'].replace('Yes',1)

# split the data into 60% training
train, testval = train_test_split(df, test_size=0.4, train_size=0.6, random_state=1)

# Split the remaining 40% into testing and validation so it's 20% each
test, val = train_test_split(testval, test_size=0.5, train_size=0.5, random_state=1)

#Specify the inputs vs outputs
x_train = train[['Age','Gender','Blood Pressure','Cholesterol Level', 'Exercise Habits', 'Smoking',
              'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol',
               'Alcohol Consumption', 'Stress Level', 'Sleep Hours', 'Sugar Consumption', 
                'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level' ]]
y_train = train[['Heart Disease Status']]
x_test = test[['Age','Gender','Blood Pressure','Cholesterol Level', 'Exercise Habits', 'Smoking',
              'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol',
               'Alcohol Consumption', 'Stress Level', 'Sleep Hours', 'Sugar Consumption', 
                'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level' ]]
y_test = test[['Heart Disease Status']]
x_val = val[['Age','Gender','Blood Pressure','Cholesterol Level', 'Exercise Habits', 'Smoking',
              'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol',
               'Alcohol Consumption', 'Stress Level', 'Sleep Hours', 'Sugar Consumption', 
                'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level' ]]
y_val = val[['Heart Disease Status']]

# create the classifier
clf = RandomForestClassifier()

# Train decision tree
clf = clf.fit(x_train, y_train)

# Evaluation metrics
y_pred = clf.predict(x_test)  # prediction on the  set

# Confusion matrix
print('Confusion matrix: [[tn fp]')
print('                   [fn tp]]')
print(confusion_matrix(y_test, y_pred))

# Accuracy, precision, recall and F1-score
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, )
recall  = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
print('Accuracy:  ', accuracy)
print('Precision: ', precision)
print('Recall:    ', recall)
print('F1-score:  ', f1)

# Initial scores hover around 0.5, similar to previous
# Now we will tune the tree depth

param_grid = {'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
              'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
              'min_samples_split': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]}

model = RandomForestClassifier()

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, 
                           scoring='recall', verbose=3, return_train_score=True)

grid_search.fit(x_train, y_train.values.ravel())

best_model = grid_search.best_estimator_
print(best_model)


# Best score is max_Depth=5, min_samples_split=15, n_estimators=20
# Test these parameters on validation data

# create the classifier
clf = RandomForestClassifier(max_depth=5, min_samples_split=15, n_estimators=20)

# Train decision tree
clf = clf.fit(x_train, y_train)
()
# Evaluation metrics
y_pred2 = clf.predict(x_val)  # prediction on the  set

# Confusion matrix
print('Confusion matrix: [[tn fp]')
print('                   [fn tp]]')
print(confusion_matrix(y_val, y_pred2))

# Accuracy, precision, recall and F1-score
accuracy  = accuracy_score(y_val, y_pred2)
precision = precision_score(y_val, y_pred2)
recall  = recall_score(y_val, y_pred2)
f1        = f1_score(y_val, y_pred2)
print('Accuracy:  ', accuracy)
print('Precision: ', precision)
print('Recall:    ', recall)
print('F1-score:  ', f1)
