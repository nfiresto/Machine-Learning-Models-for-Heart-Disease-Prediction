# Imports
import kagglehub
import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, plot_tree   # Decision tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split  # for splitting dataset
from sklearn.tree import DecisionTreeClassifier, plot_tree   # Decision tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score  # for model evaluation

# Set the future version (I was getting a warning with the replace otherwise)
pd.set_option('future.no_silent_downcasting', True)

# Download latest version
path = kagglehub.dataset_download("oktayrdeki/heart-disease")

# Get the data into python
data = pd.read_csv(path + '\heart_disease.csv')
df = pd.DataFrame(data)

print(df.head())


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

# split the data into 60% training
train, testval = sklearn.model_selection.train_test_split(df, test_size=0.4, train_size=0.6, random_state=1)

# Split the remaining 40% into testing and validation so it's 20% each
test, val = sklearn.model_selection.train_test_split(testval, test_size=0.5, train_size=0.5, random_state=1)

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
clf = DecisionTreeClassifier()

# Train decision tree
clf = clf.fit(x_train, y_train)

# Visualize decision tree
colnames = df.columns.values
features_names = colnames[0:-1]  
# exclude last column header because it's the outcome label
target_name = ('Heart Disease', 'No Heart Disease')
fig = plt.figure(figsize=(15,10))
fig = plot_tree(clf,
                feature_names=features_names,
                class_names=target_name,
                filled=True)
#plt.show()

# Evaluation metrics
y_pred = clf.predict(x_test)  # prediction on the  set

# Confusion matrix
print('Confusion matrix: [[tn fp]')
print('                   [fn tp]]')
print(confusion_matrix(y_test, y_pred))

# Accuracy, precision, recall and F1-score
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="Yes")
recall  = recall_score(y_test, y_pred,  pos_label="Yes")
f1        = f1_score(y_test, y_pred, pos_label="Yes")
print('Accuracy:  ', accuracy)
print('Precision: ', precision)
print('Recall:    ', recall)
print('F1-score:  ', f1)

# Initial scores hover around 0.5, similar to KNN
# Now we will tune the tree depth

depths = [5,10,15, 20, 25, 30]

for depth in depths:
    print('Max Depth is', depth)
    # create the classifier
    clf = DecisionTreeClassifier(max_depth=depth)

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
    precision = precision_score(y_test, y_pred, pos_label="Yes")
    recall  = recall_score(y_test, y_pred,  pos_label="Yes")
    f1        = f1_score(y_test, y_pred, pos_label="Yes")
    print('Accuracy:  ', accuracy)
    print('Precision: ', precision)
    print('Recall:    ', recall)
    print('F1-score:  ', f1)

# Best Scores were for a max depth of 10
# Now we will tune the tree depth

splits = [5, 10, 15, 20, 25, 30]

for split in splits:
    print('Min split is', split)
    # create the classifier
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=split)

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
    precision = precision_score(y_test, y_pred, pos_label="Yes")
    recall  = recall_score(y_test, y_pred,  pos_label="Yes")
    f1        = f1_score(y_test, y_pred, pos_label="Yes")
    print('Accuracy:  ', accuracy)
    print('Precision: ', precision)
    print('Recall:    ', recall)
    print('F1-score:  ', f1)

# Best score is for a min split of 15
# Test these parameters on validation data

# create the classifier
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=15)

# Train decision tree
clf = clf.fit(x_train, y_train)

# Evaluation metrics
y_pred2 = clf.predict(x_val)  # prediction on the  set

# Confusion matrix
print('Confusion matrix: [[tn fp]')
print('                   [fn tp]]')
print(confusion_matrix(y_val, y_pred2))

# Accuracy, precision, recall and F1-score
accuracy  = accuracy_score(y_val, y_pred2)
precision = precision_score(y_val, y_pred2, pos_label="Yes")
recall  = recall_score(y_val, y_pred2,  pos_label="Yes")
f1        = f1_score(y_val, y_pred2, pos_label="Yes")
print('Accuracy:  ', accuracy)
print('Precision: ', precision)
print('Recall:    ', recall)
print('F1-score:  ', f1)

# Visualize decision tree
colnames = df.columns.values
features_names = colnames[0:-1]  
# exclude last column header because it's the outcome label
target_name = ('Heart Disease', 'No Heart Disease')
fig = plt.figure(figsize=(15,10))
fig = plot_tree(clf,
                feature_names=features_names,
                class_names=target_name,
                filled=True)
#plt.show()