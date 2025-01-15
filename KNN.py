import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split  # for splitting dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler      # for normalizing/scaling features
from sklearn.neighbors import KNeighborsClassifier    # KNN
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score  # for model evaluation

# Set the future version (I was getting a warning with the replace otherwise)
pd.set_option('future.no_silent_downcasting', True)

# Download latest version
path = kagglehub.dataset_download("oktayrdeki/heart-disease")

# Get the data into python
data = pd.read_csv(path + '\heart_disease.csv')
df = pd.DataFrame(data)
#print(df.head(5))
#print(df.shape)

## KNN Figure 1
# Plot each variable against every other variable to see if preexisting relationships exist
'''
pd.plotting.scatter_matrix(df)
plt.show()
'''

# Okay so the mega plot shows no bias or relationship between any of the factors. 
# The plot is kind of scary and very full because of the 1000 data samples, i imagine. 


## KNN Figure 2
# plot the yeses and nos to see if its balance
'''
plt.plot(df['Heart Disease Status'])
plt.xlabel('Heart Disease')
plt.ylabel('Count');
plt.show()
'''
# There are 8000 Nos and 2000 Yes'

# Reduce the dataset to the last 4000 rows because there are only 2000 people with heart disease
# KNN works better when the dataset is balanced so we want there to be the same number of people
# with and without heart disease
df = df[6000:10000]

## Now, we want to remove any rows with missing values

# Remove rows with missing values (NaN)
df = df.dropna()

# Check how many rows are left
print('Rows after removing missing values:', df.shape[0])  # only returns nrow

# Okay so we still have enough data now so we can proceed

## Switch strings to integers (Male->1, Female ->2, Low->1, Med->2)
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

# this is a lot of changes and I don't know if it will have positive or negative effects
# Later, I am going to try running the KNN model after removing these columns to examine the effects

## Split the Data
X = df.iloc[:, :-1]  # all rows, 1st to 2nd-to-last col -- features
y = df.iloc[:, -1]   # all rows, last col -- outcome labels

# Split X and y into training (60%), cross val (20%), and test (20%) sets.
# also randomize so the distribution of yeses and nos in the test/train is more equal
X_train, X_orig, y_train, y_orig = train_test_split(X, y, test_size=0.4, random_state=37)
X_crossval, X_test, y_crossval, y_test = train_test_split(X_orig, y_orig, test_size=0.5, random_state=37)

## Feature Scaling
# KNN can be skewed by the magnitude of values so we scale it to make the model more effective
scaler = MinMaxScaler()
scaler.fit(X_train)  # determine the scaling factors from the training features

# Apply the scaler
X_train = scaler.transform(X_train)
X_crossval = scaler.transform(X_crossval)
X_test = scaler.transform(X_test)

## We are going to implement KNN now
# Training with various k values
scores = []
Ks = np.linspace(1,30,15)
for K in Ks:
    clf = KNeighborsClassifier(n_neighbors=int(K)) 

    # "Fit" the model
    clf.fit(X_train, y_train)  

    # Prediction
    y_pred = clf.predict(X_crossval)  

    '''
    # Confusion matrix
    print('Confusion matrix: [[tn fp]')
    print('                   [fn tp]]')
    print(confusion_matrix(y_crossval, y_pred))
    '''

    # Accuracy, precision, recall and F1-score
    accuracy  = clf.score(X_crossval, y_crossval)
    precision = precision_score(y_crossval, y_pred, pos_label='Yes')
    recall    = recall_score(y_crossval, y_pred, pos_label='Yes')
    f1        = f1_score(y_crossval, y_pred, pos_label='Yes')

    scores.append([accuracy,precision,recall,f1])

## Figure 3
'''
plt.plot(Ks, scores)
plt.legend(['Accuracy','Precision','Recall','F1 Score'])
plt.xlabel('K')
plt.ylabel('Score')
plt.show()
'''

# Now try the K with the highest recall (20) 
clf = KNeighborsClassifier(n_neighbors=int(20)) 

# "Fit" the model
clf.fit(X_train, y_train)  

# Prediction
y_pred = clf.predict(X_crossval)  


# Confusion matrix
print('Confusion matrix: [[tn fp]')
print('                   [fn tp]]')
print(confusion_matrix(y_crossval, y_pred))

# Accuracy, precision, recall and F1-score
accuracy  = clf.score(X_crossval, y_crossval)
precision = precision_score(y_crossval, y_pred, pos_label='Yes')
recall    = recall_score(y_crossval, y_pred, pos_label='Yes')
f1        = f1_score(y_crossval, y_pred, pos_label='Yes')

print('Test accuracy:  ', accuracy)
print('Test precision: ', precision)
print('Test recall:    ', recall)
print('Test F1-score:  ', f1)

## Okay, now we are going to try KNN without the categories that were converted to integers from categories
df = df.drop(['Gender'], axis=1)
df = df.drop('Exercise Habits', axis=1)
df = df.drop('Smoking', axis=1)
df = df.drop('Family Heart Disease', axis=1)
df = df.drop('Diabetes', axis=1)
df = df.drop('High Blood Pressure', axis=1)
df = df.drop('Low HDL Cholesterol', axis=1)
df = df.drop('High LDL Cholesterol', axis=1)
df = df.drop('Alcohol Consumption', axis=1)
df = df.drop('Stress Level', axis=1)
df = df.drop('Sugar Consumption', axis=1)

print(df)

## Split the Data
X = df.iloc[:, :-1]  # all rows, 1st to 2nd-to-last col -- features
y = df.iloc[:, -1]   # all rows, last col -- outcome labels

# Split X and y into training (60%), cross val (20%), and test (20%) sets.
# also randomize so the distribution of yeses and nos in the test/train is more equal
X_train, X_orig, y_train, y_orig = train_test_split(X, y, test_size=0.4, random_state=37)
X_crossval, X_test, y_crossval, y_test = train_test_split(X_orig, y_orig, test_size=0.5, random_state=37)

## Feature Scaling
# KNN can be skewed by the magnitude of values so we scale it to make the model more effective
scaler = MinMaxScaler()
scaler.fit(X_train)  # determine the scaling factors from the training features

# Apply the scaler
X_train = scaler.transform(X_train)
X_crossval = scaler.transform(X_crossval)
X_test = scaler.transform(X_test)

## We are going to implement KNN now
# Training with various k values
scores = []
Ks = np.linspace(1,30,15)
for K in Ks:
    clf = KNeighborsClassifier(n_neighbors=int(K)) 

    # "Fit" the model
    clf.fit(X_train, y_train)  

    # Prediction
    y_pred = clf.predict(X_crossval)  

    '''
    # Confusion matrix
    print('Confusion matrix: [[tn fp]')
    print('                   [fn tp]]')
    print(confusion_matrix(y_crossval, y_pred))
    '''

    # Accuracy, precision, recall and F1-score
    accuracy  = clf.score(X_crossval, y_crossval)
    precision = precision_score(y_crossval, y_pred, pos_label='Yes')
    recall    = recall_score(y_crossval, y_pred, pos_label='Yes')
    f1        = f1_score(y_crossval, y_pred, pos_label='Yes')

    scores.append([accuracy,precision,recall,f1])

## Figure 4
'''
plt.plot(Ks, scores)
plt.legend(['Accuracy','Precision','Recall','F1 Score'])
plt.xlabel('K')
plt.ylabel('Score')
plt.show()
'''

# Now try the K with the highest recall (22) 
clf = KNeighborsClassifier(n_neighbors=int(22)) 

# "Fit" the model
clf.fit(X_train, y_train)  

# Prediction
y_pred = clf.predict(X_crossval)  


# Confusion matrix
print('Confusion matrix: [[tn fp]')
print('                   [fn tp]]')
print(confusion_matrix(y_crossval, y_pred))

# Accuracy, precision, recall and F1-score
accuracy  = clf.score(X_crossval, y_crossval)
precision = precision_score(y_crossval, y_pred, pos_label='Yes')
recall    = recall_score(y_crossval, y_pred, pos_label='Yes')
f1        = f1_score(y_crossval, y_pred, pos_label='Yes')

print('Test accuracy:  ', accuracy)
print('Test precision: ', precision)
print('Test recall:    ', recall)
print('Test F1-score:  ', f1)

# Interesting that the scores (except recall) are higher for doing KNN without the categorical variables
# the margins are very small though

# One other thing I did was raise the number of neighbors up to 1000 and the results were interesting (below)
scores = []
Ks = np.linspace(1,1000,50)
for K in Ks:
    clf = KNeighborsClassifier(n_neighbors=int(K)) 

    # "Fit" the model
    clf.fit(X_train, y_train)  

    # Prediction
    y_pred = clf.predict(X_crossval)  

    '''
    # Confusion matrix
    print('Confusion matrix: [[tn fp]')
    print('                   [fn tp]]')
    print(confusion_matrix(y_crossval, y_pred))
    '''

    # Accuracy, precision, recall and F1-score
    accuracy  = clf.score(X_crossval, y_crossval)
    precision = precision_score(y_crossval, y_pred, pos_label='Yes')
    recall    = recall_score(y_crossval, y_pred, pos_label='Yes')
    f1        = f1_score(y_crossval, y_pred, pos_label='Yes')

    scores.append([accuracy,precision,recall,f1])

## Figure 5
'''
plt.plot(Ks, scores)
plt.legend(['Accuracy','Precision','Recall','F1 Score'])
plt.xlabel('K')
plt.ylabel('Score')
plt.show()
'''

# accuracy and precision settle around 0.5
# recall increases to around 0.8
# f1 score increases to around 0.6

# I am not entirely sure why this is 
# But, if the model were used for diagnosis, this could be beneficial as a high recall tends to be favored
# that being said, a quick google search said that the typical number of neighbors goes up to 30
# so i didn't really trust the results with the high K value but I thought it still worth reporting.

# gonna run the model with a high k now
clf = KNeighborsClassifier(n_neighbors=int(1000)) 

# "Fit" the model
clf.fit(X_train, y_train)  

# Prediction
y_pred = clf.predict(X_crossval)  


# Confusion matrix
print('Confusion matrix: [[tn fp]')
print('                   [fn tp]]')
print(confusion_matrix(y_crossval, y_pred))

# Accuracy, precision, recall and F1-score
accuracy  = clf.score(X_crossval, y_crossval)
precision = precision_score(y_crossval, y_pred, pos_label='Yes')
recall    = recall_score(y_crossval, y_pred, pos_label='Yes')
f1        = f1_score(y_crossval, y_pred, pos_label='Yes')

print('Test accuracy:  ', accuracy)
print('Test precision: ', precision)
print('Test recall:    ', recall)
print('Test F1-score:  ', f1)