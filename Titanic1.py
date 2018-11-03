# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 18:32:56 2018

@author: ml
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 19:00:22 2018

@author: ml
"""
import os

for root, dirs, files in os.walk("."):  
    for filename in files:
        print(filename)
        
from sklearn.ensemble import RandomForestRegressor
#error metrics use c-stat(roc/auc)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
df1 = pd.read_csv("train.csv")

print(df.head())

y = df.pop("Survived")# take survived off of df and make a new data frame y that only has survived

df.shape
df.columns
df1.columns




#eda
y.value_counts()
df.Age.value_counts()
df.Sex.value_counts()

342/891.0

#eda exploatory data analysis 
#find out
#categorical?
#if not max, min 
#missing values.cannot run sklearn with missing values
#distribution

df.Sex.value_counts()
#plot 
df.Sex.value_counts().plot(kind ="bar")

df.Sex=="female"

df[df.Sex=="female"]

df[df.Sex=="null"]

df.describe()

df.Fare.hist(bins = 5)

df[df.Fare ==0]

df[df.Cabin.isnull()]

df1[df1.Sex=="male"].Survived.value_counts().plot(kind="bar", title="Male")

df1[df1.Sex=="female"].Survived.value_counts().plot(kind="bar")

df1[(df1.Age<15) & (df1.Sex==("female"))].Survived.value_counts().plot(kind="bar")
df1[(df1.Age<15) & (df1.Sex==("male"))].Survived.value_counts().plot(kind="bar", title="Male")


y =df.Survived

y.describe()

df.describe()

df["Age"].describe()

df[df.Age.isnull()]




#fill in missing values for age
df["Age"].fillna(df.Age.mean(), inplace = True)

df[df.Age.isnull()]


#get only numeric variables. !(not).  Just to build  quick model
numeric_variables = list(df.dtypes[df.dtypes != "O"].index)
df[numeric_variables].head()

df[numeric_variables].describe()
df[numeric_variables].columns


#Logistical Regression
logreg = LogisticRegression()
logreg.fit(df[numeric_variables],y)
y_predict = logreg.predict(df[numeric_variables])

print(confusion_matrix(y, y_predict))
print(classification_report(y, y_predict))
print ("C-stat: ",roc_auc_score(y, y_predict))

y_pred_prob = logreg.predict_proba(df[numeric_variables])[:,1]
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y, y_pred_prob)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()



#n_estimators 100 because small dataset
model =RandomForestRegressor(n_estimators = 100, oob_score=True, random_state=42)
#only train numeric varibles 
model.fit(df[numeric_variables],y)
#training_ is availble after running a fit 
#produces r square 
model.oob_score_

#get c stat of oob_score_
#y_oob is an array where every obersation has a prediction
y_oob=model.oob_prediction_
print ("C-stat: ",roc_auc_score(y, y_oob))

#out of bag
#shows array. all varibles, prediction %
y_oob

#plot 
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y, y_oob)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()



categorical_variables = list(df.dtypes[df.dtypes == "O"].index)
df[categorical_variables].head()
df[categorical_variables].describe()

#drop varibles dont want to use
df.drop(["Name", "Ticket", 'PassengerId'], axis = 1, inplace= True)

df.columns

#change values of Cabin letter eliminate number or change to None
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"
#apply function    
df["Cabin"]= df.Cabin.apply(clean_cabin)

df["Cabin"]

categorical_variables = ["Sex", "Cabin", "Embarked"]

for variable in categorical_variables:
    #fill in Missing for missing variables
    df[variable].fillna("Missing", inplace=True)
    #Create array of dummies
    dummies = pd.get_dummies(df[variable], prefix=variable)
    #update df to include dummies and drop main varible
    df = pd.concat([df, dummies], axis=1)
    df.drop([variable], axis=1, inplace=True)

df

#to show all columns
def printall(df, max_rows=10):
    from IPython.display import display, HTML
    display(HTML(df.to_html(max_rows=max_rows)))
printall(df)






#Logistical Regression
logreg = LogisticRegression()
logreg.fit(df,y)
y_predict = logreg.predict(df)
print(confusion_matrix(y, y_predict))
print(classification_report(y, y_predict))
print ("C-stat: ",roc_auc_score(y, y_predict))

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = .3)
logreg.fit(X_train, y_train)
y_predict = logreg.predict(X_test)
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))
print ("C-stat: ",roc_auc_score(y_test, y_predict))






#n_estimators 100 because small dataset
model =RandomForestRegressor(n_estimators = 100, oob_score=True, random_state=42)
#only train numeric varibles 
model.fit(df,y)
#training_ is availble after running a fit 
#produces r square 
model.oob_score_

#get c stat of oob_score_
#y_oob is an array where every obersation has a prediction
y_oob=model.oob_prediction_
print ("C-stat: ",roc_auc_score(y, y_oob))

#plot
fpr, tpr, thresholds = roc_curve(y, y_oob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

#prints the importance of features, see what is important in model
model.feature_importances_

feature_importances = pd.Series(model.feature_importances_, index=df.columns)
feature_importances.plot(kind="barh", figsize=(7,6));

#Improve Parameters in model
#n_estimators: #of trees in forest.choose high as possible, computer can handle
#max_features: # of features to consider
#min_sample_leaf: min # of samples in newly created leaves

#Parameters make easy to train model
#n_jobs: Determines if multiple processors should be used to train and test model
#always set this to -1 and %%timeit vs if it is set to 1 it should be much faster
#especially when many trees are trained.


#n_jobs

model = RandomForestRegressor(1000, oob_score=True, n_jobs=1, random_state=42)
model.fit(df,y)

results=[]
n_estimator_options =[30,50,100,200,500,1000,2000]

for trees in n_estimator_options:
    model = RandomForestRegressor(trees,oob_score=True, n_jobs=1, random_state=42)
    model.fit(df,y)
    print (trees,"trees") 
    roc = roc_auc_score(y, model.oob_prediction_)
    print ("C-stat:" , roc)
    results.append(roc)
    print ("")
    
pd.Series(results, n_estimator_options).plot();

results = []
max_features_options = ["auto", "sqrt", "log2", None, 0.9, 0.2]

for max_features in max_features_options:
    model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=1, random_state=42, max_features = max_features)
    model.fit(df, y)
    print(max_features, "options")
    roc = roc_auc_score(y, model.oob_prediction_)
    print ("C-stat:" , roc)
    results.append(roc)
    print ("")

pd.Series(results, max_features_options).plot(kind = "bar", xlim=(.85, .88));

results = []
min_samples_leaf_options = [1,2,3,4,5,6,7,8,9,10]

for min_samples in min_samples_leaf_options:
    model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=1, random_state=42, max_features= None, min_samples_leaf = min_samples)
    model.fit(df, y)
    print(min_samples, "min samples")
    roc = roc_auc_score(y, model.oob_prediction_)
    print ("C-stat:" , roc)
    results.append(roc)
    print ("")

pd.Series(results, min_samples_leaf_options).plot();

#Final Model
model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=1, random_state=42, max_features= None, min_samples_leaf = 5)
model.fit(df,y)
roc = roc_auc_score(y, model.oob_prediction_)
print ("C-stat:" , roc)
fpr, tpr, thresholds = roc_curve(y, model.oob_prediction_)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = .3)
model.fit(X_train, y_train)
model_p = model.predict(X_test)
roc = roc_auc_score(y_test, model_p)
print ("C-stat:" , roc)
fpr, tpr, thresholds = roc_curve(y_test, model_p)

