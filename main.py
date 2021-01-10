#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 09:33:21 2020

@author: IsaureQuetel
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

#SIMPLE MODEL 

#Loading data
df = pd.read_csv("Donnees/data.csv", header=None)
tumor = pd.read_csv("Donnees/labels.csv", header=None)

#Data preparation
Y = tumor.iloc[1:,1]
X = df.iloc[1:,1:]

# Model building
logreg = linear_model.LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Train the model
logreg.fit(X_train, y_train)
Z = logreg.predict(X_test)

# Evaluate the model
print("Simple model")
pd.crosstab(y_test,Z)

print("----------------")

#UNDERFITTING

# Data preparation
Y = tumor.iloc[1:,1]
X = df.iloc[1:,1:]

# Model building
logreg = linear_model.LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

X_train = X_train.iloc[45:48,:]
y_train = y_train.iloc[45:48]

# Train the model
logreg.fit(X_train, y_train)
Z = logreg.predict(X_test)

print("Underfitting")
print(accuracy_score(y_test, logreg.predict(X_test)))
print(accuracy_score(y_train, logreg.predict(X_train)))

# Evaluate the model
pd.crosstab(y_test,Z)

print("----------------")

#OVERFITTING

# Data preparation
Y = tumor.iloc[1:,1]
X = df.iloc[1:,1:]

# Model building
logreg = linear_model.LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

X_train = X_train.iloc[:,1:5]
X_test = X_test.iloc[:,1:5]

# Train the model
logreg.fit(X_train, y_train)
Z = logreg.predict(X_test)

print("Overfitting")
print(accuracy_score(y_test, logreg.predict(X_test)))
print(accuracy_score(y_train, logreg.predict(X_train)))


# Evaluate the model
pd.crosstab(y_test,Z)

print("----------------")

#REGULARIZATION

# Data preparation
Y = tumor.iloc[1:,1]
X = df.iloc[1:,1:]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Model building
logreg = linear_model.LogisticRegression(C=1e40)

# Train the model
logreg.fit(X_train, y_train)
Z = logreg.predict(X_test)

# Evaluate the model
print("Régularisation")
print(pd.crosstab(y_test,Z))
print(accuracy_score(y_train, logreg.predict(X_train)))
print(accuracy_score(y_test, logreg.predict(X_test)))

print("----------------")

#LEARNING CURVE

print("Learning curve")

# Data preparation
Y = tumor.iloc[1:,1]
X = df.iloc[1:,1:]

# Model building
logreg = linear_model.LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# logreg.fit(X_train, y_train)
# print(accuracy_score(y_train, logreg.predict(X_train)))

training_accuracy = []
testing_accuracy = []

index = [50,100,150,200]
for i in index:
    logreg = linear_model.LogisticRegression()
    logreg.fit(X_train.iloc[1:i,:],y_train[1:i])
    training_accuracy.append(accuracy_score(y_train[1:i], logreg.predict(X_train.iloc[1:i,:])))
    testing_accuracy.append(accuracy_score(y_test, logreg.predict(X_test)))

plt.plot(index, training_accuracy, label = 'Training')
plt.plot(index, testing_accuracy, label = 'Testing') 
plt.show()

print(index)
print(training_accuracy)

print("----------------")

#GRID SEARCH

print("Grid search")

parameters = {'C': [0.001, 0.01, 1e3, 1e5]}

# Data preparation
Y = tumor.iloc[1:,1]
X = df.iloc[1:,1:]

# Model building
logreg = linear_model.LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

grid_search = GridSearchCV(estimator = logreg,
    param_grid = parameters,
    scoring = 'accuracy',
    cv = 10)

grid_search = grid_search.fit(X_train, y_train)
grid_search.predict(X_test)
grid_search.best_params_
