#7.3.18
#Classification of a dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn

dataset = pd.read_csv("E:/OUTPUTPYTHON/Social_Network_Ads.csv")

#To obtain the nature of the dataset and its variables
type(dataset)

type(dataset["User ID"])

type(dataset.Gender)

type(dataset.Age)

type(dataset.EstimatedSalary)

type(dataset.Purchased)

dataset.info()

#To obtain descriptive statistics
dataset.describe()

dataset.Purchased = dataset.Purchased.map({0:0,1:1})

type(dataset.Purchased)

dataset.describe()

#To obtain plots
plt.hist(dataset.Purchased)

plt.scatter(dataset.Age,dataset.Purchased)

plt.scatter(dataset.EstimatedSalary,dataset.Purchased)

#To obtain x and y variables
x = dataset.iloc[:,2:4]

y = dataset.iloc[:,4:]

#Splitting the dataset
from sklearn.cross_validation import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size = 0.75,test_size = 0.25,random_state = 0)

#Fitting logistic regression
from sklearn.linear_model import LogisticRegression

LR_classifier = LogisticRegression()

LR_classifier.fit(xtrain,ytrain)

LR_classifier.score(xtrain,ytrain)

LR_classifier.coef_

#Prediction
ypred = LR_classifier.predict(xtest)

yTest = pd.DataFrame(ytest)

yPred = pd.DataFrame(ypred)

yTest.index = yPred.index

result = pd.concat((yTest,yPred), axis = 1)

#Misclassification matrix or confusion matrix if dependent variable is categorical or a factor
from sklearn.metrics import confusion_matrix

confusionmatrix = confusion_matrix(ytest,ypred)

confusionmatrix
