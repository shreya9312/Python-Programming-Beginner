#7.3.18
#k Nearest Neighbours Classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn

dataset = pd.read_csv("E:/OUTPUTPYTHON/Social_Network_Ads.csv")

#To obtain type of dataset and variables
type(dataset)

type(dataset["User ID"])

type(dataset.Gender)

type(dataset.Age)

type(dataset.EstimatedSalary)

type(dataset.Purchased)

#Descriptive statistics
dataset.describe()

#To plot graphs
plt.hist(dataset.Purchased)

plt.scatter(dataset.Age,dataset.Purchased)

plt.scatter(dataset.EstimatedSalary,dataset.Purchased)

#To get x and y variables
x = dataset.iloc[:,2:4]

y = dataset.iloc[:,4:]

#Splitting the dataset
from sklearn.cross_validation import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size = 0.75,test_size = 0.25,random_state = 0)

#Fitting kNN to training set
from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors = 5)

knn_classifier.fit(xtrain,ytrain)

knn_classifier.score(xtrain,ytrain)

#Prediction
ypred = knn_classifier.predict(xtest)

yTest = pd.DataFrame(ytest)

yPred = pd.DataFrame(ypred)

yTest.index = yPred.index

result = pd.concat((yTest,yPred), axis = 1)

#Misclassification matrix or confusion matrix if dependent variable is categorical or factor
from sklearn.metrics import confusion_matrix

confusionmatrix = confusion_matrix(ytest,ypred)

confusionmatrix
