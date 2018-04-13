#Classification of a dataset using different types
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn

dataset = pd.read_csv("E:/OUTPUTPYTHON/Social_Network_Ads.csv")

#Check the type of dataset and its variables
type(dataset)

type(dataset["User ID"])

type(dataset.Gender)

type(dataset.Age)

type(dataset.EstimatedSalary)

type(dataset.Purchased)

#Descriptive statistics
dataset.describe()

#Graphs
plt.hist(dataset.Purchased)

plt.scatter(dataset.Age,dataset.Purchased)

plt.scatter(dataset.EstimatedSalary,dataset.Purchased)

#Defining x and y variables
x = dataset.iloc[:,2:4]

y = dataset.iloc[:,4:]

#Splitting the dataset
from sklearn.cross_validation import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size = 0.75,test_size = 0.25,random_state = 0)

############################################################################################

#1.Fitting SVM CLASSIFICATION to training set
from sklearn.svm import SVC

svm_classifier = SVC()

svm_classifier.fit(xtrain,ytrain)

svm_classifier.score(xtrain,ytrain)

#Prediction
ypred = svm_classifier.predict(xtest)

yTest = pd.DataFrame(ytest)

yPred = pd.DataFrame(ypred)

yTest.index = yPred.index

result = pd.concat((yTest,yPred), axis = 1)

#Misclassification matrix or confusion matrix if dependent variable is categorical or factor
from sklearn.metrics import confusion_matrix

confusionmatrix = confusion_matrix(ytest,ypred)

confusionmatrix

###############################################################################################################

#2.Fitting NAIVE BAYES CLASSIFICATION to training set
from sklearn.naive_bayes import GaussianNB

NB_classifier = GaussianNB()

NB_classifier.fit(xtrain,ytrain)

NB_classifier.score(xtrain,ytrain)

#Prediction
ypred = NB_classifier.predict(xtest)

yTest = pd.DataFrame(ytest)

yPred = pd.DataFrame(ypred)

yTest.index = yPred.index

result = pd.concat((yTest,yPred), axis = 1)

#Misclassification matrix or confusion matrix if dependent variable is categorical or factor
from sklearn.metrics import confusion_matrix

confusionmatrix = confusion_matrix(ytest,ypred)

confusionmatrix

###########################################################################################################

#3.Fitting DECISION TREE CLASSIFICATION to training set
from sklearn.tree import DecisionTreeClassifier

DT_classifier = DecisionTreeClassifier()

DT_classifier.fit(xtrain,ytrain)

DT_classifier.score(xtrain,ytrain)

#Prediction
ypred = DT_classifier.predict(xtest)

yTest = pd.DataFrame(ytest)

yPred = pd.DataFrame(ypred)

yTest.index = yPred.index

result = pd.concat((yTest,yPred), axis = 1)

#Misclassification matrix or confusion matrix if dependent variable is categorical or factor
from sklearn.metrics import confusion_matrix

confusionmatrix = confusion_matrix(ytest,ypred)

confusionmatrix

#################################################################################################

#4.Fitting RANDOM FOREST CLASSIFICATION to training set
from sklearn.ensemble import RandomForestClassifier

RF_classifier = RandomForestClassifier()

RF_classifier.fit(xtrain,ytrain)

RF_classifier.score(xtrain,ytrain)

#Prediction
ypred = RF_classifier.predict(xtest)

yTest = pd.DataFrame(ytest)

yPred = pd.DataFrame(ypred)

yTest.index = yPred.index

result = pd.concat((yTest,yPred), axis = 1)

#Misclassification matrix or confusion matrix if dependent variable is categorical or factor
from sklearn.metrics import confusion_matrix

confusionmatrix = confusion_matrix(ytest,ypred)

confusionmatrix

#######################################################################################################
