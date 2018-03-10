#28.2.18
#Multiple Linear Regression of 50 startups data set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sklearn

fifty = pd.read_csv("E:/OUTPUTPYTHON/50_Startups.csv")

fifty.columns #Displays the columns in the data

#To describe the type of data and its variables
type(fifty)

type(fifty["R&D"])

type(fifty.Administration)

type(fifty.Marketing)

type(fifty.State) #Only string variable

type(fifty.Profit)

#To display descriptive statistics
fifty.describe()

#To display missing values
pd.isna(fifty)

#To fill the missing values column-wise
fifty["R&D"] = fifty["R&D"].fillna(fifty["R&D"].mean())

fifty.Adminstration = fifty.Administration.fillna(fifty.Administration.mean())

fifty.Marketing = fifty.Marketing.fillna(fifty.Marketing.mean())

fifty.State[6] = "No State"

fifty.State[45] = "No State"

fifty.Profit = fifty.Profit.fillna(fifty.Profit.mean())

#To fill the missing data all at once
#fifty = fifty.fillna(fifty.mean())

#To see the relationship between the variables
fifty.corr()

#To find scatter plots between different variables
plt.scatter(fifty["R&D"],fifty.Profit)

plt.scatter(fifty.State,fifty.Profit)

#To make dummy variables for State since it is a string variable
stdum = pd.get_dummies(fifty.State)
stdum

#concating the dummy variables with the data
fifty = pd.concat([fifty,stdum],axis = 1) #Concats the dummy variables column-wise if axis = 1

#To create x(independent) and y(dependent) variables
x = fifty.iloc[:,[0,1,2,5,6,7,8]]

y = fifty.iloc[:,4]

#Splitting the data for machine learning
from sklearn.cross_validation import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,test_size = 0.2, random_state = 0)

#Building multiple linear regression model
from sklearn.linear_model import LinearRegression

linregmodel = LinearRegression()

linregmodel.fit(x_train,y_train)

#To find co-eff of reg eqt
coeff = linregmodel.coef_

#To find accuracy
rsquare = linregmodel.score(x_train,y_train)

#Predicting the y variables using test data
yprediction = linregmodel.predict(x_test)

#Converting to dataframe
y_Test = pd.DataFrame(y_test)

yPrediction = pd.DataFrame(yprediction)

y_Test.index = yPrediction.index

#Final solution
results = pd.concat((y_Test,yPrediction),axis = 1)
