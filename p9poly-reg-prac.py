#4.3.18
#Polynomial regression of a data 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sklearn

#Reading the data
salary = pd.read_csv("E:/OUTPUTPYTHON/Position_Salaries.csv")

#Type of data and its variables
type(salary)

type(salary.Position)

type(salary.Level)

type(salary.Salary)

#Descriptive statistics
salary.describe()

#Renaming independent and dependent variables
x = salary.Level

type(x)

y = salary.Salary

type(y)

#Reshaping the arrays
X = np.reshape(np.array(x), (-1,1))
Y = np.reshape(np.array(y), (-1,1))

#Check for missing values
pd.isna(salary)

#Scatter plot
plt.scatter(X,Y)

#Polynomial regression prediction
from sklearn.preprocessing import PolynomialFeatures

salary_reg = PolynomialFeatures(degree = 2)

x_sal = salary_reg.fit_transform(X)

ypred = salary_reg.fit(x_sal,Y)

ypred

#Comparing with linear regression
from sklearn.linear_model import LinearRegression

SalLinReg = LinearRegression()

SalLinReg.fit(X,Y)

#Plots
plt.scatter(x,y)
plt.plot(X,SalLinReg.predict(X),color = "red")
