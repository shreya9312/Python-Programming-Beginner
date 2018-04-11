#Different types of regression analysis for the same dataset

#Import the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sklearn

#Import the dataset
data = pd.read_csv("E:/OUTPUTPYTHON/Position_Salaries.csv")

#Type of dataset
data.info()

#Summary statistics
data.describe()

#To check for missing values
pd.isna(data)

#To plot the data
plt.scatter(data.Level,data.Salary)

#To fill the missing values
data.Level = data.Level.fillna(value = 4)

data.Salary[2] = 56000

data.Salary[11] = 50000

#Assigning x and y variables
x = np.array(data.iloc[:,1:2])

y = np.array(data.iloc[:,2:])

#Feature scaling 
from sklearn.preprocessing import StandardScaler

scalerx = StandardScaler()

scalery = StandardScaler()

x = scalerx.fit_transform(x)

y = scalery.fit_transform(y)

#Split data
from sklearn.cross_validation import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size = 0.75,test_size = 0.25,random_state = 0)

############################################################################################

#1.Fitting SUPPORT VECTOR REGRESSION
from sklearn.svm import SVR

SVR_regressor = SVR( kernel='rbf',
                     degree=3, 
                     gamma='auto',
                     coef0=0.0,
                     tol=1e-3,
                     C=1.0,
                     epsilon=0.1,
                     shrinking=True, 
                     cache_size=200,
                     verbose=False, 
                     max_iter=-1)

#Fitting SVR to training set
SVR_regressor.fit(xtrain,ytrain)

SVR_regressor.score(xtrain,ytrain)

#Fitting SVR to test set
SVR_ypred = SVR_regressor.predict(xtest)

#Converting back to original from feature scale
ytest = scalery.inverse_transform(ytest)

ypred = scalery.inverse_transform(SVR_ypred)

#Preparing result
yTest = pd.DataFrame(ytest)

yPred = pd.DataFrame(ypred)

yTest.index = yPred.index

result = pd.concat((yTest,yPred), axis = 1)

##################################################################################################

#2.Fitting DECISION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor

DT_regressor = DecisionTreeRegressor()

DT_regressor.fit(xtrain,ytrain)

DT_regressor.score(xtrain,ytrain)

DT_ypred = DT_regressor.predict(xtest)

#Converting back to original from feature scale
ytest = scalery.inverse_transform(ytest)

ypred = scalery.inverse_transform(DT_ypred)

#Preparing result
yTest = pd.DataFrame(ytest)

yPred = pd.DataFrame(ypred)

yTest.index = yPred.index

result = pd.concat((yTest,yPred), axis = 1)

##################################################################################################

#3.Fitting RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor

RF_regressor = RandomForestRegressor(n_estimators=10,
                                     criterion="mse",
                                     max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.,
                                     max_features="auto",
                                     max_leaf_nodes=None, 
                                     min_impurity_decrease=0.,
                                     min_impurity_split=None, 
                                     bootstrap=True,
                                     oob_score=False, 
                                     n_jobs=1,
                                     random_state=None,
                                     verbose=0,
                                     warm_start=False)

RF_regressor.fit(xtrain,ytrain)

RF_regressor.score(xtrain,ytrain)

RF_ypred = RF_regressor.predict(xtest)

#Converting back to original from feature scale
ytest = scalery.inverse_transform(ytest)

ypred = scalery.inverse_transform(RF_ypred)

#Preparing result
yTest = pd.DataFrame(ytest)

yPred = pd.DataFrame(ypred)

yTest.index = yPred.index

result = pd.concat((yTest,yPred), axis = 1)

###############################################################################################