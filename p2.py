#24.2.18
#Importing dataset and cleaning the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl

hsba = pd.read_excel("E:/4sem MSc/HSBA.xls")

#Obtaining the type of the dataset
type(hsba)

#Descriptive statistics
hsba.describe()

#Plotting a scatter plot to find the correlation
plt.scatter(hsba.Open, hsba.High)

plt.scatter(hsba.Close, hsba.Low) 

plt.scatter(hsba.Volume, hsba["Adj Close"])

plt.scatter(hsba.Open, hsba.Close)

#To find missing values in the data
pd.isna(hsba)

#Replacing the missing values
hsba.Open = hsba.Open.fillna(hsba.Open.mean())

hsba.Close = hsba.Close.fillna(hsba.Close.mean())

hsba.High = hsba.High.fillna(hsba.High.mean())

hsba.Low = hsba.Low.fillna(hsba.Low.mean())

hsba.Volume = hsba.Volume.fillna(hsba.Volume.mean())

hsba["Adj Close"] = hsba["Adj Close"].fillna(hsba["Adj Close"].mean())

#Converting strings to categories
hsba.Month = hsba.Month.map({'Jan':1,
                             'Feb':2, 
                             'Mar':3,
                             'Apr':4, 
                             'May':5, 
                             'Jun':6,
                             'Jul':7, 
                             'Aug':8, 
                             'Sep':9,
                             'Oct':10,
                             'Nov':11,
                             'Dec':12})

type(hsba.Month)

hsba.Month

#Creation of dummy variables
hsba = pd.read_excel("E:/4sem MSc/HSBA.xls")

pd.get_dummies(hsba.Month)

monthdummies = pd.get_dummies(hsba.Month)

monthdummies

hsba = pd.concat([hsba,monthdummies],axis = 1) #Concats column wise if axis = 1
