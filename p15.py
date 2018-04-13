#13.3.18
#1.Principle component analysis of a data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
import seaborn as sns #Graphs

wine = pd.read_csv("E:/OUTPUTPYTHON/Wine.csv")

type(wine)

wine.columns

wine.describe()

#Defining x and y variables
x = wine.iloc[:,0:13].values

y = wine.iloc[:,13:].values

#Feature scaling
from sklearn.preprocessing import StandardScaler

fs = StandardScaler()

x = fs.fit_transform(x)

#Splitting the data
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size = 0.75,test_size = 0.25,random_state = 0)

#Applying principle component analysis
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit_transform(xtrain)

pca.transform(xtest)

pca.score(xtrain,ytrain)

#Classification of training data and prediction
from sklearn.linear_model import LogisticRegression
 
lr_classifier = LogisticRegression()

pca_fit = lr_classifier.fit(xtrain,ytrain)

pca_fit

ypred = pca_fit.predict(xtest)

#Confusion matrix
from sklearn.metrics import confusion_matrix

confusionmatrix = confusion_matrix(ytest,ypred)

confusionmatrix

#Report
ytest_col = pd.DataFrame(ytest)

ypred_col = pd.DataFrame(ypred)

result = pd.concat([ytest_col,ypred_col],axis = 1)

sns.pairplot(data = wine,hue = 'Customer_Segment')

################################################################################################

#2.Linear discriminant analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
import seaborn as sns #Graphs

wine = pd.read_csv("E:/OUTPUTPYTHON/Wine.csv")

type(wine)

wine.columns

wine.describe()

#Defining x and y variables
x = wine.iloc[:,0:13].values

y = wine.iloc[:,13:].values

#Feature scaling
from sklearn.preprocessing import StandardScaler

fs = StandardScaler()

x = fs.fit_transform(x)

#Splitting the data
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size = 0.75,test_size = 0.25,random_state = 0)

#Fitting lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(solver='svd', 
                                 shrinkage=None, 
                                 priors=None, 
                                 n_components=None, 
                                 store_covariance=False, 
                                 tol=1e-4)

lda.fit_transform(xtrain,ytrain)

lda.transform(xtest)

lda.coef_

lda.score(xtrain,ytrain)

#Classification of training data and prediction
from sklearn.linear_model import LogisticRegression
 
lr_classifier = LogisticRegression()

lda_fit = lr_classifier.fit(xtrain,ytrain)

ypred = lda_fit.predict(xtest)

#Confusion matrix
from sklearn.metrics import confusion_matrix

confusionmatrix = confusion_matrix(ytest,ypred)

confusionmatrix

#Report
ytest_col = pd.DataFrame(ytest)

ypred_col = pd.DataFrame(ypred)

result = pd.concat([ytest_col,ypred_col],axis = 1)
