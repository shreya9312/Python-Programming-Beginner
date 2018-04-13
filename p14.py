#12.3.18
#Different types of clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn

#Import the dataset
mall = pd.read_csv("E:/OUTPUTPYTHON/Mall_Customers.csv")

mall = mall.iloc[:,2:5]

#To find column names
mall.columns

#To find the type of the variables 
type(mall)

type(mall.Age)

type(mall['AnnualIncome (k$)'])

type(mall['SpendingScore (1-100)'])

#Descriptive statistics
mall.describe()

#Correlation
mall.corr()

#Plots
plt.scatter(mall.Age,mall['SpendingScore (1-100)'])

plt.hist(mall.Age)

plt.hist(mall['AnnualIncome (k$)'])

plt.hist(mall['SpendingScore (1-100)'])

#1.K-Means clustering
#Method 1 of K Means clusterring
#Fitting elbow graph to find optimal number of clusters
from sklearn.cluster import KMeans

#Within cluster sum of squares
wcss = []

KM_cluster = KMeans( n_clusters=8, 
                    init='k-means++', 
                    n_init=10, 
                    max_iter=300, 
                    tol=1e-4, 
                    precompute_distances='auto', 
                    verbose=0, 
                    random_state=None, 
                    copy_x=True, 
                    n_jobs=1, 
                    algorithm='auto')

#Fitting the method 1 of clustering to the data
KM_cluster.fit(mall)

wcss.append(KM_cluster.inertia_)

KM_cluster.cluster_centers_

#Cluster prediction
cluster_membership = KM_cluster.fit_predict(mall)

#Converting to dataframe format
cluster_membership = pd.DataFrame(cluster_membership)

#Result
cluster_data = pd.concat([mall,cluster_membership],axis = 1)

plt.plot(range(1,8),wcss)

##############################################################################################
#Method 2 of K Means Clusterring
#To find optimal number of clusters
#Import the same dataset

mall2 = pd.read_csv("E:/OUTPUTPYTHON/Mall_Customers.csv")

mall2 = mall2.iloc[:,2:5]

from sklearn.cluster import KMeans

wcss2 =[]

for i in range(1,11):
    KM_cluster2 = KMeans(n_clusters = i,
                         init='k-means++', 
                         n_init=10, 
                         max_iter=300, 
                         tol=1e-4, 
                         precompute_distances='auto', 
                         verbose=0, 
                         random_state=None, 
                         copy_x=True, 
                         n_jobs=1, 
                         algorithm='auto')
    
KM_cluster2.fit(mall2)

wcss2.append(KM_cluster2.inertia_)
    
plt.plot(range(1,11),wcss2)
plt.show()
#Optimum no of clusters discovered is 5

KM_cluster2 = KMeans(n_clusters = 5,
                     init='k-means++', 
                     n_init=10, 
                     max_iter=300, 
                     tol=1e-4, 
                     precompute_distances='auto', 
                     verbose=0, 
                     random_state=None, 
                     copy_x=True, 
                     n_jobs=1, 
                     algorithm='auto')

cluster_membership2 = KM_cluster2.fit_predict(mall2)



#Visualizing the clusters
#mall2 and cluster_membership2 must not be in dataframe format

mall2 = mall2.values 

plt.scatter(mall2[cluster_membership2 == 0,0],mall2[cluster_membership2 == 0,1],s = 100,c = 'red',label = "cluster1")

plt.scatter(mall2[cluster_membership2 == 1,0],mall2[cluster_membership2 == 1,1],s = 100,c = 'blue',label = "cluster2")

plt.scatter(mall2[cluster_membership2 == 2,0],mall2[cluster_membership2 == 2,1],s = 100,c = 'pink',label = "cluster3")

plt.scatter(mall2[cluster_membership2 == 3,0],mall2[cluster_membership2 == 3,1],s = 100,c = 'yellow',label = "cluster4")

plt.scatter(mall2[cluster_membership2 == 4,0],mall2[cluster_membership2 == 4,1],s = 100,c = 'green',label = "cluster5")

plt.legend()

#Result
mall2 = pd.DataFrame(mall2)

cluster_membership2 = pd.DataFrame(cluster_membership2)

cluster_data2 = pd.concat([mall2,cluster_membership2],axis = 1)

##############################################################################################
#2.Hierarchical clustering
#Import the same dataset

hmall = pd.read_csv("E:/OUTPUTPYTHON/Mall_Customers.csv")

hmall = hmall.iloc[:,2:5]

#Descriptive statistics
hmall.describe()

#Correlation
hmall.corr()

#Plotting a dendrogram to find optimal clusters
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(hmall,method = 'ward'))

#Fitting the hierarchical clusters
from sklearn.cluster import AgglomerativeClustering

H_cluster =  AgglomerativeClustering(n_clusters=4, 
                                     affinity="euclidean", 
                                     memory=None, 
                                     connectivity=None, 
                                     compute_full_tree='auto', 
                                     linkage='ward', 
                                     pooling_func=np.mean)

H_cluster.fit(hmall)

H_cluster_membership = H_cluster.fit_predict(hmall)

#Visualizing the clusters
hmall = hmall.values

plt.scatter(hmall[H_cluster_membership == 0,0],hmall[H_cluster_membership == 0,1],s = 100,c = 'red',label = 'cluster1')

plt.scatter(hmall[H_cluster_membership == 1,0],hmall[H_cluster_membership == 1,1],s = 100,c = 'pink',label = 'cluster2')

plt.scatter(hmall[H_cluster_membership == 2,0],hmall[H_cluster_membership == 2,1],s = 100,c = 'blue',label = 'cluster3')

plt.scatter(hmall[H_cluster_membership == 3,0],hmall[H_cluster_membership == 3,1],s = 100,c = 'brown',label = 'cluster4')

plt.legend()
#hmall and H_cluster_membership must be an array

#Report
hmall = pd.DataFrame(hmall)

H_cluster_membership = pd.DataFrame(H_cluster_membership)

H_cluster_data = pd.concat([hmall,H_cluster_membership],axis = 1)