#19.2.18
#Python programming for beginners

#Mathematical calculations
88 + 23

56 * 80

56 / 12

23 % 3

#To display sentences
'Hello'+' '+'World'

greeting = "Welcome to Python Programming"

print(greeting)

#To obtain the nature of the variable greeting
type(greeting)

#To create lists
list1 = [3,4,5]

type(list1)

list2 = ['a','b','c']

type(list2)

list3 = [1,2,'d']

type(list3)

list4 = [list1,list2,list3]

list4

type(list4)

#To append values in a list
list1 + list2

#20.2.18
#Importing packages
import numpy as np #For mathematical operations of arrays
import pandas as pd #For tabular operations
import matplotlib.pyplot as plt #For interactive plots 
import sklearn as sklrn #For building machine learning models

list1 = [1,2,3,0]

list2 = [4,-5,6,-7]

#Convert lists to an array for mathematical calculations
newlist1 = np.array(list1)

type(newlist1)

newlist2 = np.array(list2)

type(newlist2)

sum = newlist1 + newlist2
sum

diff = newlist1 - newlist2
diff

prod = newlist1 * newlist2
prod

quot = newlist1/newlist2
quot

#To perform mathematical operations on lists, the number of elements in each of the lists should be equal
#In Python the index numbers start from zero

#To extract data elements from a list
#Method 1
list2[0]

list1[3]

#Method 2
list2[0:3] #Extracts elements from 0 to 2

sum[1:3]

diff[0:] #Displays the entire list

list1[1:] #Displays the elements starting from index 1

prod[:3] #Displays the elements upto index 2

#Changing the values in a list
list1[2] = 9

list1

#Creating 2-D arrays
list12d = [[1,2],[3,4],[5,6]]

list12d

list22d = [[5,6],[7,8],[9,0]]

list22d

list12d + list22d #Appending the lists

List12d = np.array(list12d)

List12d

List22d = np.array(list22d)

List22d



















