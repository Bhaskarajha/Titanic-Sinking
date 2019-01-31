# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:18:19 2019

@author: BHASKAR JHA
"""

import pandas as pd
from pandas import Series,DataFrame
import os

path="D:/analytics/kaggle dataset Machine Learning"
os.chdir(path)
os.listdir(path)

titanic_data=pd.read_csv("train.csv")
titanic_data.head()

titanic_data.info()

import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline 

#it will shows no. of male and female in titanic
sns.countplot('Sex', data = titanic_data)

#now seperating the gender by class
#dividing the male and female on the basis of class and count 
sns.countplot("Sex",data=titanic_data,hue='Pclass')

#now by pclass divided by sex
sns.countplot("Pclass",data=titanic_data,hue='Sex')

#we want male,female,children
def male_female_child(passenger):
    Age,Sex=passenger
    if Age<16:
        return 'Child'
    else:
        return Sex

#it will create the person column in which there will be three groups male,female,children     
titanic_data['Person']=titanic_data[['Age','Sex']].apply(male_female_child,axis=1)
#this graph will show the the people from each age group
titanic_data['Age'].hist(bins=70)
#for mean age group
titanic_data['Age'].mean()

#comparison of male female and children #it will count the no. of male female and children
titanic_data.groupby('Person').count()

fig=sns.FacetGrid(titanic_data,hue='Sex',aspect=2)

deck=titanic_data['Cabin'].dropna()
deck.head()

#now finding the person who are alone in the titanic
titanic_data.head()
titanic_data['Alone']=titanic_data.SibSp+titanic_data.Parch
titanic_data['Alone']


titanic_data['Alone'].loc[titanic_data['Alone']>0]='With_Family'
titanic_data['Alone'].loc[titanic_data['Alone']==0]='Alone'

#shows the no. of people who was alone and with family in the titanic
sns.countplot('Alone',data=titanic_data,palette='Blues')

#making column of survived people and giving value yes or no.
titanic_data['Survivor']=titanic_data.Survived.map({0:'no',1:'yes'})

#plot the graph of survivor column
sns.countplot('Survivor',data=titanic_data,palette='Blues')

#plotting the graph of survived people with class
sns.factorplot('Pclass','Survived',data=titanic_data)

#when we add hue='person' it will give male,female ,children survived on the basis of class
sns.factorplot('Pclass','Survived',hue='Person',data=titanic_data)

generations=[10,20,30,40,60,80]
sns.implot('Age','Survived',hue='Pclass',data=titanic_data,palette='winter',xbins=generations)










