#!/usr/bin/env python
# coding: utf-8

# # TITANIC DATASET
# 
# ### Source : Kaggle
# Algorithm : Regression - Classification                                                                                              
# Aim : Predict whether a passenger died or survived
# 
# ¬ Project 1/100

# In[70]:


#libraries required: for data analysis, multidimensional structured datam mathematical functions, plotting of refined data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:


#import and read first 5 rows
titanic = pd.read_csv('train.csv')
titanic.head()


# In[3]:


#summarised description : can be plotted for visual distribution
titanic.describe()


# ### data exploration

# In[4]:


#check null value count
titanic.isnull().any().sum()


# In[5]:


#check null null value count per column/feature/attribute
titanic.isnull().sum()


# #### visualization

# In[6]:


import seaborn as sns


# In[7]:


sns.heatmap(titanic.isnull())


# In[8]:


titanic.hist(bins=150)


# In[9]:


titanic['Age'].hist()


# In[10]:


titanic['Survived'].hist()


# In[11]:


titanic['Fare'].hist()


# In[12]:


titanic['Pclass'].hist()


# In[13]:


titanic = titanic.dropna(axis=0)


# In[14]:


sns.distplot(titanic['Age'])


# In[15]:


sns.distplot(titanic['Pclass'])


# In[16]:


sns.countplot(x='Survived',hue='Sex', data=titanic)


# #the females survival rate was higher than that of the males, whereas in death more males died

# In[17]:


#hexbin plot :shows the counts of observations that fall within hexagonal bins
y = titanic['Age']
x = titanic['Survived']
plt.hexbin(x,y, gridsize=(5,5))
plt.colorbar()
plt.show()


# most common age that survived was around 30yrs, most common age that died was 50yrs                                               
# another sample/ same data

# In[18]:


with sns.axes_style("white"):
    sns.jointplot(x, y, kind="hex", color="k");
    plt.colorbar()


# In[19]:


#age against class hexbin
plt.hexbin(titanic['Pclass'], titanic['Age'], gridsize=(5,5))
plt.colorbar()
plt.show()


# In[20]:


sns.countplot(titanic['Survived'], hue='Pclass', data=titanic)


# class 1 had the highest survival rate whereas 3 had the lowest among the survivors, among the dead class 1->3->2 respectively
# 
# below checks on the sexes against the Passenger class

# In[21]:


sns.countplot(titanic['Pclass'], hue='Sex', data=titanic)
#passenger classes against the sexes


# In[ ]:





# In[22]:


sns.countplot(titanic['Embarked'], hue='Survived', data=titanic)


# In[23]:


pp = sns.pairplot(titanic, size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))
fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)


# 

# In[24]:


plt.figure(figsize=(7,5))
sns.boxplot(titanic['Pclass'], titanic['Age'], data=titanic)


# the older, the richer/the better the class ticket
# 
# :Boxplot
# It can tell you about your outliers and what their values are. It can also tell you if your data is symmetrical, how tightly your data is grouped, and if and how your data is skewed. Similar function to a histogram plot and a distribution plot(distplot-seaborn) but takes less space                                                                                         
# *seaborn doesn't take non integers

# In[25]:


plt.figure(figsize=(7,5)
sns.boxplot(titanic['Embarked'], titanic['Age'], data=titanic)


# correlation matrix shows relationships b2n several features   *b2n two variables - correlation : Pearson correlation coef                                           
# 
# -1 = perfectly negative/weak linear correlation, 0 = no correlation, 1 = perfectly linear/strong correlation                          
# the higher the stronger, the lower the weaker 

# In[26]:


corr=titanic.corr()
sns.heatmap(corr)


# In[27]:


titanic.corr()


# the correlation levels are low. could be cause this is predictive classification.

# #### data imputation & ..

# > what if we were check what age group survived by filling the null age values with the median/mean value?                            
# > replacing the missing embarked value
# 
# * will be using age, sex, passenger class and embarking to predict death

# In[28]:


print('median age: ', titanic['Age'].median())
print('mean age: ', titanic['Age'].mean())


# In[29]:


#using the median as it's whole
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic['Age'].isnull().any()


# In[30]:


titanic.groupby('Embarked').count()


# filling the missing embarked values with the most common cabin value 

# In[31]:


titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic['Embarked'].isnull().any()


# In[ ]:





# In[32]:


#dropping cabins as it lacks many values: 600+ out of 890
titanic = titanic.drop(columns='Cabin')
titanic


# In[33]:


titanic.keys()


# In[34]:


titanic.head()


# In[ ]:





# In[35]:


titanic['Age'] = titanic['Age'].astype(float)


# Creating dummy tables for training purposes. This also ensures a backup of the original dataset                                      
# * We will have two of this as I would like to see the presence of any family members and group them as per their age and predict their survival rate too
# 
# * first removing : 'Name', SibSp', 'Parch', 'Ticket', 'Fare'

# In[39]:


titanic_df_A = titanic
titanic_df_A = titanic_df_A.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket', 'Fare'],axis=1,inplace=True)


# In[40]:


titanic_df_A.head()


# In[41]:


titanic_df_A.keys()


# In[42]:


titanic_df_A.loc[(titanic_df_A.Sex == 'female'), 'Sex'] = '1'
titanic_df_A.loc[(titanic_df_A.Sex == 'male'), 'Sex'] = '0'
titanic_df_A


# In[43]:


titanic_df_A.head()


# In[44]:


titanic_df_A.loc[(titanic_df_A.Embarked == 'C'), 'Embarked'] = '1'
titanic_df_A.loc[(titanic_df_A.Embarked == 'Q'), 'Embarked'] = '2'
titanic_df_A.loc[(titanic_df_A.Embarked == 'S'), 'Embarked'] = '3'
titanic_df_A


# drop the feature to be predicted

# In[45]:


titanic_train = titanic_df_A.drop('Survived', axis=1)


# In[76]:


titanic_train


# In[46]:


titanic['Survived'].count()


# #### dataset training & testing : regression

# In[47]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#train & test division : 75% against 25%.. altered to check performance


# In[95]:


x_train, x_test, y_train, y_test = train_test_split(titanic_train, titanic['Survived'], test_size=0.20, random_state=10)


# using 2 algorithms: A. Support Vector Machine(SVM), B. Random Forest Classifier, C. Logistic Regression, D. K Neighbors, E. Decision Tree                              
# 
# #########    
# 
# A. SUPPORT VECTOR MACHINE

# In[53]:


from sklearn import svm


# In[97]:


print('SUPPORT VECTOR MACHINE')

print('----------------------------')

sv = svm.SVC()
sv.fit(x_train, y_train)
sprediction = sv.predict(x_test)
print('predictions  : ', sprediction)
print('----------------------------')

#accuracy score
accuracy = accuracy_score(y_test, sprediction)
print('accuracy score : ', accuracy)
print('----------------------------')

#confusion matrix
acc_confmatrix=confusion_matrix(y_test,sprediction)
print('confusion matrix : ', acc_confmatrix)
print('----------------------------')
print('----------------------------')


# B. RANDOM FOREST CLASSIFIER

# In[58]:


from sklearn.ensemble import RandomForestClassifier


# In[98]:


print('RANDOM FOREST CLASSIFIER')

print('----------------------------')

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rprediction = rfc.predict(x_test)
print('predictions  : ', rprediction)
print('----------------------------')

#accuracy score
accuracy = accuracy_score(y_test, rprediction)
print('accuracy score : ', accuracy)
print('----------------------------')

#confusion matrix
acc_confmatrix=confusion_matrix(y_test,rprediction)
print('confusion matrix : ', acc_confmatrix)
print('----------------------------')
print('----------------------------')


# C: LOGISTIC REGRESSION

# In[60]:


from sklearn.linear_model import LogisticRegression


# In[ ]:





# In[138]:


print('LOGISTIC REGRESSION')

print('----------------------------')

lr = LogisticRegression()
lr.fit(x_train, y_train)
lprediction = lr.predict(x_test)
#survival_prediction = log_model.predict(titanic_df_test)
print('predictions  : ', lprediction)
print('----------------------------')

#accuracy score
accuracy = accuracy_score(y_test, lprediction)
print('accuracy score : ', accuracy)
print('----------------------------')

#confusion matrix
acc_confmatrix=confusion_matrix(y_test,lprediction)
print('confusion matrix : ', acc_confmatrix)
print('----------------------------')
print('----------------------------')


# In[94]:


lprediction.sum()


# D: K NEIGHBORS CLASSIFIER

# In[99]:


from sklearn.neighbors import KNeighborsClassifier

print('K NEIGHBORS CLASSIFIER')
print('----------------------------')

knc = KNeighborsClassifier(n_neighbors=15)
knc.fit(x_train, y_train)
kprediction = knc.predict(x_test)
print('predictions  : ', kprediction)
print('----------------------------')

#accuracy score
accuracy = accuracy_score(y_test, kprediction)
print('accuracy score : ', accuracy)
print('----------------------------')

#confusion matrix
acc_confmatrix=confusion_matrix(y_test, kprediction)
print('confusion matrix : ', acc_confmatrix)
print('----------------------------')
print('----------------------------')


# In[ ]:


E: DECISION TREE CLASSIFIER


# In[112]:


from sklearn.tree import DecisionTreeClassifier

print('DECISION TREE CLASSIFIER')
print('----------------------------')

dtc = DecisionTreeClassifier(max_depth=10, min_samples_leaf=15)
dtc.fit(x_train, y_train)
dprediction = dtc.predict(x_test)

print('predictions  : ', dprediction)
print('----------------------------')

#accuracy score
accuracy = accuracy_score(y_test, dprediction)
print('accuracy score : ', accuracy)
print('----------------------------')

#confusion matrix
acc_confmatrix=confusion_matrix(y_test,dprediction)
print('confusion matrix : ', acc_confmatrix)
print('----------------------------')
print('----------------------------')


# * with a lower training sample, the performance of each algorithm improved
# * a lower training sample and an increased random state led to poor algorithm performance

# In[101]:



#evaluation of all  A. Support Vector Machine(SVM), B. Random Forest Classifier, 
#C. Logistic Regression, D. K Neighbors Classifier, E. Decision Tree Classifier
print('Support Vector Machine(SVM) : ',classification_report(y_test,sprediction))
print('Random Forest Classifier :',classification_report(y_test,rprediction))
print('Logistic Regression :',classification_report(y_test,lprediction))
print('K Neighbors Classifier :',classification_report(y_test,kprediction))
print('Decision Tree Classifier :',classification_report(y_test,dprediction))


# #### test and predictions

# In[113]:


titanic_df_A.head()


# In[126]:


titanic_test = pd.read_csv('test.csv')
titanic_test


# In[131]:


titanic_test.isnull().any()


# In[132]:


titanic_test.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Fare'],axis=1,inplace=True)
titanic_test


# In[134]:


#ensuring the train and test tables are similar

titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].median())
titanic_test['Age'] = titanic_test['Age'].astype(float)
print(titanic['Age'].isnull().any())

titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')
print(titanic_test['Embarked'].isnull().any())


titanic_test.loc[(titanic_test.Sex == 'female'), 'Sex'] = '1'
titanic_test.loc[(titanic_test.Sex == 'male'), 'Sex'] = '0'
print(titanic_test['Sex'])

titanic_test.loc[(titanic_test.Embarked == 'C'), 'Embarked'] = '1'
titanic_test.loc[(titanic_test.Embarked == 'Q'), 'Embarked'] = '2'
titanic_test.loc[(titanic_test.Embarked == 'S'), 'Embarked'] = '3'
print(titanic_test['Embarked'])


# In[135]:


#besides the predicted survival column
Pass_Id = titanic_test.PassengerId
Sex_fm = titanic_test.Sex
pass_class = titanic_test.Pclass
age = titanic_test.Age
embark = titanic_test.Embarked


# In[139]:


new_lprediction = lr.predict(titanic_test)


# In[174]:


#making some columns the original values for easier reading
titanic_test.loc[(titanic_test.Sex == '1'), 'Sex'] = 'female'
titanic_test.loc[(titanic_test.Sex == '0'), 'Sex'] = 'male'
print(titanic_test['Sex'])

titanic_test.loc[(titanic_test.Embarked == '1'), 'Embarked'] = 'C'
titanic_test.loc[(titanic_test.Embarked == '2'), 'Embarked'] = 'Q'
titanic_test.loc[(titanic_test.Embarked == '3'), 'Embarked'] = 'S'
print(titanic_test['Embarked'])


# In[202]:


titanic_survival_pred = pd.DataFrame({'Passenger ID':Pass_Id, 'Sex':Sex_fm, 'Age': age, 'Survived':new_lprediction})
titanic_survival_pred.head()

#PassengerId 	Pclass 	Sex 	Age 	Embarked
#PassengerId 	Pclass 	Name 	Sex 	Age 	SibSp 	Parch 	Ticket 	Fare 	Cabin 	Embarked


# In[203]:


titanic_survival_pred[(titanic_survival_pred['Age'] >= 50) & (titanic_survival_pred['Survived'] == 1)]


# In[204]:


titanic_survival_pred.head()


# In[206]:


titanic_survival_pred.loc[(titanic_survival_pred.Survived == 1), 'Survived'] = 'yes'
titanic_survival_pred.loc[(titanic_survival_pred.Survived == 0), 'Survived'] = 'no'


# In[215]:


titanic_survival_pred.head()


# In[224]:


#btw.. kukuwastia tu time

surv_cnt = titanic_survival_pred['Survived'] == 'yes'
non_surv_cnt = titanic_survival_pred['Survived'] == 'no' 

print('survivors : ', surv_cnt.sum())
print('non-survivors : ', non_surv_cnt.sum())
print('')
print('test count : ', titanic_survival_pred.count())
print('')
print('trained count : ', titanic.count())


# In[ ]:




