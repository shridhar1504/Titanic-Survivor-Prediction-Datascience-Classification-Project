#!/usr/bin/env python
# coding: utf-8

# # Titanic Survivor Prediction - Classification Project

# ***
# _**Importing the required libraries & packages**_

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import ydata_profiling as pf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')


# _**Changing The Default Working Directory Path & Reading the Dataset using Pandas Command and displaying the first five observations in the DataFrame**_

# In[2]:


os.chdir('C:\\Users\\Shridhar\\Desktop\\Projects')
df = pd.read_csv('Titanic.csv')
df.head()


# ## Exploratory Data Analysis(EDA)

# _**Checking for the duplicate values in the DataFrame**_

# In[3]:


df.duplicated().sum()


# _**Checking for the null values in all the columns from the DataFrame**_

# In[4]:


df.isna().sum()


# _**Getting the shape of the DataFrame**_

# In[5]:


df.shape


# _**Getting the Data types and Non-null count of all the columns from the DataFrame using <span style = 'background : green'><span style = 'color : white'> .info() </span> </span> statement**_

# In[6]:


df.info()


# _**Getting the summary of various descriptive statistics for all the numeric columns in the DataFrame**_

# In[7]:


df.describe()


# _**Automated Exploratory Data Analysis (EDA) with ydata_profiling(pandas_profiling)**_

# In[8]:


EDA_Report = pf.ProfileReport(df)
EDA_Report.to_file("EDA_Report.html")
EDA_Report


# ## Data Cleaning

# _**Getting the null values from the `Fare` column for the null value treatment process and displaying the DataFrame with null values in the `Fare` column**_

# In[9]:


Fare_null = df[df['Fare'].isnull()]
Fare_null


# _**Getting the mean value of `Fare` column with respect to `Pclass` column of the DataFrame and filling the missing value of the `Fare` column with the calculated mean value and displaying the mean fare of the 3rd class passenger which is filled in the place of null value**_

# In[10]:


Fare_mean = df.Fare[df.Pclass == 3].mean()
df['Fare'].fillna(value = Fare_mean, inplace = True)
print('Mean Fare of 3rd Class Passenger : ',Fare_mean)


# _**After missing value treatment, checking for the null values in the `Fare` column**_

# In[11]:


df['Fare'].isnull().sum()


# _**Displaying all the null values in the `Age` column from the DataFrame**_

# In[12]:


df[df['Age'].isnull()]


# _**Getting the mean value of the `Age` column in the appropriate format depending on the categories found in the `Name` column such as <span style="color:red">"Mr., Mrs., Miss., Master."</span> and displaying the mean values of `Age` for all the four categories from the `Name` column in the DataFrame**_

# In[13]:


mean_age_mr = "{0:.2f}".format(df[df['Name'].str.contains('Mr.', na = False)]['Age'].mean())
mean_age_mrs = "{0:.2f}".format(df[df['Name'].str.contains('Mrs.', na = False)]['Age'].mean())
mean_age_miss = "{0:.2f}".format(df[df['Name'].str.contains('Miss.', na = False)]['Age'].mean())
mean_age_master = "{0:.2f}".format(df[df['Name'].str.contains('Master.', na = False)]['Age'].mean())
print('Mean Age of Adult Men : ',mean_age_mr)
print('Mean Age of Married Women : ',mean_age_mrs)
print('Mean Age of Unmarried Women  : ',mean_age_miss)
print('Mean Age of Male Child : ',mean_age_master)


# _**Defining the function to fill the null values in the `Age` column from the DatFrame, we got mean values for the four categories and hereby finding one another category which is having the null value so filling it out with the founded mean values of <span style="color:red">"Mr., Mrs., Miss., Master." </span> if there's no missing value return the actual age from the `Age` column**_

# In[14]:


def fill_age(Name_Age):
    name = Name_Age[0]
    age = Name_Age[1]
    if pd.isnull(age):
        if 'Mr.' in name:
            return mean_age_mr
        if 'Mrs.' in name:
            return mean_age_mrs
        if 'Miss.' in name:
            return mean_age_miss
        if 'Ms.' in name:
            return mean_age_miss
        if 'Master.' in name:
            return mean_age_master
    else:
        return age


# _**Filling out the null values of `Age` column using the defined function and displaying the null values in the `Age` column from the DataFrame after missing value to treatment to cross-verify**_

# In[15]:


df['Age'] = df[['Name', 'Age']].apply(fill_age, axis = 1)
df['Age'].isna().sum()


# _**Label Encoding the `Sex` column from the DataFrame using Mapping function**_

# In[16]:


df['Sex'] = df['Sex'].map({'male' : 1, 'female' : 0})


# _**One Hot Encoding the `Embarked` column from the DataFrame using pandas get dummies function**_

# In[17]:


emb_dum = pd.get_dummies(df['Embarked'], drop_first = True)
df = pd.concat([df, emb_dum], axis = 1)


# _**Checking the data types of all the columns from the DataFrame to drop the <span style="color:blue"> ~"object"~ </span> data type column for further proceedings**_

# In[18]:


df.dtypes


# _**Dropping the columns `PasssengerID`, `Name`, `Cabin`, `Embarked`, `Ticket` from the DataFrame which is not needed**_

# In[19]:


df.drop(['PassengerId','Name','Cabin','Embarked','Ticket'], axis = 1, inplace = True)


# ## Data Visualisation

# _**Getting the Correlation Values from all the numeric columns from the DataFrame using Seaborn Heatmap & saving the PNG File**_

# In[20]:


plt.rcParams['figure.figsize'] = 15,6
sns.heatmap(df.corr(), annot = True, square = True, cbar = True, cmap = 'Purples')
plt.title('Correlation Heat Map')
plt.savefig('Correlation Heat Map.png')
plt.show()


# _**Plotting the Bar Graph with count of `Survived` passengers and identify the number of passengers who survived and saving the PNG File**_

# In[21]:


plot = sns.countplot(x = df['Survived'])
for p in plot.patches:
    plot.annotate(p.get_height(),(p.get_x() + p.get_width() / 2.0,p.get_height()),
                 ha = 'center',va = 'center',xytext = (0,5),textcoords = 'offset points')
plt.title('Count of Survived and Dead')    
plt.savefig('Count of Survived and Dead.png')
plt.show()


# _**Plotting the Bar Graph with count of passengers travelled in the various class with their gender count got from `Pclass`, `Sex` column and identifying the gender of the passenger travelled in the different passenger classes from the DataFrame and saving the PNG File**_

# In[22]:


plot = sns.countplot(x = df['Pclass'],hue = df['Sex'])
for p in plot.patches:
    plot.annotate(p.get_height(),(p.get_x() + p.get_width() / 2.0,p.get_height()),
                 ha = 'center',va = 'center',xytext = (0,5),textcoords = 'offset points')
plt.title('Count of Male & Female in Passenger Class')
plt.savefig('Count of Male & Female in Passenger Class.png')
plt.show()


# _**Plotting the Bar Graph with count of `Age` of passenger and identify all the age groups of passengers travelled in the Titanic and saving the PNG File**_

# In[23]:


plot = sns.countplot(x = df['Age'])
for p in plot.patches:
    plot.annotate(p.get_height(),(p.get_x() + p.get_width() / 2.0,p.get_height()),
                 ha = 'center',va = 'center',xytext = (0,5),textcoords = 'offset points')
plt.xticks(rotation = 90)
plt.title("Count of Passenger's Age")
plt.savefig("Count of Passenger's Age.png")
plt.show()


# _**Assigning the dependent and independent variable. In the independent variable along with the dependent variable column also dropping `Sex` column since the correlation heat map shows negative correlation in the `Sex` column from the DataFrame**_

# In[24]:


x = df.drop(['Sex','Survived'], axis = 1)
y = df['Survived']


# ## Data Preprocessing

# _**Standardizing the independent variable of the DataFrame using MinMaxScaler function**_

# In[25]:


MM = MinMaxScaler()
x = MM.fit_transform(x)


# ## Model Fitting

# _**Defining the Function for the ML algorithms using <span style="color:purple">RandomizedSearchCV</span> Algorithm and splitting the dependent variable & independent variable into training and test dataset and Predicting the Dependent Variable by fitting the given model and create the pickle file of the model with the given Algo_name. Further getting the Algorithm Name, Best Parameters of the algorithm, Best Estimators of the fitted model, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.Further visualising the confusion matrix through Seaborn Heat Map**_

# In[26]:


def Fitmodel(x, y, algo_name, algorithm, params, cv):
    np.random.seed(10)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state = 16)
    RSC = RandomizedSearchCV(algorithm, params, n_iter = 100, scoring = 'accuracy', n_jobs = -1, cv = cv, verbose = 0)
    model = RSC.fit(x_train, y_train)
    pred = model.predict(x_test)
    best_params = model.best_params_
    best_estimator = model.best_estimator_
    pickle.dump(model, open(algo_name,'wb'))
    cm = confusion_matrix(pred, y_test)
    print('Algorithm Name : ',algo_name,'\n')
    print('Best Params : ',best_params,'\n')
    print('Best Estimator : ',best_estimator,'\n')
    print('Percentage of Accuracy Score : {0:.2f} %'.format(100*(accuracy_score(y_test,pred))),'\n')
    print('Classification Report : \n',classification_report(y_test,pred))
    print('Confusion Matrix : \n',cm,'\n')
    plt.figure(figsize = (3,3))
    sns.heatmap(cm, annot = True, cbar = True, square = True)
    plt.show()


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> Logistic Regression </span> </span> model doesn't need any special parameters and fitting the Logistic Regression Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Best Estimators of the fitted model, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.Further visualising the confusion matrix through Seaborn Heat Map and also creating the pickle file with the name Logistic Regression**_

# In[27]:


params = {}
Fitmodel(x,y,'Logistic Regression',LogisticRegression(), params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> SVC </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Best Estimators of the fitted model, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.Further visualising the confusion matrix through Seaborn Heat Map and also creating the pickle file with the name SVC**_

# In[28]:


params = {'C' : [0.1, 1, 10, 100],
          'gamma' : [0.001,0.01,0.1,1],
         'kernel' : ['rbf', 'linear']}
Fitmodel(x,y,'SVC',SVC(),params,cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> KNeighbors Classifier </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Best Estimators of the fitted model, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.Further visualising the confusion matrix through Seaborn Heat Map and also creating the pickle file with the name KNeighbors**_

# In[29]:


params = {'n_neighbors' : [5,10,15,20,25,30,35,40],
           'p' : [1,2]}
Fitmodel(x,y,'KNeighbors',KNeighborsClassifier(), params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> Decision Tree Classifier </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Best Estimators of the fitted model, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.Further visualising the confusion matrix through Seaborn Heat Map and also creating the pickle file with the name Decision Tree**_

# In[30]:


params = {'criterion' : ['gini','entropy'],
          'max_features' : ['auto','sqrt'],
          'splitter' : ['best','random']}
Fitmodel(x,y,'Decision Tree',DecisionTreeClassifier(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> Random Forest Classifier </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Best Estimators of the fitted model, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.Further visualising the confusion matrix through Seaborn Heat Map and also creating the pickle file with the name Random Forest**_

# In[31]:


params = {'n_estimators' : [111,222,333,444,555],
          'criterion' : ['entropy','gini'],
          'max_features' : ['auto','sqrt']}
Fitmodel(x,y,'Random Forest',RandomForestClassifier(), params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> Extra Trees Classifier </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Best Estimators of the fitted model, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.Further visualising the confusion matrix through Seaborn Heat Map and also creating the pickle file with the name Extra Trees Classifier**_

# In[32]:


params = {'n_estimators' : [111,222,333,444,555],
          'criterion' : ['entropy','gini'],
          'max_features' : ['auto','sqrt']}
Fitmodel(x,y,'Extra Trees Classifier',ExtraTreesClassifier(),params, cv = 10)


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> Gaussian Naive Bayes </span> </span> model doesn't need any special parameters and fitting the Gaussian Naive Bayes Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Best Estimators of the fitted model, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.Further visualising the confusion matrix through Seaborn Heat Map and also creating the pickle file with the name GaussianNB**_

# In[33]:


params = {}
Fitmodel(x,y,'GaussianNB',GaussianNB(),params, cv = 10)


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> Bernoulli Naive Bayes </span> </span> model doesn't need any special parameters and fitting the Bernoulli Naive Bayes Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Best Estimators of the fitted model, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.Further visualising the confusion matrix through Seaborn Heat Map and also creating the pickle file with the name BernoulliNB**_

# In[34]:


params = {}
Fitmodel(x,y,'BernoulliNB',BernoulliNB(),params, cv = 10)


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> Multinomial Naive Bayes </span> </span> model doesn't need any special parameters and fitting the Multinomial Naive Bayes Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Best Estimators of the fitted model, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.Further visualising the confusion matrix through Seaborn Heat Map and also creating the pickle file with the name MultinomialNB**_

# In[35]:


params = {}
Fitmodel(x,y,'MultinomialNB',MultinomialNB(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> XGB Classifier </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Best Estimators of the fitted model, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.Further visualising the confusion matrix through Seaborn Heat Map and also creating the pickle file with the name XGB Classifier**_

# In[36]:


params = {'n_estimators' : [111,222,333,444,555]}
Fitmodel(x,y,'XGB Classifier',XGBClassifier(),params,cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> CatBoost Classifier </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Best Estimators of the fitted model, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.Further visualising the confusion matrix through Seaborn Heat Map and also creating the pickle file with the name CatBoost**_

# In[37]:


params = {'verbose' : [0]}
Fitmodel(x,y,'CatBoost', CatBoostClassifier(),params, cv = 10)


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> LightGBM Classifier </span> </span> model doesn't need any special parameters and fitting the LightGBM Classifier\ Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Best Estimators of the fitted model, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.Further visualising the confusion matrix through Seaborn Heat Map and also creating the pickle file with the name LightGBM**_

# In[38]:


params = {}
Fitmodel(x,y,'LightGBM',LGBMClassifier(),params, cv = 10)


# _**Loading the pickle file with the algorithm which gives highest accuracy score**_

# In[39]:


model = pickle.load(open('GaussianNB','rb'))


# _**For Further Predictions, use the model loaded with the pickle file**_
