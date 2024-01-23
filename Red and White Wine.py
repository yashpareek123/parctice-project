#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter("ignore")
import joblib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV 


# In[2]:


data=pd.read_csv('https://raw.githubusercontent.com/FlipRoboTechnologies/ML-Datasets/main/Red%20Wine/winequality-red.csv')
data


# # Exploratory Data Analysis

# In[3]:


data.shape


# In[4]:


data.isnull().sum()


# In[5]:


data.info()


# In[6]:


#All dtypes are simliar and label is in int
# Here we can see there is no null value as well as missing values
#Now we are going to see the statistics of data set by using describe method


# In[7]:


data.describe()

According to my observation
1-There is very low difference in between 25% to 50% and as well compare to 75%(In fixed acidity,residual sugar,chlorides etc.)
2-There is big gap between 75% and max value(In free sulfur dioxide,residual sugar)
3-Mean and std looks like have some issue.The data is not distributed normally.
4-There are some outliers need to treated to predict  model best accuracy.
# # Now we can see outlier by using boxplot that shows outlier present in data set.

# In[8]:


plt.figure(figsize=(20,15))
plotnumber=1
for column in data:
    if plotnumber<=13:
        plt.subplot(3,6,plotnumber)
        ax=sns.boxplot(data[column])
        plt.xlabel(column,fontsize=20)
    plotnumber+=1
plt.tight_layout()    

# Here we can see there are outliers in each classes,before treating them we are going to see realtionship between featuers and
target label.
For understanding hoe many features are realted to label 
# In[9]:


index=0
labels = data['quality']
features = data.drop('quality', axis=1)

for col in features.items():
    plt.figure(figsize=(10,2))
    sns.barplot(x=labels, y=col[index], data=data, color="deeppink")
plt.show()


# we can see here citric acid,sulphate and alcohol have strong realtion to the label because they are going in upward
# direction and all the remaining are increasing or decreasing thats shows data is imbalanced
# 

# # outliers remove

# In[10]:


data.shape


# In[ ]:





# In[11]:


#check data is balanced or imbalanced by using countpolt
sns.countplot(x='quality',data=data)
plt.show()
data.quality.value_counts()


# In[12]:


#here we can see our data set is imbalaced ,it willbe baised when prediction we need to treat it accordingly
#using Z score for removing outliers


# In[13]:


z=np.abs(zscore(data))
threshold=3
np.where(z>3)

data=data[(z<3).all(axis=1)]
data


# In[14]:


data.shape


# In[15]:


data.quality.value_counts()


# In[16]:


#Here we can see that our label column data is imbalanced we need to treat it by using oversampling method


# In[17]:


#now splitting data into two parts
X = data.drop('quality', axis=1)
Y = data['quality']


# In[18]:


# adding samples to make all the categorical quality values same

oversample = SMOTE()
X, Y = oversample.fit_resample(X, Y)


# In[19]:


Y.value_counts()


# In[20]:


data.shape


# In[21]:


Y = Y.apply(lambda y_value:1 if y_value>=7 else 0) # 1 is for good quality and 0 for bad (not good) quality
Y # Displaying the label after applying label binarization


# In[22]:


scaler = StandardScaler()
X_scaled=scaler.fit_transform(X)


# In[23]:


X_scaled.shape[1]


# In[24]:


#checking multicollinearity by using vif score


# In[25]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[26]:


vif=pd.DataFrame()
vif['vif']=[variance_inflation_factor(X_scaled,i)for i in range(X_scaled.shape[1])]
vif['features']=X.columns
vif


# In[27]:


#as we can see that there are multicollinearity we going to remove those features are more score than 5
X=X.drop('fixed acidity',axis=1)


# In[28]:


X=X.drop('density',axis=1)


# In[29]:


X


# In[30]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=21)


# In[31]:


# Classification Model Function

def classify(model, X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
    
    # Training the model
    model.fit(X_train, Y_train)
    
    # Predicting Y_test
    pred = model.predict(X_test)
    
    # Accuracy Score
    acc_score = (accuracy_score(Y_test, pred))*100
    print("Accuracy Score:", acc_score)
    
    # Classification Report
    class_report = classification_report(Y_test, pred)
    print("\nClassification Report:\n", class_report)
    
    # Cross Validation Score
    cv_score = (cross_val_score(model, X, Y, cv=5).mean())*100
    print("Cross Validation Score:", cv_score)
    
    # Result of accuracy minus cv scores
    result = acc_score - cv_score
    print("\nAccuracy Score - Cross Validation Score is", result)


# In[32]:


# Logistic Regression

model=LogisticRegression()
classify(model, X, Y)


# In[33]:


# Support Vector Classifier

model=SVC(C=1.0, kernel='rbf', gamma='auto', random_state=42)
classify(model, X, Y)


# In[34]:


# Decision Tree Classifier

model=DecisionTreeClassifier(random_state=21, max_depth=15)
classify(model, X, Y)


# In[35]:


# Random Forest Classifier

model=RandomForestClassifier(max_depth=15, random_state=111)
classify(model, X, Y)


# In[36]:


# K Neighbors Classifier

model=KNeighborsClassifier(n_neighbors=15)
classify(model, X, Y)


# In[37]:


# Extra Trees Classifier

model=ExtraTreesClassifier()
classify(model, X, Y)


# In[38]:


# Choosing Support Vector Classifier

svc_param = {'kernel' : ['poly', 'sigmoid', 'rbf'],
             'gamma' : ['scale', 'auto'],
             'shrinking' : [True, False],
             'random_state' : [21,42,104],
             'probability' : [True, False],
             'decision_function_shape' : ['ovo', 'ovr'],
             'verbose' : [True, False]}


# In[39]:


GSCV = GridSearchCV(SVC(), svc_param, cv=5)


# In[ ]:


GSCV.fit(X_train,Y_train)


# In[ ]:


GSCV.best_params_ 


# In[ ]:


Final_Model = SVC(decision_function_shape='ovo', gamma='scale', kernel='rbf', probability=True, random_state=21,
                 shrinking=True, verbose=True)
Classifier = Final_Model.fit(X_train, Y_train)
fmod_pred = Final_Model.predict(X_test)
fmod_acc = (accuracy_score(Y_test, fmod_pred))*100
print("Accuracy score for the Best Model is:", fmod_acc)


# In[ ]:


fpr,tpr,threshold=roc_curve(Y_test,fmod_pred)


# In[ ]:




