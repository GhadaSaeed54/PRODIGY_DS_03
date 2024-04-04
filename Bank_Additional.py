#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r'C:\Users\ascom\Desktop\prodigy_info_tech\Task_3\bank-additional\bank-additional\bank-additional.csv',delimiter=';')
df


# In[4]:


df.columns


# In[6]:


df.rename(columns={'y':'deposit'},inplace=True)
df.head()


# In[8]:


df.shape


# In[7]:


df.info()


# In[9]:


df.dtypes


# In[10]:


df.dtypes.value_counts()


# In[11]:


df.isnull().sum()


# In[14]:


df.duplicated().sum()


# In[19]:


df.describe()


# In[32]:


df.hist(figsize=(20,20),color = 'green')
plt.show


# In[52]:


categorical_columns = df.select_dtypes(include = 'object').columns
print(categorical_columns)

numerical_columns = df.select_dtypes(exclude='object').columns
print(numerical_columns)


# In[20]:


df.describe(include='object')


# In[39]:


for attribute in categorical_columns:
    plt.figure(figsize=(7,7))
    sns.countplot(x= attribute,data= df)
    plt.title(f'barplot of {attribute}')
    plt.xlabel(attribute)
    plt.ylabel('count')
    plt.xticks(rotation=90)
    plt.show()


# In[44]:


df.plot(kind='box',subplots=True,layout=(2,5),figsize=(20,10))
plt.show()


# In[46]:


columns = df[['age','duration','campaign']]
q1 = np.percentile(columns,25)
q3 = np.percentile(columns,75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df[['age','duration','campaign']] = columns[(columns > lower_bound) & (columns < upper_bound)]


# In[47]:


df.plot(kind='box' , subplots=True,layout=(2,5),figsize=(20,10))
plt.show()


# In[54]:


numerical_data = df.select_dtypes(exclude='object')
print(numerical_data)


# In[55]:


corr = numerical_data.corr()
print(corr)
corr = corr[abs(corr)>=0.90]
sns.heatmap(corr,annot=True,cmap='Set3',linewidths=0.2)
plt.show()


# In[62]:


high_corr_columns = ['emp.var.rate','euribor3m','nr.employed']
df_new = df.copy()


# In[63]:


df_new.columns


# In[66]:


df_new.drop(high_corr_columns,axis=1,inplace= True)


# In[67]:


df_new.columns


# In[68]:


df_new.shape


# In[72]:


from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
encoded_df = df_new.apply(lb.fit_transform)
encoded_df


# In[74]:


x = encoded_df.drop('deposit',axis = 1)
y = encoded_df['deposit']


# In[76]:


print(x.shape)
print(y.shape)


# In[79]:


encoded_df['deposit'].value_counts()


# In[80]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.30)


# In[81]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='gini',max_depth = 5 , min_samples_split=10)
dt.fit(x_train,y_train)


# In[82]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

train_score = dt.score(x_train,y_train)
test_score = dt.score(x_test,y_test)
print('training score: ', train_score)
print('testing score: ', test_score)


# In[83]:


y_pred = dt.predict(x_test)
print(y_pred)


# In[85]:


# evaluation of model

acc = accuracy_score(y_test,y_pred)
conv_mat = confusion_matrix(y_test,y_pred)
class_repo = classification_report(y_test,y_pred)

print('accuracy score: ', acc)
print()
print('confusion matrix\n', conv_mat)
print()
print('Classification Report\n', class_repo)


# In[113]:


from sklearn.tree import plot_tree

cl=['no','yes']

plt.figure(figsize=(20,20))
plot_tree(dt , class_names=cl,filled=True)
plt.show


# In[108]:


df1 = DecisionTreeClassifier(criterion='entropy',max_depth=4,min_samples_split=20)
df1.fit(x_train,y_train)


# In[109]:


train_score = dt.score(x_train,y_train)
test_score = dt.score(x_test,y_test)
print('training score: ', train_score)
print('testing score: ', test_score)


# In[110]:


y_pred = dt.predict(x_test)
print(y_pred)


# In[111]:


# evaluation of model

acc = accuracy_score(y_test,y_pred)
conv_mat = confusion_matrix(y_test,y_pred)
class_repo = classification_report(y_test,y_pred)

print('accuracy score: ', acc)
print()
print('confusion matrix\n', conv_mat)
print()
print('Classification Report\n', class_repo)


# In[112]:


plt.figure(figsize=(20,20))
plot_tree(df1,class_names=cl,filled=True)
plt.show


# In[ ]:




