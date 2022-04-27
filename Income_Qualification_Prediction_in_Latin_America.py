#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Import The Data

# In[3]:


df_train=pd.read_csv("C:\\Users\\az\\Documents\\Data_Science_with_python\\Project\\Dataset for the project\\train.csv")
df_test=pd.read_csv("C:\\Users\\az\\Documents\\Data_Science_with_python\\Project\\Dataset for the project\\test.csv")


# # Understand The Data

# In[4]:


df_train.head()


# In[5]:


df_train.info()


# In[6]:


df_test.head()


# In[7]:


df_test.info()


# The important piece of information here is that we don’t have ‘Target’ feature in Test Dataset. There are 3 Types of the features:
# 
# 5 object type,
# 130(Train set)/ 129 (test set) integer type,
# 8 float type.
# 
# Lets analyze features:

# In[8]:


print('Integer Type: ')
print(df_train.select_dtypes(np.int64).columns)
print('\n')
print('Float Type: ')
print(df_train.select_dtypes(np.float64).columns)
print('\n')
print('Object Type: ')
print(df_train.select_dtypes(np.object).columns)


# In[9]:


df_train.select_dtypes('int64').head()


# In[10]:


df_train.select_dtypes('int64').isnull().sum()


# In[11]:


df_train.select_dtypes('float64').head()


# In[12]:


df_train.select_dtypes('float64').isnull().sum()


# In[13]:


df_train.select_dtypes('object').head()


# In[14]:


df_train.select_dtypes('object').isnull().sum()


# Looking at the different type of Data types and null values for each features. We found following:

# 1. There is no null values for object type features.
# 2. There is no null values for integer type features.
# 3.For float type features following features has null values:
# 
#     A. v2a1               6860
#     
#     B. v18q1              7342
#     
#     C. rez_esc            7928
#     
#     D. meaneduc              5
#     
#     E. SQBmeaned             5
#     
#     
#  We also noticed that object types features dependency, edjefe, edjefa has mixed values.
#  
# Lets fixed the data for features with null values and mixed values.

# # Data Cleaning

# Lets fix the columns with mixed values:
#     
#  dependency, Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64) 
# 
# edjefe: years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0 
# 
# edjefa: years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0 
# 
# For these three variables, it seems “yes” = 1 and “no” = 0. We can correct the variables using a mapping and convert to floats.

# In[15]:


mapping={'yes':1, 'no':0}

for df in [df_train, df_test]:
    df['dependency']=df['dependency'].replace(mapping).astype(np.float64)
    df['edjefe']=df['edjefe'].replace(mapping).astype(np.float64)
    df['edjefa']=df['edjefa'].replace(mapping).astype(np.float64)
    
df_train[['dependency', 'edjefe', 'edjefa']].describe()  


# Lets fix the columns with null values:
#     
# v2a1: Monthly rent payment
# 
# v18q1: number of tablets household owns
# 
# rez_esc: Years behind in school 
# 
# meaneduc: average years of education for adults (18+) 
# 
# SQBmeaned: square of the mean years of education of adults (>=18) in the household
#     
#     

# Lets look at v2a1 (total nulls: 6860) : Monthly rent payment
# 
# 
# Other columns related to monthly rent payment are:
#     
#         1. tipovivi1 =1 own and fully paid house 
#         2. tipovivi2 =1 own,  paying in installments
#         3. tipovivi3 =1 rented
#         4. tipovivi4 =1 precarious
#         5. tipovivi5 =1 other(assigned,  borrowed)
# 
# 

# In[16]:


data = df_train[df_train['v2a1'].isnull()].head()

columns = ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']
data[columns]
   


# In[17]:


own_var = [i for i in df_train if i.startswith('tipo')]

df_train.loc[df_train['v2a1'].isnull(), own_var].sum().plot.bar(figsize=(10,8), color='c', edgecolor='k',linewidth=3)

plt.xticks([0,1,2,3,4], ['Owns and paid off', 'Owns and paying', 'Rented', 'Precarious','Other'], rotation=45)

plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18)
plt.show()


# Looking at the above data we can say that when the house is fully paid, there will be no monthly rent, so we can add 0 to all null values for v2a1 column. 

# In[18]:


for df in [df_train, df_test]:
    df['v2a1'].fillna(value=0, inplace=True)
    
df_train[['v2a1']].isnull().sum()    


# Lets look at v18q1 (total nulls: 7342) : number of tablets household owns
# 
# Columns related to number of tablets household owns:
#     
#     v18q: owns a tablet
# 
# Since this is a household variable, it only makes sense to look at it on a household level, so we'll only select the rows for the head of household.
# 
# 
# 
# 

# In[19]:


heads = df_train.loc[df_train['parentesco1'] == 1].copy()

heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())


# In[20]:


plt.figure(figsize = (8, 6))
col='v18q1'
df_train[col].value_counts().sort_index().plot.bar(color = 'm', edgecolor = 'k',linewidth = 2)
plt.xlabel(f'{col}')
plt.title(f'{col} Value Counts')
plt.ylabel('Count')
plt.show()


# Looking at the above data it makes sense that when owns a tablet column is 0, there will be no number of tablets household owns. Lets add 0 for all the null values.
# 
# 

# In[21]:


for df in [df_train, df_test]:
    df['v18q1'].fillna(value=0, inplace=True)

df_train[['v18q1']].isnull().sum()


# Lets look at rez_esc (total nulls: 7928) : Years behind in school
# 
# Columns related to Years behind in school
# 
# Age in years

# In[22]:


df_train[df_train['rez_esc'].notnull()]['age'].describe()


# From the above , we see that when min age is 7 and max age is 17 for Years, then the 'behind in school' column has a value.
# 
# Lets confirm
# 
# 

# In[23]:


df_train.loc[df_train['rez_esc'].isnull()]['age'].describe()


# In[24]:


df_train.loc[(df_train['rez_esc'].isnull() & ((df_train['age'] > 7) & (df_train['age'] < 17)))]['age'].describe()


# In[25]:


df_train[(df_train['age'] ==10) & df_train['rez_esc'].isnull()].head()
df_train[(df_train['Id'] =='ID_f012e4242')].head()


# In[26]:


for df in [df_train, df_test]:
    df['rez_esc'].fillna(value=0, inplace=True)
df_train[['rez_esc']].isnull().sum()


# Lets look at meaneduc (total nulls: 5) : average years of education for adults (18+)
# 
# 
# Columns related to average years of education for adults (18+):
# 
# edjefe: years of education of male head of household, based on the interaction of escolari (years of education),
# head of household and gender, yes=1 and no=0
# 
# edjefa: years of education of female head of household, based on the interaction of escolari (years of education),
# head of household and gender, yes=1 and no=0
# 
# instlevel1: =1 no level of education
# 
# instlevel2: =1 incomplete primary
# 
# 

# In[27]:


data = df_train[df_train['meaneduc'].isnull()].head()

columns=['edjefe','edjefa','instlevel1','instlevel2']
data[columns][data[columns]['instlevel1']>0].describe()


# from the above, we find that meaneduc is null when no level of education is 0
# 
# Lets fix the data
# 

# In[28]:


for df in [df_train, df_test]:
    df['meaneduc'].fillna(value=0, inplace=True)
df_train[['meaneduc']].isnull().sum()


# Lets look at SQBmeaned (total nulls: 5) : square of the mean years of education of adults (>=18) in the household 142
# 
# Columns related to average years of education for adults (18+):
#     
# edjefe: years of education of male head of household, based on the interaction of escolari (years of education),
# head of household and gender, yes=1 and no=0
# 
# edjefa: years of education of female head of household, based on the interaction of escolari (years of education),
# head of household and gender, yes=1 and no=0
# 
# instlevel1: =1 no level of education
#     
# instlevel2: =1 incomplete primary

# In[29]:


data = df_train[df_train['SQBmeaned'].isnull()].head()

columns=['edjefe','edjefa','instlevel1','instlevel2']
data[columns][data[columns]['instlevel1']>0].describe()


# from the above, we find that SQBmeaned is null when no level of education is 0
# 
# Lets fix the data
# 

# In[30]:


for df in [df_train, df_test]:
    df['SQBmeaned'].fillna(value=0, inplace=True)
df_train[['SQBmeaned']].isnull().sum()


# Lets look at the overall data
# 

# In[31]:


null_counts = df_train.isnull().sum()
null_counts[null_counts > 0].sort_values(ascending=False)


# In[32]:


all_equal = df_train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
not_equal = all_equal[all_equal != True]

print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))


# In[33]:


df_train[df_train['idhogar'] == not_equal.index[0]][['idhogar', 'parentesco1', 'Target']]


# In[34]:


#Lets use Target value of the parent record (head of the household) and update rest. But before that lets check
# if all families has a head. 

households_head = df_train.groupby('idhogar')['parentesco1'].sum()

# Find households without a head
households_no_head = df_train.loc[df_train['idhogar'].isin(households_head[households_head == 0].index), :]

print('There are {} households without a head.'.format(households_no_head['idhogar'].nunique()))


# In[35]:


# Find households without a head and where Target value are different

households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

print('{} Households with no head have different Target value.'.format(sum(households_no_head_equal == False)))


# In[36]:


#Lets fix the data
#Set poverty level of the members and the head of the house within a family.
# Iterate through each household
for household in not_equal.index:
    # Find the correct label (for the head of household)
    true_target = int(df_train[(df_train['idhogar'] == household) & (df_train['parentesco1'] == 1.0)]['Target'])
    
    # Set the correct label for all members in the household
    df_train.loc[df_train['idhogar'] == household, 'Target'] = true_target
    
    
# Groupby the household and figure out the number of unique values
all_equal = df_train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))


# Lets look at the dataset and plot head of household and Target
# 
# 

# In[37]:


# 1 = extreme poverty 2 = moderate poverty 3 = vulnerable households 4 = non vulnerable households 
target_counts = heads['Target'].value_counts().sort_index()
target_counts


# In[38]:


target_counts.plot.bar(figsize = (8, 6),linewidth = 2,edgecolor = 'k',title="Target vs Total_Count")
plt.show()


# extreme poverty is the smallest count in the train dataset. The dataset is biased.
# 
# Lets look at the Squared Variables:
# 
# ‘SQBescolari’
# 
# ‘SQBage’
# 
# ‘SQBhogar_total’
# 
# ‘SQBedjefe’
# 
# ‘SQBhogar_nin’
# 
# ‘SQBovercrowding’
# 
# ‘SQBdependency’
# 
# ‘SQBmeaned’
# 
# ‘agesq’

# In[39]:


#Lets remove them
print(df_train.shape)
cols=['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 
        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']


for df in [df_train, df_test]:
    df.drop(columns = cols,inplace=True)

print(df_train.shape)


# In[40]:


id_ = ['Id', 'idhogar', 'Target']

ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone']

ind_ordered = ['rez_esc', 'escolari', 'age']

hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']


# In[41]:


#Check for redundant household variables
heads = df_train.loc[df_train['parentesco1'] == 1, :]
heads = heads[id_ + hh_bool + hh_cont + hh_ordered]
heads.shape


# In[42]:


# Create correlation matrix
corr_matrix = heads.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop


# In[43]:


['coopele', 'area2', 'tamhog', 'hhsize', 'hogar_total']


# In[44]:


corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9]


# In[45]:


sns.heatmap(corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9], annot=True, cmap = plt.cm.Accent_r, fmt='.3f');


# There are several variables here having to do with the size of the house:
#     
# r4t3: Total persons in the household
#     
# tamhog: size of the household
#     
# tamviv: number of persons living in the household
#     
# hhsize: household size
#     
# hogar_total: # of total individuals in the household
#     
# These variables are all highly correlated with one another.

# In[46]:


cols=['tamhog', 'hogar_total', 'r4t3']
for df in [df_train, df_test]:
    df.drop(columns = cols,inplace=True)

df_train.shape


# In[47]:


#Check for redundant Individual variables

ind = df_train[id_ + ind_bool + ind_ordered]
ind.shape


# In[48]:


# Create correlation matrix
corr_matrix = ind.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop


# In[49]:


# This is simply the opposite of male! We can remove the male flag.
for df in [df_train, df_test]:
    df.drop(columns = 'male',inplace=True)

df_train.shape


# In[50]:


#lets check area1 and area2 also
# area1, =1 zona urbana 
# area2, =2 zona rural 
#area2 redundant because we have a column indicating if the house is in a urban zone

for df in [df_train, df_test]:
    df.drop(columns = 'area2',inplace=True)

df_train.shape


# In[51]:


#Finally lets delete 'Id', 'idhogar'
cols=['Id','idhogar']
for df in [df_train, df_test]:
    df.drop(columns = cols,inplace=True)

df_train.shape


# # Predict the accuracy using random forest classifier
# 

# In[52]:


df_train.iloc[:,0:-1]


# In[53]:


df_train.iloc[:,-1]


# In[54]:


x_features=df_train.iloc[:,0:-1] # feature without target
y_features=df_train.iloc[:,-1] # only target
print(x_features.shape)
print(y_features.shape)


# In[55]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report

x_train,x_test,y_train,y_test=train_test_split(x_features,y_features,test_size=0.2,random_state=1)
rmclassifier = RandomForestClassifier()


# In[56]:


rmclassifier.fit(x_train,y_train)


# In[57]:


y_predict = rmclassifier.predict(x_test)


# In[58]:


print(accuracy_score(y_test,y_predict))


# In[59]:


print(confusion_matrix(y_test,y_predict))


# In[60]:


print(classification_report(y_test,y_predict))


# In[61]:


y_predict_testdata = rmclassifier.predict(df_test)

y_predict_testdata


# # Check the accuracy using random forest with cross validation.
# 

# In[62]:


from sklearn.model_selection import KFold,cross_val_score


# In[63]:


seed=7
kfold=KFold(n_splits=5,random_state=seed,shuffle=True)

rmclassifier=RandomForestClassifier(random_state=10,n_jobs = -1)
print(cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy'))
results=cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy')
print(results.mean()*100)


# In[64]:


num_trees= 100

rmclassifier=RandomForestClassifier(n_estimators=100, random_state=10,n_jobs = -1)
print(cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy'))
results=cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy')
print(results.mean()*100)


# In[65]:


rmclassifier.fit(x_features,y_features)

labels = list(x_features)
feature_importances = pd.DataFrame({'feature': labels, 'importance': rmclassifier.feature_importances_})
feature_importances=feature_importances[feature_importances.importance>0.015]
feature_importances.head()


# In[66]:


y_predict_testdata = rmclassifier.predict(df_test)
y_predict_testdata


# In[67]:


feature_importances.sort_values(by=['importance'], ascending=True, inplace=True)
feature_importances['positive'] = feature_importances['importance'] > 0
feature_importances.set_index('feature',inplace=True)
feature_importances.head()

feature_importances.importance.plot(kind='barh', figsize=(11, 6),color = feature_importances.positive.map({True: 'green', False: 'red'}))
plt.xlabel('Importance')


# From the above figure we can assume that meaneduc,dependency,overcrowding has significant influence on the model.
# 

# In[ ]:




