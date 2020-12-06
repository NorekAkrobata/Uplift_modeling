# Uplift modeling - marketing dataset

# Libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Loading data

PATH = r'C:\Users\Norbert\Desktop\Empik'
FILENAME = r'bank_data_prediction_task.csv'

def load_data(path=PATH, filename=FILENAME):
    """Return dataframe from .csv file

    :param path: path to the folder
    :param filename: name of the .csv file
    :return: dataframe
    """
    csv_path = os.path.join(path,filename)
    return pd.read_csv(csv_path, index_col = 0)

df = load_data()

# Data preprocessing

cat_cols = ['contact', 'month', 'day_of_week']
num_cols = ['duration', 'campaign']

for i in cat_cols:
    df[i] = df[i].fillna(df['test_control_flag'].map({'control group': 'None'}))
for j in num_cols:
    df[j] = df[j].fillna(df['test_control_flag'].map({'control group': 0}))

column1 = 'cons.price.idx'
column2 = 'cons.conf.idx'
list_price = df[column1].unique()
list_conf = df[column2].unique()

def checker(list=list_price, column1=column1, column2=column2):
    """Check if for unique value in column1, there is specific unique value on column2

    :param list: unique values in column1
    :param column1: 1st column
    :param column2: 2nd column
    :return: print(True/False)
    """
    list_min, list_max = [], []
    for i in list:
        a = df[df[column1] == i]
        list_min.append(a[column2].min())
        list_max.append(a[column2].max())
    print("Do we have unique value in column {} for unique value in column {}? {}".format(column1, column2,
                                                                                          list_min == list_max))
checker()

dictionary = dict(zip(list_conf, list_price))
df['cons.price.idx'] = df['cons.price.idx'].fillna(df['cons.conf.idx'].map(dictionary))
print("We currently have {} non null values in column {}".format(df['cons.price.idx'].count(),"cons.price.idx"))

df['pdays']=np.where(df['pdays']<=7, 'Groupd_A', df['pdays'])
df['pdays'] = df['pdays'].replace('999', 'Group_D')
df['pdays'] = df['pdays'].replace(('8','9','10','11','12','13','14'), 'Group_B')
df['pdays'] = df['pdays'].replace(('15','16','17','18','19','20','21','22','25','26','27'), 'Group_C')
df['pdays'].unique()

print(df[df['age'] < 18])
df = df[df['age'] >= 18]
print('We currently have {} values in column {}'.format(df['age'].count(), 'age'))

numeric_list = ['age','campaign','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
categorical_list = ['job','marital','education','default','housing','loan','contact','pdays','month','day_of_week','poutcome','y','test_control_flag']


def hist_numeric(feature, dataframe=df, bins=50, target_col='y'):
    """Creates histogram from numeric columns according to the different target values (yes/no)

    :param feature: Column name
    :param dataframe: Dataframe
    :param bins: Number of bins to divide values
    :param target_col: Target value column name
    :return: Plot
    """
    df1 = dataframe[dataframe[target_col] == 'no']
    df2 = dataframe[dataframe[target_col] == 'yes']
    plt.figure()
    plt.hist(df1[feature], bins=bins, label='no')
    plt.hist(df2[feature], bins=bins, label='yes')
    plt.legend()
    plt.title('Feature: {}'.format(feature, ))
    plt.xlabel('Unique values')
    plt.ylabel('Count')

def percentage_num(feature='age', dataframe=df,target_col='y', bins=20):
    """Creates list with percentage of positive answer (yes) to the total number of answers (yes/no) according to created bins

    :param feature: Column name
    :param dataframe: Dataframe
    :param target_col: Target values column name
    :param bins: Number of bins to divide data
    :return: List
    """
    feature_bins = feature+'_bins'
    df[feature_bins] = pd.qcut(dataframe[feature], q=bins)
    df1 = dataframe[dataframe[target_col] == 'yes']
    return df1[feature_bins].value_counts()/df[feature_bins].value_counts()*100

def percentage_val(feature='age', dataframe=df,target_col='y', amount=10):
    """Creates list with percentage of positive answer (yes) to the total number of answers (yes/no) for specific amount of values

    :param feature: Column name
    :param dataframe: Dataframe
    :param target_col: Target values column name
    :param amount: Amount of values to present
    :return: List
    """
    df = dataframe[dataframe[target_col] == 'yes']
    return (df[feature].value_counts()/dataframe[feature].value_counts()*100).head(amount)


def hist_categorical(feature, dataframe=df, target_col='y'):
    """Creates histogram from categorical columns according to the different target values (yes/no).

    :param feature: Column name
    :param dataframe: Dataframe
    :param target_col: Target value column name
    :return: Plot
    """
    final_df = pd.DataFrame()
    df = dataframe[[feature, target_col]]
    df1 = df[df[target_col] == 'no']
    df2 = df[df[target_col] == 'yes']
    final_df['no'] = df1[feature].value_counts()
    final_df['yes'] = df2[feature].value_counts()
    final_df.plot(kind='bar')
    plt.title('Feature: {}'.format(feature, ))
    plt.ylabel('Count')
    plt.xlabel

def percentage_cat(feature,dataframe=df,target_col='y'):
    """Creates list with percentage of positive answer (yes) to the total number of answers (yes/no) according to unique values present

    :param feature: Column name
    :param dataframe: Dataframe
    :param target_col: Target values column name
    :return: List
    """
    df = dataframe[[feature, target_col]]
    df1 = df[df[target_col] == 'yes']
    return (df1[feature].value_counts()/df[feature].value_counts()*100).sort_values(ascending=False)

# Data visualizations

# Age
hist_numeric('age')
percentage_num()
df = df.drop(['age_bins'], axis=1) #Remove columns created by 'percentage_num' function.

#Job
hist_categorical('job')
percentage_cat('job')

#Marital
hist_categorical('marital')
percentage_cat('marital')

#Education
hist_categorical('education')
percentage_cat('education')
print('Number of entries with illiterate education: {}'.format(df[df['education'] == 'illiterate'].shape[0],))
print('Number of entries with unknown education: {}'.format(df[df['education'] == 'unknown'].shape[0],))

#Credit defaults
hist_categorical('default')
percentage_cat('default')

#Housing loan
hist_categorical('housing')
percentage_cat('housing')

#Personal loan
hist_categorical('loan')
percentage_cat('loan')

#Way of contact
hist_categorical('contact')
percentage_cat('contact')

#Month of contact
hist_categorical('month')
percentage_cat('month')
print('Number of entries with contact in apr, oct, sep, mar, dec: {}'.format(df[df['month'].isin(['oct','apr','sep','mar','dec'])].shape[0],))

#Weekday of contact
hist_categorical('day_of_week')
percentage_cat('day_of_week')

#Duration of last call
hist_numeric('duration', bins=50)

#Amount of contacts during this campaign
hist_numeric('campaign', bins=25)
percentage_val('campaign')

#Pdays
hist_categorical('pdays')
percentage_cat(feature='pdays')

#Amount of previous contacts
hist_categorical('previous')
percentage_cat('previous')
df[df['previous'] == 1].shape[0]

#Outcome of previous campaign
hist_categorical('poutcome')
percentage_cat('poutcome')
df[df['poutcome'] == 'success'].shape[0]

#Employment variation rate
hist_categorical('emp.var.rate')
percentage_val('emp.var.rate')

#cons.price.idx
hist_numeric('cons.price.idx', bins=26)
percentage_val('cons.price.idx',amount=26)

#euribor3m
hist_numeric('euribor3m', bins=50)

#nr.employed
hist_numeric('nr.employed', bins=11)
percentage_val('nr.employed', amount=11)

#Control group flag
print(df['test_control_flag'].value_counts())
hist_categorical('test_control_flag')
percentage_val('test_control_flag')

#Subscription made - y/n
print(df['y'].value_counts())

df1 = df

#Columns for boolean,dummy variables
boolean = ('y', 'test_control_flag')
categorical_cols = ['job','marital','education','default','housing','pdays','loan','month','day_of_week','poutcome','contact']

#Boolean
df1['y'] = df1['y'].map({'no':0,'yes':1})
df1['test_control_flag'] = df1['test_control_flag'].map({'control group':0,'campaign group':1})

#Dummy
df_final = pd.get_dummies(df1,columns=categorical_cols)

#Check linear correlation between target value 'y' and rest of the features.
correlation_matrix = df_final.corr()
corr_list = correlation_matrix['y'].abs().sort_values(ascending=False)
#print(corr_list)
print(corr_list[1:11])

#Create target class column according to target value and control/campaign group for uplift modeling
""" Control Non-Responders(CN)
    Customers that don't make a subscription without an offer (value = 0)
    Control Responders(CR)
    Customers that make a subscription without an offer (value = 1)
    Treatment Non-Responders(TN)
    Customers that don't make a subscription ang receive an offer (value = 2)
    Treatment Responders(TR)
    Customers that make a subscription and receive an offer (value = 3)
    
    LGWUM method
    Uplift Score = P(TR)/P(T) + P(CN)/P(C) - P(TN)/P(T) - P(CR)/P(C)
    P stands for probability
    T stands for total campaigned amount (TR+TN)
    C stands for total non campaigned amount (CN+CR)
"""
#CN
df_final['target_class'] = 0
#CR
df_final.loc[(df_final.test_control_flag == 0) & (df_final.y == 1), 'target_class'] = 1
#TN
df_final.loc[(df_final.test_control_flag == 1) & (df_final.y == 0), 'target_class'] = 2
#TR
df_final.loc[(df_final.test_control_flag == 1) & (df_final.y == 1), 'target_class'] = 3

# Final check
df_final.head()
df_final.columns
df_final.info()

# Machine learning libraries
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Preparation of training and test set
X = df_final.drop(['y', 'test_control_flag', 'target_class'], axis=1)
y = df_final['target_class']

print(X.shape)
print(y.shape)

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Number of entries in {} : {}'.format('X_train', X_train.shape[0]))
print('Number of entries in {} : {}'.format('X_test', X_test.shape[0]))
print('Number of entries in {} with target class = 0 : {}'.format('y_train', sum(y_train == 0)))
print('Number of entries in {} with target class = 1 : {}'.format('y_train', sum(y_train == 1)))
print('Number of entries in {} with target class = 2 : {}'.format('y_train', sum(y_train == 2)))
print('Number of entries in {} with target class = 3 : {}'.format('y_train', sum(y_train == 3)))

# RandomOverSampler to ensure same amount of each class in training dataset
random = RandomOverSampler(random_state=42)
X_train_res, y_train_res = random.fit_sample(X_train, y_train.ravel())
columns = X_train.columns
X_train_res = pd.DataFrame(data=X_train_res, columns=columns)
print('Number of entries in {} with target class = 0 : {}'.format('y_train_res', sum(y_train_res == 0)))
print('Number of entries in {} with target class = 1 : {}'.format('y_train_res', sum(y_train_res == 1)))
print('Number of entries in {} with target class = 2 : {}'.format('y_train_res', sum(y_train_res == 2)))
print('Number of entries in {} with target class = 3 : {}'.format('y_train_res', sum(y_train_res == 3)))

# Numeric column transformation with StandardScaler
numeric_cols = ['age','campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'duration']
c_trans = ColumnTransformer([('Numeric columns', StandardScaler(), numeric_cols)], remainder='passthrough')
X_scaled = c_trans.fit_transform(X_train_res)
X_test = c_trans.transform(X_test) #Final transformation for test data

