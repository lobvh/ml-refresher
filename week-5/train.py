#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


#Parameters
'''
For the parameter C we've never created a function to return us the best C. We've just used for loop to print different ones and used the one
based on our decision. Here, we will use the best C that we 'eyeballed'. 
'''
C = 1.0 
n_splits = 5
output_file = f'model_C={C}.bin'
RANDOM_STATE = 42

#We've already prepared the data in week-3 
df = pd.read_csv('./data/data-preparation.csv')

'''
We will use the Cross Validation on df_full_train so we don't need to create df_val like in the previous lession.
'''
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


#Assesing 'numerical' and 'categorical' variables
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

#Training the model

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

#Converting a DataFrame into something that model can interpret and do predictions
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


#Cross Validation

print(f'Doing validation with C={C}.')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'AUC on fold {fold} is:  {round(auc, 4)}.')
    fold = fold + 1

#Printing the results
print('Validation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


#Training the final model

print('Training the final model:')

dv, model = train(df_full_train, df_full_train.churn.values, C=C)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)

print(f'AUC={round(auc,3)}')


#Save the model
'''
By pickling the model it will save DictVectorizer and LogisticRegression objects, and you can also use it's functions. 
'''

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is saved to a file name: {output_file}')
