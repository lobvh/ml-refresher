#!/usr/bin/env python
# coding: utf-8

import pickle

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 24,
    "monthlycharges": 29.85,
    "totalcharges": (24 * 29.85)
}

X = dv.transform([customer]) #DictVectorizer expects list of dictionaries and hence list with one dictionary
y_pred = model.predict_proba(X)[0,1] #print only model.precit(X) to unerstand why [0,1] is needed

print(f'Customer: {customer}')
print('------')
print(f'Predicted probability of churning: {round(y_pred,4)}')
