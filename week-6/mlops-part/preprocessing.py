import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


RANDOM_STATE = 42

"""
------------------
DATA PREPROCESSING       
------------------
"""

#Importing the raw CSV file
df = pd.read_csv('./../data/CreditScoring.csv')

#Filtering only values which contain 1 or 2 since only they represent
#if person is defaulting (2) or not (1). There is one instance of 0, which could be dropped.
df = df[df['Status'] != 0].reset_index(drop = True)

#Since in documentation it states that these values of 99999999 are NaNs we will convert them as so
#such that they don't mess up with statistics:
for c in ['Income', 'Assets', 'Debt']:
    df[c] = df[c].replace(to_replace=99999999, value=np.nan)

#Making all the column names standardized
df.columns = df.columns.str.lower()

'''
-----------------------------------------------------------
CONVERTING ALL THE CATEGORICAL VALUES TO A MEANINGFUL NAMES
-----------------------------------------------------------
'''

status_values = {
    1: 'ok',
    2: 'default',
}

df.status = df.status.map(status_values)

home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}

df.home = df.home.map(home_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}

df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes'
}

df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}

df.job = df.job.map(job_values)

#Look at the .ipynb why we decided to fill NaNs with the values of 0
for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=np.nan, value=0)

df = df.reset_index(drop=True)

#When you want to make a vector of true labels and compare them with the ones model predicted you need to convert them to integera
status_values = {
    'ok': 0,
    'default': 1
}

df.status = df.status.map(status_values)


'''
----------------------------------------------
PREPARING THE DATAFRAME AND SPLITTING THE DATA
----------------------------------------------
'''

#These are the values we've decided to be used in modeling
categorical = ['records', 'job', 'home']
numerical = ['seniority', 'income', 'assets', 'time', 'amount']


#So, the final feature matrix should have these types of features and also the y_true
df = df[categorical + numerical + ['status']].reset_index(drop = True)


#We've engineered one feature we might think might have an impact, and indeed it does. Look for .ipynb version to check it out! 
df['monthly_payment'] = (df['amount']/df['time']).round(2)

#Finally, after introducing that feature we need to reset the index on the whole df
df = df[categorical+numerical+['monthly_payment', 'status']].reset_index(drop = True)


#Now, we need to use z-score to scale the features such that large valued don't have much of an impact on final weights
scaler = StandardScaler()

columns_to_scale = ['seniority', 'income', 'assets', 'time', 'amount', 'monthly_payment']
columns_to_leave_unchanged = ['job', 'home', 'records', 'status']


#Create a ColumnTransformer
column_transformer = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), columns_to_scale),
        ('passthrough', 'passthrough', columns_to_leave_unchanged)
    ],
    remainder='drop'  # Drop columns not specified
)

# Apply the transformation to your DataFrame
df_scaled = column_transformer.fit_transform(df)

# Convert the result back to a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=columns_to_scale + columns_to_leave_unchanged).reset_index(drop = True)


"""
-------------------
OPTIONAL: RESCALING THE DATA
------------------

This part is used for Streamlit to convert standard values used by human to something in backend that is used by model. 

Note: It is very, very necessary to export the scaler which was applied to an ENTIRE dataset aka z-score for each column is calculated on 
all rows in your dataset. 

"""

#import joblib
#joblib.dump(column_transformer, './models/column_transformer.pkl')

#Splitting the data
df_full_train, df_test = train_test_split(df_scaled, test_size=0.2, random_state=RANDOM_STATE)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=RANDOM_STATE)

#After shuffling we need some order. :P
df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


#Extracting true labels from feature matrix
y_full_train = df_full_train.status.values.astype(int)
y_train = df_train.status.values.astype(int)
y_val = df_val.status.values.astype(int)
y_test = df_test.status.values.astype(int)

#Deleting them from the feature matrix
del df_full_train['status']
del df_train['status']
del df_val['status']

'''
--------------------------------------------------------------------------------------------
OPTIONAL: Extracting the df_test and y_test for the purposes of testing our FastAPI endpoint.
Refer to part where we are creating an API to make this clear.
--------------------------------------------------------------------------------------------
'''

#CSV_PATH = './test-data.csv'
#df_test.to_csv(CSV_PATH, index=False)

#To make the rest of the script work for the purposes of preprocessing we have to delete the df_test['status'] also!
del df_test['status']

#Creating a DictVectorizer to make OneHotEncoding
dv = DictVectorizer(sparse=False)

#Using that DictVetorizer in doing so
train_dicts = df_train.to_dict(orient='records')
full_train_dicts = df_full_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

#Fitting the train data to 'learn' which features need to be transformed as categorical 
X_train = dv.fit_transform(train_dicts)
X_valid = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)
X_full_train = dv.transform(full_train_dicts)


