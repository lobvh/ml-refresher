import pandas as pd
import numpy as np


df = pd.read_csv('./../data/CreditScoring.csv')

df = df[df['Status'] != 0].reset_index(drop = True)

for c in ['Income', 'Assets', 'Debt']:
    df[c] = df[c].replace(to_replace=99999999, value=np.nan)


df.columns = df.columns.str.lower()

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

for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=np.nan, value=0)

df = df.reset_index(drop=True)


status_values = {
    'ok': 0,
    'default': 1
}

df.status = df.status.map(status_values)


categorical = ['records', 'job', 'home']
numerical = ['seniority', 'income', 'assets', 'time', 'amount']

df = df[categorical + numerical + ['status']].reset_index(drop = True)

df['monthly_payment'] = (df['amount']/df['time']).round(2)

df = df[categorical+numerical+['monthly_payment', 'status']].reset_index(drop = True)


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

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


from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

df_full_train, df_test = train_test_split(df_scaled, test_size=0.2, random_state=RANDOM_STATE)

df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state=RANDOM_STATE)

#After shuffling we need some order. :P
df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_full_train = df_full_train.status.values.astype(int)
y_train = df_train.status.values.astype(int)

y_val = df_val.status.values.astype(int)
y_test = df_test.status.values.astype(int)


del df_full_train['status']
del df_train['status']
del df_val['status']
del df_test['status']


from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


dv = DictVectorizer(sparse=False)


train_dicts = df_train.to_dict(orient='records')
full_train_dicts = df_full_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

X_train = dv.fit_transform(train_dicts)
X_valid = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)
X_full_train = dv.transform(full_train_dicts)

