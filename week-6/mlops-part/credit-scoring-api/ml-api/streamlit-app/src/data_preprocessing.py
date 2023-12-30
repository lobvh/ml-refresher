import joblib
import pandas as pd
import httpx
from typing import Tuple

JSON = dict[str: float|str]

'''
Ideas to improve:

    [] Split code into meaningful functions
    [] Add docstrings to those function. Here is the example of docstirng pattern you should follow:

    """
    Scale customer data using the provided ColumnTransformer.

    Parameters:
        customer_dict (JSON): Dictionary containing customer data.
        column_transformer (object): Loaded ColumnTransformer model.

    Returns:
        pd.DataFrame: Scaled customer data as a DataFrame.
    """
'''
#Example of data as an input:

"""
{
"seniority": 2,
"income": 200,
"assets": 200,
"time": 1,
"amount": 1000,
"monthly_payment": 1000,
"job": "fixed",
"home": "rent",
"records": "no",
"status": 1,
"id": 1
}
"""
def stripping_id(customer: JSON) -> Tuple[str, JSON] :

     customer_id = customer.pop('id')  # Removes 'id' key and returns its value

    return customer_id, customer

cust_id, cutomer_dict = stripping_id(customer)

def preprocess_customer(customer_dict: JSON) -> JSON:
    #Stripping the id for now since ColumnTransformer doesn't expects it
    cust_id = customer_dict['id']
    del customer_dict['id']

    #Loading ColumnTransformer
    loaded_column_transformer = joblib.load('../model/column_transformer.pkl')

    #Converting dict to dataframe
    df = pd.DataFrame([customer_dict])

    #Scaling the df using loaded ColumnTransformer (Returns np.array!)
    cust_scaled = loaded_column_transformer.transform(df)

    #Loading feature names from ColumnTransformer
    feature_names = loaded_column_transformer.get_feature_names_out()

    #Before converting it to meaningful dictionary cust_scaled array needs to be converted to df
    df_before_dict = pd.DataFrame(cust_scaled, columns = feature_names)

    #Stripping the misc parts of column names
    df_before_dict.columns = df_before_dict.columns.str.replace('scaler__', '').str.replace('passthrough__', '')

    #Now we are safe to drop the unused 'status' column
    df_before_dict = df_before_dict.drop('status', axis=1)

    #Final dictionary ready for FastAPI
    customer_dict = df_before_dict.iloc[0].to_dict()

    #Giving back the id since FastAPI expects it 
    customer_dict['id'] = str(cust_id)

    return customer_dict

def send_request(cust_dict: JSON) -> JSON:
    response = httpx.post(url = "http://localhost:8000/", json = cust_dict).json()
    return response
