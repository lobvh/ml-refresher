import joblib
import pandas as pd
import httpx

def preprocess_customer(customer_dict: dict) -> dict:
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

def send_request(cust_dict: dict) -> dict:
    response = httpx.post(url = "http://localhost:8000/", json = cust_dict).json()
    return response
