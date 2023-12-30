from src.data_preprocessing_refactored import *
import streamlit as st

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('./imgs/bank.png')

st.markdown("<h1 style='text-align: center;'>CREDIT SCORING API</h1>", unsafe_allow_html=True)
st.markdown("**This Streamlit app utilizes a predictive model to calculate and display the probability of default for individuals or entities.\
            Users input relevant financial data through an interactive interface, allowing the app to process and analyze key parameters.\
            The tool delivers real-time results, enabling quick and informed decisions on potential default risks.** ")


#----------------------
#Get inputs from users 
#----------------------

data = {}

data["seniority"] = st.number_input(
        "Job Seniority (Years)",
        min_value = 0,
        step = 1,
        value = 2)

data["income"] = st.number_input(
        "Amount of Income ($)",
        min_value = 0,
        step = 1,
        value = 200)

data["assets"] = st.number_input(
        "Amount of Assets ($)",
        min_value = 0.,
        step = 1.,
        value = 200.,
        format="%.2f")


data["time"] = st.number_input(
        "Time of Requested Loan (Months)",
        min_value = 1,
        step = 1,
        value = 1)

data["amount"] = st.number_input(
        "Amount of Loan Requested By Customer ($)",
        min_value = 1000.,
        step = 100.,
        value = 1000.,
        format = "%.2f")

data["monthly_payment"] = round(data["amount"]/data["time"], 2)

job_categories = ['fixed', 'partime','freelance', 'others', 'unk']
data["job"]  = st.selectbox('Type of job customer is doing', job_categories)

home_categories = ['rent', 'owner', 'private', 'ignore', 'parents', 'other', 'unk']
data["home"]  = st.selectbox('Type of home ownership', home_categories)

records_categories = ['no', 'yes']
data["records"]  = st.selectbox('Existence of records', records_categories)

data["status"] = 1 

data["id"] = st.number_input(
        "Enter the customer's ID",
        min_value = 1,
        step = 1,
        value = 1)

#st.write(data)

_ = '''
    This is part of a pipeline that needs to be done in order for Streamlit app to send a proper dictionary for FastAPI endpoint. 
    It should be run in this order. And yes, it had to be stored in a God damn variable because Streamlit is not ignoring the comments.
'''

if st.button("Send request"):
    PATH = '../model/column_transformer.pkl'

    #Obtain customer's id and strip it from the 'data'
    cust_id, cust_df = stripping_id(data)
    
    #Load the ColumnTransformer
    ct = load_column_transformer(PATH)

    #Scale the dataa using pre-loaded ColumnTransformer
    df_scaled = scale_customer_data(column_transformer=ct, customer_df = cust_df)

    #Get the final dictrionary ready to be sent to a FastAPI endpoint
    dict_final = preprocess_scaled_customer_dataframe(scaled_df = df_scaled, customer_id = cust_id)

    #Send the request using the dictionary and write the result in the Streamlit app
    st.write(send_request(customer_dict = dict_final, url = "http://fastapi:8000/")) 


_ = '''
    When you do docker-compose up it creates two containers one for FastAPI and another one for Streamlit app, and even if they are part of the same network,
    they have separate 'localhosts' and to distingish them on that network we use their names provided in the docker-compose.yml.
        services:
            fastapi: <--- This is what you use instead of 'localhost' when you want to access FastAPI's localhost
             ...

            streamlit: <-- This is the name you would use if you have need for reaching the Streamlit's loclahost.

    That is why I've used url = "http://fastapi:8000/".
    Read more here: https://docs.docker.com/compose/networking/
'''

