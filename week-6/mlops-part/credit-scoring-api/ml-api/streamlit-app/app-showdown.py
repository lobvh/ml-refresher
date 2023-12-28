from data_preprocessing import preprocess_customer, send_request
import streamlit as st

st.title("Credit scoring API")


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
        "Amount of Income",
        min_value = 0,
        step = 1,
        value = 200)

data["assets"] = st.number_input(
        "Amount of Assets",
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
        "Amount of loan requested by customer ($)",
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
        "Enter the customers ID",
        min_value = 1,
        step = 1,
        value = 1)


if st.button("Send request"):
    cust_dict = preprocess_customer(data)
    st.write(send_request(cust_dict))
