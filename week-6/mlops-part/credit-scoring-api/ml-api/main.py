import pickle
from pydantic import BaseModel, Field
from fastapi import FastAPI

app = FastAPI(
        title="Credit Risk Scoring",
        description="A simple API to predict if the customer will default or not.",
        version="0.1.0",
        )


class Customer(BaseModel):

    """

    Representation of a customer as a datapoint incoming to our API for predicting probability of defaulting.
    Some of them are floats and that is because we had to do feature scaling, and we also trained a model with scaled features so even
    if it doesn't makes sense here we have to provide what model expects. ;) 

    """

    seniority: float = Field(description="Job seniority (years).")
    income: float = Field(description="Amount of income.")
    assets: float = Field(description="Amount of assets.")
    time: float = Field(description="Time of requested loan.")
    amount: float = Field(description="Amount of loan requested by customer.")
    monthly_payment: float = Field(description="Amount of money customer needs to return per month. (Feature Engineering)")
    job: str = Field(description="Type of job customer is doing.")
    home: str = Field(description="Type of home ownership.")
    records: str = Field(description="Existance of records.")
    id: str = Field(description="Fake ID representing each customer as an individual.\
                                 Maybe not that necessary when someone comes and you send a request to an API,\
                                 but definetely might come handy as historical data/logs.")

#TODO 1
#Extracting a model goes here

#This serves as testing if API works at all
@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hello World"}

#Test if the post request actually works
@app.post("/post")
def add_item(customer: Customer) -> dict[str, str]:
    return {"Yo": "It Works!",
            "cust": str(customer.seniority)} #httpx don't understands nothing. be explicit. 

#TODO 2
#Implement something like this:
"""
request here is part of Flask "from Flask import request"

    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    #Marketing team for sending promotional emails doesn't understand
    #for which kind of probability it will send a promotional email
    #so we will be explicit here and send True if p>=0.5
    churn = y_pred >= 0.5

    #We have to use float() and bool() since JSON doesn't understand
    #numpy's version of float and bool, only Pythons.
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result) #Turn the response to JSON type!


"""
