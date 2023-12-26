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
    monthly_payment: float = Field(
        description="Amount of money customer needs to return per month. (Feature Engineering)"
    )
    job: str = Field(description="Type of job customer is doing.")
    home: str = Field(description="Type of home ownership.")
    records: str = Field(description="Existance of records.")
    id: str = Field(
        description="Fake ID representing each customer as an individual.\
                                 Maybe not that necessary when someone comes and you send a request to an API,\
                                 but definetely might come handy as historical data/logs."
    )


# Extracting a model

MODEL_FILE = "./model/best_model.bin"
with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)


# This serves as testing if API works at all
@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hello World"}


# Test if the post request actually works
@app.post("/")
async def add_item(customer: Customer) -> dict[str, bool | float]:
    cust_dict = dict(customer)
    X_customer = dv.transform([cust_dict])
    y_pred = model.predict_proba(X_customer)[0, 1]

    default = y_pred >= 0.5

    return {"probability_of_defaulting": float(y_pred), "is_defaulting": bool(default)}
