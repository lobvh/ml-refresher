"""

The whole idea is that each time you run this script one of the customers at random will be selected from the 'test set' since it serves
as a proxy for 'real world' incoming traffic.

In the context of credit scoring and the likelihood of a customer defaulting, a logical action might be to take preventive measures
or provide additional support. This way, if the customer is flagged as potentially defaulting, you are signaling the initiation of proactive steps
to assist the customer and mitigate the risk of default.

"""
import httpx
import random
import asyncio
import pandas as pd
import numpy as np

# Type-hints
JSON_IN = dict[str : float | int]
JSON_OUT = dict[str : float | bool]

# Reading the test set
df_test = pd.read_csv("./data/test-data.csv")

# Deleting the actual values since we don't want to rely on that
del df_test["status"]


def generate_random_id(_: None) -> str:
    """
    A helper function that serves for generating fake id's for customers, because in the ideal world it might be happy to have
    that kind of information on your PC when you send a request to an API service.

    Example:
    >>> generate_random_id()
        '321'
    """
    return "".join([str(i) for i in np.random.choice(range(10), size=4)])


# Applying the function to the whole dataset
df_test["id"] = df_test.apply(generate_random_id, axis=1)

# Generating random row given the index
random_index = random.randint(0, len(df_test) - 1)
random_row = df_test.iloc[random_index].to_dict()


async def is_defaulting(
    customer: JSON_IN, client: httpx.AsyncClient, url: str = "http://localhost:8000/"
) -> JSON_OUT:
    """
    This function serves in a way of sending the request to our FlaskAPI and receiving a response in form of predicting if the customer will default
    or not. In our API we've defined that if p>=0.5 the customer will default, and vice versa.

    Example:
    >>> is_defaulting(customer = {'seniority': 0.7357104035909454,
                                  'income': -0.3407031698861751,
                                  'assets': -0.4639393146108877,
                                  'time': 0.1065451583089126,
                                  'amount': -0.5035235815768409,
                                  'monthly_payment': -0.443498375820261,
                                  'job': 'fixed',
                                  'home': 'parents',
                                  'records': 'yes'},
                    client = httpx.AsyncClient)

    Returns:
        {'probability_of_defaulting': 0.8666973114013672, 'is_defaulting': True}
        Initiating customer support measures for 'id-2135'...
    """
    response = await client.post(url, json=customer)
    return response.json()


# The main function which creates proper connection pool (even if it is an overkill for one request), but might be scalable later for more requests...
async def main() -> None:
    async with httpx.AsyncClient() as client:
        response = await is_defaulting(customer=random_row, client=client)
        print(response)
        print("-------")
        if response["is_defaulting"]:
            print(
                f"Initiating customer support measures for 'id-{random_row['id']}'..."
            )
        else:
            print(f"No preventive measures needed for 'id-{random_row['id']}'!")


if __name__ == "__main__":
    asyncio.run(main())
