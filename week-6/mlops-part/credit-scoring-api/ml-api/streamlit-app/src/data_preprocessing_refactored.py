import joblib
from pathlib import Path
import pandas as pd
import httpx
from typing import Tuple

JSON = dict[str : float | str]
ColumnTransformer = "sklearn.ColumnTransformer"
NumpyArray = "np.ndarray"


def stripping_id(customer: JSON) -> Tuple[str, pd.DataFrame]:
    """
    Strip the 'id' part of an incoming JSON file since the preloaded ColumnTransformer doesn't expects it.
    It would take a whole new ColumnTransformer to train just for the sake of 'id', and at this point of development I'm too lazy to fix
    everything. Will make it into consideration next time I develop an app.
    At the same time, we need to keep the `id` because after the ColumnTransformer makes its magic, the FastAPI expects an 'id' as a part
    of the "base model". Also, ColumnTransformer expects a DataFrame, and hence we convert and return customer dictionary as DataFrame.

    Parameters:
        customer (JSON): Dictionary containing customer data.

    Returns:
        customer_id, customer (Tuple): Tuple containing both the id of a customer and the stripped version of the customer dictionary as a DataFrame.
    """

    customer_id = customer.pop("id")  # Removes 'id' key and returns its value
    customer_df = pd.DataFrame([customer])

    return customer_id, customer_df


def load_column_transformer(model_path: Path) -> ColumnTransformer:
    """
    Load the ColumnTransformer model from the specified path.

    Parameters:
        model_path (str or Path): Path to the saved ColumnTransformer model.

    Returns:
        object: Loaded ColumnTransformer model.
    """
    # Convert the input path to a Path object
    model_path = Path(model_path)

    # Use the resolved path to handle relative paths
    resolved_model_path = model_path.resolve()

    return joblib.load(resolved_model_path)


def scale_customer_data(
    column_transformer: ColumnTransformer, customer_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Given the pre-loaded ColumnTransformer use it to properly scale the columns.

    Parameters:
        column_transformer (ColumnTransformer): Preloaded ColumnTransformer.
        customer_df (pd.DataFrame): Expects a particular type of DataFrame with particular columns suitable just for this case.
        customer_id (str): ID of a customer.

    Returns:
        cust_scaled (pd.DataFrame): DataFrame scaled using the pre-loaded ColumnTransformer.
    """

    ct = column_transformer
    cust_df = customer_df

    # Scaling the df using pre-loaded ColumnTransformer
    cust_scaled: NumpyArray = ct.transform(cust_df)

    # Loading feature names from ColumnTransformer
    feature_names = ct.get_feature_names_out()

    # Converting the cust_scaled to DataFrame
    cust_df = pd.DataFrame(cust_scaled, columns=feature_names)

    return cust_df


def preprocess_scaled_customer_dataframe(
    scaled_df: pd.DataFrame, customer_id: int
) -> JSON:
    """
        Since the scaled DataFrame returned from scale_customer_data function returns a DataFrame which has artifacts in column names
        from ColumnTransformer such as scaler__ and passthrough__ we need to strip it. We also need to drop the unused columns which FastAPI
        doesn't expect such as 'status', and finally, we need to convert the DataFrame to a dictionary.

    Parameters:
        scaled_df (pd.DataFrame): Expects the DataFrame scaled using the pre-loaded ColumnTransformer.
    Returns:
        json (JSON): Dictionary ready for the FastAPI entrypoint.

    """

    # Stripping the misc parts of column names
    scaled_df.columns = scaled_df.columns.str.replace("scaler__", "").str.replace(
        "passthrough__", ""
    )

    # Now we are safe to drop the unused 'status' column
    scaled_df = scaled_df.drop("status", axis=1)

    # Convert the DataFrame to a dictionary
    customer_dict = scaled_df.iloc[0].to_dict()

    # Fix the 'id' column to be of type 'str' since FastAPI endpoint expects it that way
    customer_dict["id"] = str(customer_id)

    return customer_dict


def send_request(customer_dict: JSON, url: str = "http://localhost:8000/") -> JSON:
    """
    I don't want to overcomplicate this part and use concurrent programming, since I'm not really sure if Streamlit handles it
    out-of-the-box and how good. I'm going to use a simple httpx function to send a POST request and receive the proper JSON response.

    Parameters:
        customer_dict (JSON): Expects a dictionary that is properly defined as an input to a FastAPI endpoint
        url (str): URL of a FastAPI endpoint.
    """
    response = httpx.post(url=url, json=customer_dict).json()
    return response
