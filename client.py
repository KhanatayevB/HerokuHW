"""Bazham Khanatayev
   This is an assignment practicing with flask
   This file essentially implements a client that gets predictions from our inference server"""


import pandas as pd
import numpy as np
import requests

INDEXES = [1, 5, 8, 11, 40]

COLUMNS = ["is_male", "num_inters", "late_on_payment", "age", "years_in_contract"]


def pred_check():
    """
    Create an array of the previous predictions for testing
    """
    preds = np.loadtxt("preds.csv")
    return preds


def rest_api():
    """
    Iterate through the list of indexes (I chose random numbers just for this exercise)
    Input the values from a dataframe into the API and return the predictions
    """

    rest_preds = []

    X_test = pd.read_csv("X_test.csv", usecols=COLUMNS)

    for i in INDEXES:
        sample = pd.DataFrame(X_test.iloc[i, :])

        parameters = {col: [sample.loc[col]] for col in COLUMNS}

        rest_request = requests.get('http://127.0.0.1:5000/predict_churn', params=parameters)
        rest_pred = int(rest_request.text.rsplit("</h1>")[1])

        rest_preds.append(rest_pred)

    return rest_preds


def rest_test():
    """
    Check that the predictions from the API match the predictions in the CSV
    """
    preds = [int(pred_check()[i]) for i in INDEXES]
    rest_preds = rest_api()
    mask = (preds != rest_preds)

    assert np.sum(mask) == 0, "Predictions requested from API did not match all predictions in the CSV"

    print("All predictions requested from API matched the prediction in the CSV")


if __name__ == "__main__":
    rest_test()
