"""Bazham Khanatayev
   This is an assignment practicing with flask
   This file answers to the REST API with predictions """

import pandas as pd
import pickle
from flask import Flask, request
import os

app = Flask("__name__")


def unpickled_model():
    filename = "churn_model.pkl"

    with open(filename, "rb") as file:
        rfc = pickle.load(file)

    return rfc

def flask_api(model):
    """
    API that takes parameters from the URL and returns a webpage showing the prediction for those parameters
    """
    @app.route('/')
    @app.route('/home')
    def home():
        return "<h1>Home Page</h1>"

    @app.route("/predict_churn")
    def predict_churn():
        header = "<h1>Churn prediction:\n</h1>"
        age = request.args.get("age")
        is_male = request.args.get("is_male")
        late_on_payment = request.args.get("late_on_payment")
        years_in_contract = request.args.get("years_in_contract")
        num_inters = request.args.get("num_inters")

        sample = pd.DataFrame({"is_male": [is_male],
                               "num_inters": [num_inters],
                               "late_on_payment": [late_on_payment],
                               "age": [age],
                               "years_in_contract": [years_in_contract]})

        sample_pred = model.predict(sample)
        return header + str(sample_pred[0])




if __name__ == "__main__":
    model = unpickled_model()
    flask_api(model)
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run(debug=True)
    
