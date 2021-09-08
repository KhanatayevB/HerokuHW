"""Bazham Khanatayev
   This is an assignment practicing with flask
   This file trains our model and saves it as a pkl"""

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from sklearn.metrics import accuracy_score


def main():
    df = pd.read_csv("cellular_churn_greece.csv")

    target = "churned"
    features = list(df.columns)
    features.remove(target)

    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    pd.DataFrame(X_test).to_csv("X_test.csv", index=False)

    rfc = RandomForestClassifier(random_state=42)

    rfc.fit(X_train, y_train)

    filename = "churn_model.pkl"

    with open(filename, "wb") gas file:
        pickle.dump(rfc, file)

    y_preds = rfc.predict(X_test)
    np.savetxt('preds.csv', y_preds, delimiter=',')

    accuracy = accuracy_score(y_preds, y_test)
    print(f"Accuracy = {accuracy}")


if __name__ == "__main__":
    main()
