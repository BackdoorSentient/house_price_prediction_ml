# This file is responsible for cleaning and transforming raw data into a format that an ML model can understand.

import pandas as pd

def preprocess(df):
    """
    Cleans data and prepares features for ML.
    """
    # Handle missing values
    df=df.copy()
    # df['bedrooms'].fillna(df['bedrooms'].median(),inplace=True)

    # #seprate target
    # X=df.drop("price",axis=1)
    # y=df["price"]

    # #encode categorical features
    # X=pd.get_dummies(X,columns=['location'],drop_first=True)

    # return X,y

    df.drop(
        columns=["date","street","country"],
        inplace=True,
        errors="ignore"
    )

    df.fillna(0,inplace=True)
    X=df.drop("price",axis=1)
    y=df["price"]

    categorical_cols=["city","statezip"]
    X=pd.get_dummies(X,columns=categorical_cols,drop_first=True)

    return X,y