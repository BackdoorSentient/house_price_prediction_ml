from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_model(X,y):
    """
    Splits data, trains Linear Regression model, saves it.
    """
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    model=LinearRegression()
    model.fit(X_train,y_train)

    os.makedirs("models",exist_ok=True)

    joblib.dump(model,"models/linear_regression.pkl")

    return model,X_test,y_test