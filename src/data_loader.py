import pandas as pd

def load_data(path):
    """
    Loads raw housing data from CSV.
    """
    return pd.read_csv(path)

