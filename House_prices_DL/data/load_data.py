import pandas as pd  
from sklearn.Datasets import fetch_openml


def load_data():
    # Load the Ames Housing dataset from OpenML
    
    X, y = fetch_openml(data_id=42165, as_frame=True, return_X_y=True)


    return X, y