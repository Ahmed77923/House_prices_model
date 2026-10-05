import pandas as pd 
from sklearn.model_selection import train_test_split

from config.config import Config


def split_data(X, y , test_size=Config.Trainingconfig.TEST_SIZE,random_state=Config.Trainingconfig.RANDOM_STATE):
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    - X: Features DataFrame
    - y: Target Series
    - test_size: Proportion of the dataset to include in the test split
    - random_state: Seed used by the random number generator
    
    Returns:
    - X_train, X_test, y_train, y_test: Split datasets
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train , X_val , y_train , y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test , X_val , y_val