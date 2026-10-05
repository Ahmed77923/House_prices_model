import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline



# Right-skewed distribution
#    │
#    │ ███████
#    │ █████████
#    │ ███████████
#    │ █████
#    │ ██
#    │ █
#    └──────────────────────► Price
#        low          high

# def preprocess_data(X_train, X_test,X_val):
#    """l_transformer, numerical_cols),
   