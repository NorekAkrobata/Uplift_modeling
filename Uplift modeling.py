# Uplift modeling - marketing dataset

# Libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Loading data

PATH = r'C:\Users\Norbert\Desktop\Empik'
FILENAME = r'bank_data_prediction_task.csv'

def load_data(path=PATH, filename=FILENAME):
    """Return dataframe from .csv file

    Arguments:
        path -- path to the folder
        filename -- name of the .csv file
    """
    csv_path = os.path.join(path,filename)
    return pd.read_csv(csv_path, index_col = 0)

df = load_data()

# Data preprocessing

cat_cols = ['contact', 'month', 'day_of_week']
num_cols = ['duration', 'campaign']

for i in cat_cols:
    df[i] = df[i].fillna(df['test_control_flag'].map({'control group': 'None'}))

for j in num_cols:
    df[j] = df[j].fillna(df['test_control_flag'].map({'control group': 0}))