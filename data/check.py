import pandas as pd 
import matplotlib.pyplot as plt 
import os 


DATA_DIRS = '/mnt/infonas/data/pratham/Forecasting/DILATE'

df = pd.read_csv(
    os.path.join(DATA_DIRS, 'data', 'electricity_load_forecasting_panama', 'continuous_dataset.csv')
)



df.head()