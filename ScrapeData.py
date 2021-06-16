import requests
import pandas as pd
from io import StringIO
from func import get_eod_data
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 400)

#TODO token accces from external file
hnt_data = get_eod_data('HNT-USD.CC', api_token='5d73bac6b50d08.86816226').reset_index()

#write the data as CSV FILE TODO the name should be changed
with open('Data/HNT_historical_data_v2.csv', 'wb') as f:
    hnt_data.to_csv(f, sep=';')