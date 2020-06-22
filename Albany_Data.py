# Get Albany data

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('Data/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
ny_data = df.loc[df['Province_State'] == 'New York']
albany_data = ny_data.loc[df['Admin2'] == 'Albany']

Extra_columns = ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State', 'Country_Region', 'Lat', 'Long_', 'Combined_Key']
albany_data = albany_data.drop(Extra_columns, axis=1)

#albany_data.T.plot()
#plt.show()
a = np.array(albany_data)[0]
b = a[a > 0]
print(b)

