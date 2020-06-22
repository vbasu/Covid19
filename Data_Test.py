import pandas as pd

df = pd.read_csv('Data/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df.to_html('temp.html')