import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

### Read the data in with pandas
df_weather=pd.read_csv("YVR_2024.csv")
df_pm25=pd.read_csv("pm2_5.csv")
df_param1=pd.read_csv('parameter_january.csv')
df_param2=pd.read_csv('parameter_february.csv')
df_param3=pd.read_csv('parameter_march.csv')
df_param4=pd.read_csv('parameter_april.csv')
df_param5=pd.read_csv('parameter_may.csv')
df_param6=pd.read_csv('parameter_june.csv')
df_param7=pd.read_csv('parameter_july.csv')
df_param8=pd.read_csv('parameter_august.csv')
df_param9=pd.read_csv('parameter_september.csv')
df_param10=pd.read_csv('parameter_october.csv')
df_param11=pd.read_csv('parameter_november.csv')
df_param12=pd.read_csv('parameter_december.csv')
df_fire=pd.read_csv('Fire_bc.csv')

# split the column of parameter in multiple columns

columns_p=['co','no2','pm10','pm25']
# Pivot to reshape the data for january
df_pivot1 = df_param1.pivot(index='datetimeUtc', columns='parameter', values='value')
df_pivot1.to_csv('df_january.csv')
# Pivot to reshape the data for february
df_pivot2 = df_param2.pivot(index='datetimeUtc', columns='parameter', values='value')
df_pivot2.to_csv('df_february.csv')

# Pivot to reshape the data for march
df_pivot3 = df_param3.pivot(index='datetimeUtc', columns='parameter', values='value')
df_pivot3.to_csv('df_march.csv')

# Pivot to reshape the data for april
df_pivot4 = df_param4.pivot(index='datetimeUtc', columns='parameter', values='value')
df_pivot4.to_csv('df_april.csv')

# Pivot to reshape the data for may
df_pivot5 = df_param5.pivot(index='datetimeUtc', columns='parameter', values='value')
df_pivot5.to_csv('df_may.csv')

# Pivot to reshape the data for june
df_pivot6 = df_param6.pivot(index='datetimeUtc', columns='parameter', values='value')
df_pivot6.to_csv('df_june.csv')

# Pivot to reshape the data for july
df_pivot7 = df_param7.pivot(index='datetimeUtc', columns='parameter', values='value')
df_pivot7.to_csv('df_july.csv')

# Pivot to reshape the data for august
df_pivot8 = df_param8.pivot(index='datetimeUtc', columns='parameter', values='value')
df_pivot8.to_csv('df_august.csv')

# Pivot to reshape the data for september
df_pivot9 = df_param9.pivot(index='datetimeUtc', columns='parameter', values='value')
df_pivot9.to_csv('df_september.csv')

# Pivot to reshape the data for october
df_pivot10 = df_param10.pivot(index='datetimeUtc', columns='parameter', values='value')
df_pivot10.to_csv('df_october.csv')

# Pivot to reshape the data for november
df_pivot11 = df_param11.pivot(index='datetimeUtc', columns='parameter', values='value')
df_pivot11.to_csv('df_november.csv')

# Pivot to reshape the data for december
df_pivot12 = df_param12.pivot(index='datetimeUtc', columns='parameter', values='value')
df_pivot12.to_csv('df_december.csv')

df_params_merged=pd.concat([df_pivot1,df_pivot2,df_pivot3,df_pivot4,df_pivot5,df_pivot6,df_pivot7,df_pivot8,df_pivot9,df_pivot10,df_pivot11,df_pivot12], axis=0)

df_params_merged.reset_index(inplace=True)  # Move datetimeUtc back to a column
df_params_merged['datetimeUtc'] = pd.to_datetime(df_params_merged['datetimeUtc'], errors='coerce')

print(df_params_merged.head())

df_params_merged['index'] = range(1, len(df_params_merged) + 1)
df_params_merged=df_params_merged[['index', 'datetimeUtc', 'co', 'no2',  'pm25']]
print(df_params_merged.dtypes)
df_params_merged.to_csv("combined_data.csv", index=False)


############### fix the format of date column for both datasets
df_weather['DATE']=pd.to_datetime(df_weather['DATE'])
df_params_merged['datetimeUtc'] = pd.to_datetime(df_params_merged['datetimeUtc'], errors='coerce')


print(df_params_merged.dtypes)

df_weather['DATE'] = df_weather['DATE'].dt.tz_localize(None)
df_params_merged['datetimeUtc']=df_params_merged['datetimeUtc'].dt.tz_localize(None)

df_params_merged.rename(columns={'datetimeUtc': 'DATE'}, inplace=True)
df_params_merged.to_csv('PARAMETERS.csv')
#################### merge the two dataset. Weather and air parameters

df_merged = pd.merge(df_weather,df_params_merged, on="DATE", how='outer')

# Strip all values (if needed)
df_merged = df_merged.apply(lambda x: x.astype(str).str.strip())

# Replace empty strings and invalid entries with NaN
df_merged.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df_merged.replace(['', ' ', 'nan', 'NaN'], np.nan, inplace=True)

df_merged.replace('', np.nan, inplace=True)
df_merged.to_csv("DATASET.csv", index=False)





df_interest=['TMP','VIS']



for col in df_interest:
   df_merged[col]=pd.to_numeric(df_merged[col].astype(str).str.replace(",", "."), errors= 'coerce')

df_merged['TMP']=  ((df_merged['TMP']-32) * (5/9))

print(df_merged.dtypes)

df_final_dataset = df_merged[['DATE','TMP','WND','CIG','DEW','SLP', 'co','no2','pm25']]
df_final_dataset.to_csv('Final_Dataset.csv', index=False,na_rep='NA')

df_merged.to_csv('DATASET.csv', index=False,na_rep='NA')
print(df_final_dataset.isna().sum())

df_final_dataset.loc[:, 'co'] = pd.to_numeric(df_final_dataset['co'], errors='coerce')
df_final_dataset.loc[:, 'no2'] = pd.to_numeric(df_final_dataset['no2'], errors='coerce')
df_final_dataset.loc[:, 'pm25'] = pd.to_numeric(df_final_dataset['pm25'], errors='coerce')

# Ensure numeric columns are properly inferred
df_final_dataset = df_final_dataset.infer_objects(copy=False)

# Now safely interpolate
df_final_dataset.loc[:, 'co'] = df_final_dataset['co'].interpolate(method='linear', limit_direction='both')
df_final_dataset.loc[:, 'no2'] = df_final_dataset['no2'].interpolate(method='linear', limit_direction='both')
df_final_dataset.loc[:, 'pm25'] = df_final_dataset['pm25'].interpolate(method='linear', limit_direction='both')

df_final_dataset.to_csv('Final_Dataset.csv')



# extact time-related features:

df_final_dataset['DATE']=pd.to_datetime(df_final_dataset['DATE'])

df_final_dataset['year']=df_final_dataset['DATE'].dt.year
df_final_dataset['month']=df_final_dataset['DATE'].dt.month
df_final_dataset['day'] = df_final_dataset['DATE'].dt.day
df_final_dataset['hour'] = df_final_dataset['DATE'].dt.hour
df_final_dataset=df_final_dataset[['DATE','year','month',	'day',	'hour'	,'TMP',	'WND',	'CIG',	'DEW'	,'SLP'	,'co',	'no2',	'pm25'	]]

df_final_dataset.to_csv('Final_Dataset.csv')

#Final clean for Dataset

# Split WND (wind direction, wind type, speed, etc.)
df_final_dataset['WND']=df_final_dataset['WND'].str.split(',').str[3] #wind speed
df_final_dataset['WND']=pd.to_numeric(df_final_dataset['WND'], errors='coerce')

# Split CIG (ceiling height)
df_final_dataset['CIG'] = df_final_dataset['CIG'].str.split(',').str[0]
df_final_dataset['CIG'] = pd.to_numeric(df_final_dataset['CIG'], errors='coerce')

# Split DEW (dew point temp)
df_final_dataset['DEW'] = df_final_dataset['DEW'].str.split(',').str[0]
df_final_dataset['DEW'] = pd.to_numeric(df_final_dataset['DEW'], errors='coerce')
df_final_dataset['DEW'] = df_final_dataset['DEW']/10 # NOAA DEW is in tenths of °C

df_final_dataset['SLP']= df_final_dataset['SLP'].str.split(',').str[0]
df_final_dataset['SLP'] = pd.to_numeric(df_final_dataset['SLP'], errors='coerce')
df_final_dataset['SLP'] = df_final_dataset['SLP']/10 # NOAA SLP is in tenths of hPa

#################################### adding fire info to dataset####################
df_fire['acq_date']=pd.to_datetime(df_fire['acq_date'])

# Define a bounding box around Vancouver (adjust as needed)
lat_min, lat_max = 48.5, 50.5
lon_min, lon_max = -125, -120

df_fire_bc = df_fire[
    (df_fire['latitude'] >= lat_min) &
    (df_fire['latitude'] <= lat_max) &
    (df_fire['longitude'] >= lon_min) &
    (df_fire['longitude'] <= lon_max)
]

fire_dates = df_fire_bc['acq_date'].unique()
# Create a new column 'fire_event' = 1 if DATE (any hour) is on a fire day

fire_dates = pd.to_datetime(fire_dates).normalize()
df_final_dataset['fire_event'] = df_final_dataset['DATE'].dt.normalize().isin(fire_dates).astype(int)



df_final_dataset.to_csv('Final_Dataset.csv')