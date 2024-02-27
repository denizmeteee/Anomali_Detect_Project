#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from fbprophet import Prophet
import holidays

# In[2]:


# !pip install fbprophet


# In[3]:

# Reading daily and hourly bike sharing data from CSV files
df_daily = pd.read_csv("bike_sharing_daily.csv")
df_hourly = pd.read_csv("bike_sharing_hourly.csv")

# Reading the KPIs of the last 30 days from an Excel file
df_deneme = pd.read_excel("Last_30_day_KPIs.xlsx", sheet_name="Sayfa1", engine='openpyxl')


# In[4]:

# Function to fix the datatypes of the columns in the dataframes
def fixing_datatypes(df):
    # Fixing the datatypes
    df['dteday'] = df['dteday'].astype('datetime64')
    df.loc[:, 'season':'mnth'] = df.loc[:, 'season':'mnth'].astype('category')
    df[['holiday', 'workingday']] = df[['holiday', 'workingday']].astype('bool')
    df[['weekday', 'weathersit']] = df[['weekday', 'weathersit']].astype('category')
    return df


# In[5]:

# Applying the function to fix the datatypes of the columns in the daily and hourly dataframes
df_daily = fixing_datatypes(df_daily)
df_hourly = fixing_datatypes(df_hourly)
df_hourly['hr'] = df_hourly['hr'].astype('category')

# In[6]:


datas = []
for col in df_deneme.columns[3:]:
    datas.append(df_deneme[['Date', col]])
datas.pop(0)
print(datas[4])

# In[7]:


for data in datas:
    data.columns = ["ds", "y"]
    data = pd.DataFrame(data=data)
    print(data.head())

# In[8]:


df_daily = df_daily[["cnt", "dteday"]]
df_daily.columns = ["y", "ds"]
df_daily.head()

# ## Lets Predict

# In[9]:


# display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", 500)

# Function to fit the model and make predictions
def fit_predict_model(dataframe):
    # seasonality, holidays
    # it can integrate football matches to data.
    m = Prophet(yearly_seasonality=True, daily_seasonality=True)

    m.add_country_holidays(country_name="TR")

    m = m.fit(dataframe)

    forecast = m.predict(dataframe)
    forecast["fact"] = dataframe["y"].reset_index(drop=True)
    return forecast


preds = []
for i, data in enumerate(datas):
    preds.append(fit_predict_model(data))
preds.pop(0)

# In[10]:


print(preds[0].head())

# In[12]:


pd.options.plotting.backend = "plotly"
preds[0].plot(x='ds', y=["yhat_lower", "fact", "yhat_upper", "yhat"])


# # Detecting Anomalies:
# * The light blue boundaries in the above graph are yhat_upper and yhat_lower.
# * If y value is greater than yhat_upper and less than yhat lower then it is an anomaly.
# * Also getting the importance of that anomaly based on its distance from yhat_upper and yhat_lower.

# In[17]:

# Function to detect anomalies in the prediction
def detect_anomalies(forecast):
    forecasted = forecast[["ds", "trend", "yhat", "yhat_lower", "yhat_upper", "fact"]].copy()

    forecasted["anomaly"] = 0
    forecasted.loc[forecasted["fact"] > forecasted["yhat_upper"], "anomaly"] = 1
    forecasted.loc[forecasted["fact"] < forecasted["yhat_lower"], "anomaly"] = -1

    # anomaly importances
    forecasted["importance"] = 0
    forecasted.loc[forecasted["anomaly"] == 1, "importance"] = \
        (forecasted["fact"] - forecasted["yhat_upper"]) / forecast["fact"]
    forecasted.loc[forecasted["anomaly"] == -1, "importance"] = \
        (forecasted["yhat_lower"] - forecasted["fact"]) / forecast["fact"]

    return forecasted


pred = detect_anomalies(preds[0])

# In[20]:

# Last 30 days trend information
last_30_days = preds[-30:]
trend_info = last_30_days['trend'].mean()
print("Last 30 days trend information: ", trend_info)
