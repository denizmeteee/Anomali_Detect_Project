import numpy as np
import pandas as pd
from fbprophet import Prophet
import holidays

df_daily = pd.read_csv("bike_sharing_daily.csv")
df_hourly = pd.read_csv("bike_sharing_hourly.csv")
df_deneme = pd.read_excel("Last_30_day_KPIs.xlsx", sheet_name="Sayfa1", engine='openpyxl')

def fixing_datatypes(df):
    df['dteday'] = df['dteday'].astype('datetime64')
    df.loc[:,'season':'mnth'] = df.loc[:,'season':'mnth'].astype('category')
    df[['holiday','workingday']] = df[['holiday','workingday']].astype('bool')
    df[['weekday','weathersit']] = df[['weekday','weathersit']].astype('category')
    return df

df_daily = fixing_datatypes(df_daily)
df_hourly = fixing_datatypes(df_hourly)
df_hourly['hr'] = df_hourly['hr'].astype('category')

datas = []
for col in df_deneme.columns[3:]:
    datas.append(df_deneme[['Date', col]])
datas.pop(0)

for data in datas:
    data.columns = ["ds", "y"]
    data = pd.DataFrame(data=data)

df_daily = df_daily[["cnt","dteday"]]
df_daily.columns = ["y", "ds"]

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", 500)

def fit_predict_model(dataframe):
    m = Prophet(yearly_seasonality = True, daily_seasonality = True)
    m.add_country_holidays(country_name="TR")
    m = m.fit(dataframe)
    forecast = m.predict(dataframe)
    forecast["fact"] = dataframe["y"].reset_index(drop = True)
    return forecast

preds = []
for i, data in enumerate(datas):
    preds.append(fit_predict_model(data))
preds.pop(0)

def detect_anomalies(forecast):
    forecasted = forecast[["ds","trend", "yhat", "yhat_lower", "yhat_upper", "fact"]].copy()
    forecasted["anomaly"] = 0
    forecasted.loc[forecasted["fact"] > forecasted["yhat_upper"], "anomaly"] = 1
    forecasted.loc[forecasted["fact"] < forecasted["yhat_lower"], "anomaly"] = -1
    forecasted["importance"] = 0
    forecasted.loc[forecasted["anomaly"] ==1, "importance"] = \
        (forecasted["fact"] - forecasted["yhat_upper"])/forecast["fact"]
    forecasted.loc[forecasted["anomaly"] ==-1, "importance"] = \
        (forecasted["yhat_lower"] - forecasted["fact"])/forecast["fact"]
    return forecasted

pred = detect_anomalies(preds[0])

# Last 30 days trend information
last_30_days = preds[-30:]
trend_info = last_30_days['trend'].mean()
print("Last 30 days trend information: ", trend_info)