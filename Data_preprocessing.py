import pandas as pd
import numpy as np

df=pd.read_csv("gig_worker_task_pricing_dataset_1000.csv")
print(df.head())

#view total column
print(df.columns.tolist())

#drop leakage column
leakage_cols = [
'net_income_rm',
'hourly_income_rm',
'suggested_price_rm'   # optional if predicting accept
]

df_model = df.drop(columns=leakage_cols)

# check missing value
print(df.isnull().sum())

#check duplicate 
print("Duplicated rows:", df.duplicated().sum())

# convert and extract date column
df["date"] = pd.to_datetime(df["date"])

df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

# drop redundant column
df = df.drop(columns=["date","task_id"],axis=1)

print(df.columns)





