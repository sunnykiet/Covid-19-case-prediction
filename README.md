import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# STEP 1: Extract ZIP file
zip_file_path = '/content/covid.zip'
extract_to = 'covid_data'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

# STEP 2: Load India data from the convenient dataset
file_path = os.path.join(extract_to, 'CONVENIENT_global_confirmed_cases.csv')
df = pd.read_csv(file_path)

# STEP 3: Clean data
df = df.drop(index=0)  # remove header row inside data
df = df.rename(columns={"Country/Region": "Date"})
df['Date'] = pd.to_datetime(df['Date'])  # convert to datetime
india_df = df[['Date', 'India']].copy()
india_df['India'] = india_df['India'].astype(float)

# STEP 4: Feature engineering
india_df['Day'] = (india_df['Date'] - india_df['Date'].min()).dt.days

# STEP 5: Train/test split
train = india_df[:-30]
test = india_df[-30:]

# STEP 6: Linear Regression model
model = LinearRegression()
model.fit(train[['Day']], train['India'])
test['Predicted'] = model.predict(test[['Day']])

# STEP 7: Forecast next 30 days
future_days = 30
last_day = india_df['Day'].max()
future = pd.DataFrame({'Day': range(last_day + 1, last_day + 1 + future_days)})
future['Predicted'] = model.predict(future[['Day']])
future['Date'] = india_df['Date'].max() + pd.to_timedelta(future['Day'] - last_day, unit='D')

# STEP 8: Plot the results
plt.figure(figsize=(12, 6))
plt.plot(india_df['Date'], india_df['India'], label='Actual')
plt.plot(test['Date'], test['Predicted'], label='Test Prediction', linestyle='--')
plt.plot(future['Date'], future['Predicted'], label='Future Forecast', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Cumulative COVID-19 Cases')
plt.title(' COVID-19 Case Forecast (Linear Regression)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()# Covid-19-case-prediction
