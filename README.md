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












# COVID-19 Case Forecasting with Linear Regression

This project aims to **predict future COVID-19 case counts** using historical data and a **machine learning regression model**. It leverages **time-series analysis techniques** to model and forecast the daily number of new COVID-19 cases in India.

## 📦 Dataset

The dataset is sourced from a convenient global CSV file (`CONVENIENT_global_confirmed_cases.csv`) containing cumulative COVID-19 case counts by country. The project uses **India's data** for demonstration.

## 🚀 Project Workflow

1. **Extract the Dataset**  
   - The dataset is provided in a ZIP file (`covid.zip`) which is extracted in the working directory.

2. **Load and Clean Data**  
   - CSV file is loaded and unnecessary rows/columns are cleaned.
   - Dates are converted to datetime format, and India’s data is isolated.

3. **Feature Engineering**  
   - A new feature, `Day`, is created, representing the number of days since the start of the dataset.

4. **Train-Test Split**  
   - Data is split into training (all data except last 30 days) and testing (last 30 days) sets.

5. **Model Training**  
   - A **Linear Regression model** is trained using the historical case count data.

6. **Predictions and Forecasting**  
   - Model predicts case counts for the last 30 days (test data).
   - Future predictions for the next 30 days are generated.

7. **Visualization**  
   - A line plot visualizes:
     - Actual data  
     - Model’s predictions for test data  
     - Future 30-day forecast  

## 🧰 Libraries Used
- **pandas** for data manipulation  
- **numpy** for numerical operations  
- **scikit-learn** for Linear Regression  
- **matplotlib** for visualization  
- **zipfile** and **os** for file extraction  

## 📈 Results

The project produces a plot comparing:
✅ Actual COVID-19 cases  
✅ Model’s predictions on the last 30 days  
✅ Forecast for the next 30 days  

This allows for a straightforward understanding of how the model performs and its forecast for future COVID-19 trends.

## 📁 How to Run

1. **Clone the repository** (if applicable)  
2. Place your `covid.zip` file in the working directory (e.g., Google Colab or local environment).  
3. Run the Python script / Jupyter notebook to extract, clean, train the model, and visualize the results.

## 🤝 Contribution

Feel free to suggest improvements, additional forecasting techniques (like ARIMA, LSTM), or expand the analysis to other countries!

## 📄 License

This project is for educational purposes and uses publicly available data.

---

**Date:** [27-06-2025]


