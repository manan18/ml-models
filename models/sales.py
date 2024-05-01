import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)

# # Define the total number of transactions
# num_transactions = 1000

# # Generate random dates and times within a year
# start_date = pd.to_datetime('2023-01-01')
# end_date = pd.to_datetime('2023-12-31')
# date_range = pd.to_datetime(np.random.choice(pd.date_range(start_date, end_date, freq='H'), num_transactions, replace=True))

# data = {
#     'Date': date_range,
#     'Transaction Amount': np.random.uniform(20, 500, size=num_transactions),
#     'Product ID': np.random.randint(1000, 1050, size=num_transactions),
#     'Customer ID': np.random.randint(1, 300, size=num_transactions),
#     'Payment Method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash'], size=num_transactions),
#     'Sales Channel': np.random.choice(['Online', 'In-store'], size=num_transactions),
#     'Product Category': np.random.choice(['Electronics', 'Clothing', 'Household'], size=num_transactions),
#     'Product Specs': np.random.choice(['Small-Red', 'Medium-Blue', 'Large-Green'], size=num_transactions),
#     'Listed Price': np.random.uniform(25, 600, size=num_transactions),
#     'Discount': np.random.uniform(0, 0.2, size=num_transactions) * np.random.uniform(25, 600, size=num_transactions)
# }

# # Create DataFrame
# df = pd.DataFrame(data)

# # Optionally calculate the final price after the discount
# df['Final Price'] = df['Listed Price'] - df['Discount']




# # Set 'Date' as the index
# df.set_index('Date', inplace=True)

# # Resample the data to daily, summing up the transaction amounts
# daily_transactions = df.resample('D').sum()

# # Reset the index to turn 'Date' back into a column
# daily_transactions.reset_index(inplace=True)

# # Check the aggregated daily data
# print(daily_transactions.head())
# # Resample the data to daily, aggregating transaction amounts
# daily_data = df.resample('D').agg({
#     'Transaction Amount': 'sum',  # Sum transaction amounts
#     'Product ID': 'mean',  # You could choose 'mean' or another method that makes sense
#     'Customer ID': 'mean',  # Same as above
#     'Payment Method': lambda x: x.mode()[0] if not x.mode().empty else np.NaN,  # Most frequent or NaN if no mode
#     'Sales Channel': lambda x: x.mode()[0] if not x.mode().empty else np.NaN,
#     'Product Category': lambda x: x.mode()[0] if not x.mode().empty else np.NaN,
#     'Product Specs': lambda x: x.mode()[0] if not x.mode().empty else np.NaN,
#     'Listed Price': 'mean',
#     'Discount': 'mean',
#     'Final Price': 'mean'
# })

# # Rename 'Transaction Amount' to 'TransactionAmount'
# daily_data.rename(columns={'Transaction Amount': 'TransactionAmount'}, inplace=True)

# # Reset the index to turn 'Date' back into a column
# daily_data.reset_index(inplace=True)
# daily_data.dropna(inplace=True)
# # Check the aggregated daily data
# df=daily_data
# df['Date'] = pd.to_datetime(df['Date'])

# print(df)

# df.to_csv('daily_sales_data.csv', index=True)

df=pd.read_csv('daily_sales_data.csv')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBRegressor
import pickle
# Define features and target variable
X = df[['Product ID', 'Payment Method', 'Sales Channel', 'Product Category', 'Product Specs', 'Final Price']]
y = df['TransactionAmount']
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# Encode categorical variables
label_encoder = LabelEncoder()
for col in X.columns[X.dtypes == 'object']:
    X[col] = label_encoder.fit_transform(X[col])

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y = y.values.reshape(-1, 1)
y_scaled = scaler.fit_transform(y)

# Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Initialize and train the XGBRegressor
# model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
# model.fit(X_train, y_train)
model = pickle.load(open('sales_model.pkl', 'rb'))
with open('sales_model.pkl', 'wb') as f:
    pickle.dump(model, f)
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
# Define US business day taking account of federal holidays
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
import matplotlib.pyplot as plt
# Generate future business days for prediction
n_days_for_prediction = 17  # days to predict into the future
last_date = df['Date'].max()
# predict_period_dates = pd.date_range(last_date, periods=n_days_for_prediction+1, freq=us_bd)[1:]
#print(predict_period_dates)

# Make prediction (using the last part of X_test as a stub; replace appropriately in practice)
# predictions_scaled = model.predict(X_scaled[-n_days_for_prediction:])
# predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
#print(predictions)
# Visualize predictions against actual data
# plt.figure(figsize=(30, 15))
# plt.plot(df['Date'][-90:], y[-90:], label='Actual Sales', marker='o')  # Last 30 actual points
# plt.plot(predict_period_dates, predictions, label='Predicted Sales', marker='x')
# plt.title('Forecast of Sales')
# plt.xlabel('Date')
# plt.ylabel('Transaction Amount')
# plt.legend()
# plt.show()

def predict2(n_days_for_prediction):
    predict_period_dates = pd.date_range(last_date, periods=n_days_for_prediction+1, freq=us_bd)[1:]
    predictions_scaled = model.predict(X_scaled[-n_days_for_prediction:])
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
    print(predictions)

    return predictions.tolist(), predict_period_dates.tolist()
