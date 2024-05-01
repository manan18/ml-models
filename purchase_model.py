# -*- coding: utf-8 -*-
"""Purchase Model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OIzSd53V7ZXeNR09wy9Xs8_Apzqgf27i
"""

import pandas as pd
import numpy as np
import pickle
np.random.seed(42)  # For reproducibility

# # Create vendors and products
# vendors = [f"Vendor{i+1}" for i in range(100)]
# products = [f"Product{i+1}" for i in range(100)]

# # Attributes
# data = []
# for vendor in vendors:
#     for product in products:
#         price = np.random.uniform(50, 200)
#         delivery_days = np.random.choice([3, 5, 7, 10, 14])
#         quality_rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.15, 0.4, 0.25, 0.1])
#         fulfillment_rate = np.random.uniform(90, 100)
#         on_time_delivery_rate = np.random.uniform(80, 100)
#         order_accuracy_rate = np.random.uniform(90, 100)
#         financial_health_score = np.random.uniform(0, 1)
#         response_time = np.random.uniform(5, 48)  # Hours

#         data.append([
#             vendor, product, price, delivery_days, quality_rating,
#             fulfillment_rate, on_time_delivery_rate, order_accuracy_rate,
#             financial_health_score, response_time,

#         ])

# # Create DataFrame
# df = pd.DataFrame(data, columns=[
#     'VendorID', 'ProductID', 'Price', 'DeliveryDays', 'QualityRating',
#     'FulfillmentRate', 'OnTimeDeliveryRate', 'OrderAccuracyRate', 'FinancialHealthScore', 'ResponseTime'
# ])



# # Display the first few rows of the dataframe
# print(df)

# # Assign weights to each attribute
# weights = {
#     'Price': 0.50,                 # Lower price is better, hence negative weight
#     'DeliveryDays': 0.1,          # Fewer delivery days are better, hence negative weight
#     'QualityRating': -0.15,         # Direct impact on customer satisfaction
#     'FulfillmentRate': -0.10,       # Importance of fulfilling orders as promised
#     'OnTimeDeliveryRate': -0.10,     # Critical for reliability
#     'OrderAccuracyRate': -0.05,      # Direct impact on customer satisfaction
#     'FinancialHealthScore': -0.05,  # Lower impact but important for long-term stability
#     'ResponseTime': 0.05
# }

# # Normalize the weights to ensure their sum is 1 (optional based on your scoring design)
# total_weight = sum(abs(w) for w in weights.values())
# normalized_weights = {k: v / total_weight for k, v in weights.items()}

# # Calculate the Reliability Score
# def calculate_reliability_score(row):
#     score = 0
#     score += normalized_weights['Price'] * (200 - row['Price']) / 150  # Normalize price between 0 and 1, inverse because lower is better
#     score += normalized_weights['DeliveryDays'] * (14 - row['DeliveryDays']) / 11  # Normalize days, inverse because fewer is better
#     score += normalized_weights['QualityRating'] * row['QualityRating'] / 5  # Normalize quality between 0 and 1
#     score += normalized_weights['FulfillmentRate'] * row['FulfillmentRate'] / 100  # Already 0-100, normalize to 0-1
#     score += normalized_weights['OnTimeDeliveryRate'] * row['OnTimeDeliveryRate'] / 100  # Normalize to 0-1
#     score += normalized_weights['OrderAccuracyRate'] * row['OrderAccuracyRate'] / 100  # Normalize to 0-1
#     score += normalized_weights['FinancialHealthScore'] * row['FinancialHealthScore']  # Already 0-1
#     score += normalized_weights['ResponseTime'] * (48 - row['ResponseTime']) / 43  # Normalize response time, inverse because faster is better
#     return score

# # Apply the function
# df['ReliabilityScore'] = df.apply(calculate_reliability_score, axis=1)

# # Display the updated DataFrame
# print(df[['VendorID', 'ProductID', 'ReliabilityScore']])

# df.to_csv('vendor_data.csv', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('models/vendor_data.csv')

# Assuming 'ReliabilityScore' is the target and all other columns except 'VendorID' and 'ProductID' are features
X = df.drop(['VendorID', 'ProductID', 'ReliabilityScore'], axis=1)
y = df['ReliabilityScore']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialize and train the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)


with open('vendor_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Calculate the Mean Squared Error (MSE) for the model
mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")

import matplotlib.pyplot as plt
import seaborn as sns




def predict(data):
    model = pickle.load(open('vendor_model.pkl', 'rb'))
    new_vendors_df = pd.DataFrame(data)
    X_new_vendors = new_vendors_df.drop(['VendorID', 'ProductID'], axis=1)
    new_vendor_scores = model.predict(X_new_vendors)

    new_vendors_df['PredictedReliabilityScore'] = new_vendor_scores
    # print(new_vendors_df[['VendorID', 'ProductID', 'PredictedReliabilityScore']])

# Group by ProductID and find the entry with the maximum PredictedReliabilityScore for each product
    best_vendors = new_vendors_df.loc[new_vendors_df.groupby('ProductID')['PredictedReliabilityScore'].idxmax()]

# This will return a DataFrame with each row being the best vendor for each ProductID
    

    return best_vendors













# Example data for two vendors offering the same product "Product1"
new_data = {
    'VendorID': ['Vendor101', 'Vendor102', 'Vendor103', 'Vendor104'],
    "ProductID": ["Product1", "Product1", "Product2" , "Product2"],
    "Price": [110, 105 , 210, 190],
    "DeliveryDays": [4, 3, 7, 14],
    "QualityRating": [5, 4, 3, 5],
    "FulfillmentRate": [98, 97, 95, 96],
    "OnTimeDeliveryRate": [92, 95, 90, 96],
    "OrderAccuracyRate": [94, 96, 100, 94],
    "FinancialHealthScore": [0.88, 0.90, 0.92, 0.95],
    "ResponseTime": [20, 18, 12, 20]
}

# Create a DataFrame with this new data
new_vendors_df = pd.DataFrame(new_data)

# Assume any preprocessing steps done during training need to be applied here too

# Prepare the feature matrix from the new DataFrame, omitting identifiers and non-feature columns
X_new_vendors = new_vendors_df.drop(['VendorID', 'ProductID'], axis=1)

# Predict reliability scores using the trained model
new_vendor_scores = model.predict(X_new_vendors)

# Add these scores to the new DataFrame for visibility
new_vendors_df['PredictedReliabilityScore'] = new_vendor_scores
# print(new_vendors_df[['VendorID', 'ProductID', 'PredictedReliabilityScore']])

# Group by ProductID and find the entry with the maximum PredictedReliabilityScore for each product
best_vendors = new_vendors_df.loc[new_vendors_df.groupby('ProductID')['PredictedReliabilityScore'].idxmax()]

# This will return a DataFrame with each row being the best vendor for each ProductID
# print(best_vendors[['ProductID', 'VendorID', 'PredictedReliabilityScore']])