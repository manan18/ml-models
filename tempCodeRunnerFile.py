num_transactions = 1000

# Generate random dates and times within a year
start_date = pd.to_datetime('2023-01-01')
end_date = pd.to_datetime('2023-12-31')
date_range = pd.to_datetime(np.random.choice(pd.date_range(start_date, end_date, freq='H'), num_transactions, replace=True))

data = {
    'Date': date_range,
    'Transaction Amount': np.random.uniform(20, 500, size=num_transactions),
    'Product ID': np.random.randint(1000, 1050, size=num_transactions),
    'Customer ID': np.random.randint(1, 300, size=num_transactions),
    'Payment Method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash'], size=num_transactions),
    'Sales Channel': np.random.choice(['Online', 'In-store'], size=num_transactions),
    'Product Category': np.random.choice(['Electronics', 'Clothing', 'Household'], size=num_transactions),
    'Product Specs': np.random.choice(['Small-Red', 'Medium-Blue', 'Large-Green'], size=num_transactions),
    'Listed Price': np.random.uniform(25, 600, size=num_transactions),
    'Discount': np.random.uniform(0, 0.2, size=num_transactions) * np.random.uniform(25, 600, size=num_transactions)
}

# Create DataFrame
df = pd.DataFrame(data)

# Optionally calculate the final price after the discount
df['Final Price'] = df['Listed Price'] - df['Discount']

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Resample the data to daily, summing up the transaction amounts
daily_transactions = df.resample('D').sum()

# Reset the index to turn 'Date' back into a column
daily_transactions.reset_index(inplace=True)

# Check the aggregated daily data
print(daily_transactions.head())

# Resample the data to daily, aggregating transaction amounts
daily_data = df.resample('D').agg({
    'Transaction Amount': 'sum',  # Sum transaction amounts
    'Product ID': 'mean',  # You could choose 'mean' or another method that makes sense
    'Customer ID': 'mean',  # Same as above
    'Payment Method': lambda x: x.mode()[0] if not x.mode().empty else np.NaN,  # Most frequent or NaN if no mode
    'Sales Channel': lambda x: x.mode()[0] if not x.mode().empty else np.NaN,
    'Product Category': lambda x: x.mode()[0] if not x.mode().empty else np.NaN,
    'Product Specs': lambda x: x.mode()[0] if not x.mode().empty else np.NaN,
    'Listed Price': 'mean',
    'Discount': 'mean',
    'Final Price': 'mean'
})

# Rename 'Transaction Amount' to 'TransactionAmount'
daily_data.rename(columns={'Transaction Amount': 'TransactionAmount'}, inplace=True)

# Reset the index to turn 'Date' back into a column
daily_data.reset_index(inplace=True)
daily_data.dropna(inplace=True)
# Check the aggregated daily data
df=daily_data

df['Date'] = pd.to_datetime(df['Date'])