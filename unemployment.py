import yfinance as yf
import matplotlib.pyplot as plt

# Fetch historical unemployment rate data from Yahoo Finance
unemployment_data = yf.download("UNRATE", start="2000-01-01", end="2023-01-01")

# Print the first few rows of the data
print(unemployment_data.head())

# Plot the unemployment rate over time
plt.figure(figsize=(10, 6))
plt.plot(unemployment_data['Close'], label='Unemployment Rate')
plt.title('U.S. Unemployment Rate Over Time')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.show()


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Function to fetch unemployment rate data from Yahoo Finance
def fetch_unemployment_data():
    data = yf.download("UNRATE", start="2000-01-01", end="2023-01-01")
    return data['Close'].reset_index()

# Function to load unemployment rate data from a CSV file
def load_unemployment_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Date'])

# Function for data cleaning and preprocessing
def clean_and_process_data(data):
    data = data.dropna()  # Drop rows with missing values
    return data

# Function for statistical analysis
def analyze_data(data):
    summary_stats = data.describe()
    correlation_matrix = data.corr()
    return summary_stats, correlation_matrix

# Function for visualization
def visualize_data(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Unemployment Rate')
    plt.title('U.S. Unemployment Rate Over Time')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to train a simple linear regression model
def train_linear_regression_model(data):
    X = data[['Date']].astype(int)  # Convert date to numerical format
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = metrics.mean_squared_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)

    return model, mse, r2

# Fetch or load unemployment rate data
# Uncomment one of the following lines based on your preference
# unemployment_data = fetch_unemployment_data()
unemployment_data = load_unemployment_data('path/to/your/unemployment_data.csv')

# Data cleaning and preprocessing
cleaned_data = clean_and_process_data(unemployment_data)

# Statistical analysis
summary_stats, correlation_matrix = analyze_data(cleaned_data)
print("\nSummary Statistics:")
print(summary_stats)
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualization
visualize_data(cleaned_data)

# Train a linear regression model
model, mse, r2 = train_linear_regression_model(cleaned_data)

# Print model evaluation metrics
print("\nLinear Regression Model:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
