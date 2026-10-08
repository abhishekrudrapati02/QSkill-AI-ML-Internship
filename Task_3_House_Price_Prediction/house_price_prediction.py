# QSkill AI/ML Internship - Task 3: House Price Prediction

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load the California Housing dataset
housing = fetch_california_housing()

# Create a DataFrame
df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

# Add the house price as the target
df["Price"] = housing.target


# Explore the dataset
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())


# Display basic statistics
print("\nDataset Statistics:")
print(df.describe())


# Visualize house prices
plt.figure(figsize=(8, 5))

plt.hist(df["Price"], bins=30)

plt.title("Distribution of House Prices")
plt.xlabel("House Price")
plt.ylabel("Number of Houses")

plt.savefig("house_price_distribution.png")
plt.show()


# Select features and target
X = df.drop("Price", axis=1)
y = df["Price"]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)


# Compare actual and predicted prices
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.savefig("actual_vs_predicted.png")
plt.show()


print("\nHouse Price Prediction completed successfully!")