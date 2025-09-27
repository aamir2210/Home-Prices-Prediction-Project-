import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Title
st.title("🏡 Home Prices Prediction (Linear Regression)")
st.write("Predict house prices from square footage using a simple linear regression model.")

# Generate synthetic dataset
np.random.seed(0)
house_sizes = np.random.randint(800, 5000, 100)
house_prices = 150 * house_sizes + np.random.normal(scale=10000, size=len(house_sizes))

# Split data (fixed test size + random state, no sliders)
x_train, x_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size=0.2, random_state=42)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# User input for prediction
sqft = st.number_input("Enter house size (sq.ft):", min_value=800, max_value=5000, value=2000, step=50)
predicted_price = model.predict([[sqft]])[0]

st.success(f"Estimated Price for {sqft} sq.ft: **${int(predicted_price):,}**")

# Plot 1: House Prices vs House Sizes
fig1, ax1 = plt.subplots()
ax1.scatter(house_sizes, house_prices, color="blue")
ax1.set_title("House Prices vs. House Sizes")
ax1.set_xlabel("House Size (sq.ft)")
ax1.set_ylabel("House Price ($)")
st.pyplot(fig1)

# Plot 2: Actual vs Predicted
predictions = model.predict(x_test)
fig2, ax2 = plt.subplots()
ax2.scatter(x_test, y_test, color="blue", label="Actual Prices")
ax2.plot(x_test, predictions, color="red", linewidth=2, label="Predicted Prices")
ax2.set_title("House Price Prediction with Linear Regression")
ax2.set_xlabel("House Size (sq.ft)")
ax2.set_ylabel("House Price ($)")
ax2.legend()
st.pyplot(fig2)

