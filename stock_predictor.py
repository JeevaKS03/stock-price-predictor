import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Download Apple stock data
data = yf.download('AAPL', start='2023-01-01', end='2023-12-31')
if data is None:
	raise ValueError("Failed to download stock data. Please check the ticker symbol or internet connection.")
data = data[['Close']].reset_index()
data['Day'] = range(len(data))

# Prepare data
X = data[['Day']]
y = data['Close']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next day
next_day = np.array([[len(data)]])
prediction = model.predict(next_day)
print("Predicted price for next day:", prediction[0])

# Plot
plt.plot(data['Day'], y, label="Actual")
plt.plot(data['Day'], model.predict(X), label="Predicted", linestyle='--')
plt.legend()
plt.title("Stock Price Prediction (Linear Regression)")
plt.show()

#Testing Predict next 10 days 

future_days = pd.DataFrame({'Day': range(len(data), len(data) + 10)})
future_prices = model.predict(future_days)

# Plot future predictions
plt.plot(future_days['Day'], future_prices, label='Future Prediction', color='green', linestyle='dotted')
plt.legend()
plt.title("Stock Price Prediction with Future Forecast")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()

# Print future predicted prices
print("\nFuture predictions:")
for i, price in enumerate(future_prices, start=1):
    print(f"Day +{i}: ₹{price[0]:.2f}")
