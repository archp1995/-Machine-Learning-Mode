import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship with noise

# Step 2: Model Training
model = LinearRegression()
model.fit(X, y)

# Step 3: Model Evaluation
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

# Step 4: Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.title('Linear Regression')
plt.xlabel('Number of Rooms')
plt.ylabel('House Price')
plt.legend()
plt.show()
