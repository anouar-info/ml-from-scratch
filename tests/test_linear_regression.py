import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("C:/Users/anouar/Desktop/ml-from-scratch")
from algorithms.supervised.regression.linear_regression.linear_regression import LinearRegression  # Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Create an instance of LinearRegression
lr = LinearRegression(learning_rate=0.01, n_iterations=1000)

# Fit the model
lr.fit(X, y)

# Make predictions
X_new = np.array([[0], [2]])
y_predict = lr.predict(X_new)

# Plot the data and predictions
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(X, y, 'b.')
plt.plot(X_new, y_predict, 'r-')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')

plt.subplot(1, 2, 2)
plt.plot(range(len(lr.cost_history)), lr.cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost History Over Iterations')

plt.tight_layout()
plt.show()