from linear_regression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
model = LinearRegression()
noice = np.random.normal(100)
X = np.linspace(0,10,100).reshape(-1,1)
y = 2* X +1 + noice
theta = model.fit(X,y)
y_pred = X @ theta
plt.scatter(X,y)
plt.plot(X,y_pred,color="red")
plt.show