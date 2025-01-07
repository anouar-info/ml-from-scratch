import numpy as np

class LinearRegression :
    def __init__(self,learning_rate=0.01,n_iterations=1000):
        self.learning_rate=learning_rate
        self.n_iterations=n_iterations
        self.theta=None
        self.cost_history=[]
    def fit(self,X,y):
        m, n = X.shape 
        self.theta= np.random.randn(n+1,1)
        X_b = np.c_[np.ones((m,1)),X]
        y = y.reshape(-1, 1)
        for _ in range(self.n_iterations) :
            gradients = (2/m) * X_b.T @ (X_b @ self.theta - y)
            self.theta -= self.learning_rate * gradients 
            y_pred =  X_b @ self.theta
            cost = np.mean((y - y_pred) ** 2)
            self.cost_history.append(cost)
        
    def predict(self, X):
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        return X_b @ self.theta 