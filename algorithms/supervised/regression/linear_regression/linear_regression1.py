import numpy as np

class LinearRegression1 :
    def __init__(self,learning_rate=0.01,n_iterations=1000):
        self.learning_rate=learning_rate
        self.n_iterations=n_iterations
        self.theta=None
        self.cost_history=[]
    def fit(self,X,y):
        m, n = X.shape 
        print(f"ana homa m : {m} and n : {n} ")
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
    def mse(self,y_true, y_pred) :
        mse  = np.mean((y_true-y_pred)**2)
        return mse 
    def r2_score(self,y_true, y_pred):
        y_mean = np.mean(y_true)
        r2 = 1 - (((y_true-y_pred)**2)/((y_true-y_mean)**2))
        return r2