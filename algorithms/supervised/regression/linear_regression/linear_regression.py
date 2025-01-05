import numpy as np


class LinearRegression :
    def __init__(self,learning_rate=0.01,n_iterations=1000,theta=np.array(),cost_history=[]):
        self.learning_rate=learning_rate
        self.n_iterations=n_iterations
        self.theta=theta
        self.cost_history=cost_history
    def fit(self,X,y):
        