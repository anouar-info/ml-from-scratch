import numpy as np

class LinearRegression:
    """
    A simple Linear Regression model using Batch Gradient Descent.
    
    Attributes:
        learning_rate (float): The step size for gradient descent updates.
        n_iterations (int): The number of iterations for training.
        theta (np.ndarray): The parameter vector including the intercept.
        cost_history (list): History of cost function values during training.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initializes the LinearRegression model with a learning rate and number of iterations.
        
        Args:
            learning_rate (float): The learning rate for gradient descent.
            n_iterations (int): The number of iterations to run gradient descent.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Trains the Linear Regression model using Batch Gradient Descent.
        
        Args:
            X (np.ndarray): Feature matrix of shape (m, n).
            y (np.ndarray): Target vector of shape (m,).
        """
        # Input validation
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y should be NumPy arrays.")
        
        if X.ndim != 2:
            raise ValueError("X should be a 2D NumPy array.")
        
        if y.ndim != 1:
            raise ValueError("y should be a 1D NumPy array.")
        
        m, n = X.shape  # Number of samples and features
        
        # Initialize theta (n+1 for intercept) with zeros
        self.theta = np.zeros(n + 1)
        
        # Add intercept term to X
        X_b = np.hstack((np.ones((m, 1)), X))  # Shape: (m, n+1)
        
        for iteration in range(self.n_iterations):
            # Compute predictions
            y_pred = X_b @ self.theta  # Shape: (m,)
            
            # Compute residuals
            residuals = y_pred - y  # Shape: (m,)
            
            # Compute cost (Mean Squared Error)
            cost = np.mean(residuals ** 2)
            self.cost_history.append(cost)
            
            # Compute gradients
            gradients = (2/m) * (X_b.T @ residuals)  # Shape: (n+1,)
            
            # Update theta
            self.theta -= self.learning_rate * gradients
            
            # Optional: Print cost every 100 iterations
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: Cost = {cost:.4f}")
    
    def predict(self, X):
        """
        Predicts target values using the trained Linear Regression model.
        
        Args:
            X (np.ndarray): Feature matrix of shape (m, n).
        
        Returns:
            np.ndarray: Predicted values of shape (m,).
        """
        if self.theta is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' with appropriate data.")
        
        if not isinstance(X, np.ndarray):
            raise TypeError("X should be a NumPy array.")
        
        if X.ndim != 2:
            raise ValueError("X should be a 2D NumPy array.")
        
        m, n = X.shape
        
        if n + 1 != self.theta.shape[0]:
            raise ValueError(f"Expected {self.theta.shape[0] - 1} features, but got {n}.")
        
        # Add intercept term to X
        X_b = np.hstack((np.ones((m, 1)), X))  # Shape: (m, n+1)
        
        return X_b @ self.theta  # Shape: (m,)
    
    def mse(self, y_true, y_pred):
        """
        Calculates the Mean Squared Error between true and predicted values.
        
        Args:
            y_true (np.ndarray): Actual target values of shape (m,).
            y_pred (np.ndarray): Predicted target values of shape (m,).
        
        Returns:
            float: Mean Squared Error.
        """
        if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
            raise TypeError("y_true and y_pred should be NumPy arrays.")
        
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        
        return np.mean((y_true - y_pred) ** 2)
    
    def r2_score(self, y_true, y_pred):
        """
        Calculates the R-squared (coefficient of determination) score.
        
        Args:
            y_true (np.ndarray): Actual target values of shape (m,).
            y_pred (np.ndarray): Predicted target values of shape (m,).
        
        Returns:
            float: R-squared score.
        """
        if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
            raise TypeError("y_true and y_pred should be NumPy arrays.")
        
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            raise ValueError("Total sum of squares (ss_tot) is zero, cannot compute R-squared.")
        
        return 1 - (ss_res / ss_tot)
