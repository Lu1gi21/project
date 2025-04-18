import numpy as np
import matplotlib.pyplot as plt

class GradientDescentRegression:
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.cost_history = []
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        self.weights = np.zeros(2)
        
        for i in range(self.n_iterations):
            y_pred = X_b.dot(self.weights)
            
            gradients = 2/X.shape[0] * X_b.T.dot(y_pred - y)
            
            self.weights = self.weights - self.learning_rate * gradients
            
            cost = self._compute_cost(X_b, y)
            self.cost_history.append(cost)
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weights)
    
    def _compute_cost(self, X_b, y):
        y_pred = X_b.dot(self.weights)
        return np.mean((y_pred - y) ** 2)
    
    def score(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def plot_regression_line(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        y_pred = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Data points')
        plt.plot(X, y_pred, color='red', label='Regression line')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Gradient Descent Linear Regression')
        plt.legend()
        plt.show()
        
    def plot_cost_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.cost_history) + 1), self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost History During Training')
        plt.show()
