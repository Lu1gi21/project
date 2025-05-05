import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    
    def __init__(self):
        self.weights = None
        
    def fit(self, X, y):
        X = np.array(X) 
        y = np.array(y)
        
        # Add bias term 
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Normal equation 
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        return self
    
    def predict(self, X):
        X = np.array(X)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weights)
    
    def score(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        
        # Total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        # Residual sum of squares
        ss_res = np.sum((y - y_pred) ** 2)
        
        # R^2 score
        r2 = 1 - (ss_res / ss_tot) 
        return r2
    
