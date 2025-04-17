import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_indices = distances.argsort()[:self.k]
            k_labels = self.y_train[k_indices]
            majority_vote = Counter(k_labels).most_common(1)[0][0]
            predictions.append(majority_vote)
        return np.array(predictions)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    def score(self, X, y_true):
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)
