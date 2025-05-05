import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # index of the feature to split on
        self.threshold = threshold          # value of the threshold to split at
        self.left = left                    # left subtree (if any)
        self.right = right                  # right subtree (if any)
        self.value = value                  # class label (if leaf)

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _gini(self, y):
        m = len(y)
        counts = np.bincount(y)
        probabilities = counts / m
        return 1.0 - np.sum(probabilities ** 2)

    def _split(self, X, y, feature_index, threshold):
        left_idx = X[:, feature_index] <= threshold
        right_idx = X[:, feature_index] > threshold
        return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_idx, best_thresh = None, None
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)
                weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_idx = feature_index
                    best_thresh = threshold

        return best_idx, best_thresh

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_classes = len(y), len(set(y))
        if depth >= self.max_depth or num_classes == 1 or num_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            return Node(value=self._most_common_label(y))

        X_left, y_left, X_right, y_right = self._split(X, y, feature_idx, threshold)
        left = self._grow_tree(X_left, y_left, depth + 1)
        right = self._grow_tree(X_right, y_right, depth + 1)
        return Node(feature_index=feature_idx, threshold=threshold, left=left, right=right)

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]
