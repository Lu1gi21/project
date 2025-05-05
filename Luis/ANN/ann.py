import numpy as np


class SimpleNeuralNetwork:
    """
    A simple feedforward neural network with one or two hidden layers.
    """
    def __init__(self, input_dim, hidden_dim1, output_dim, hidden_dim2=None, learning_rate=0.01, n_iterations=1000):
        self.hidden_dim2 = hidden_dim2
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.cost_history = []

        # First hidden layer
        self.W1 = np.random.randn(input_dim, hidden_dim1)
        self.b1 = np.zeros((1, hidden_dim1))

        if hidden_dim2 is not None:
            # Second hidden layer
            self.W2 = np.random.randn(hidden_dim1, hidden_dim2)
            self.b2 = np.zeros((1, hidden_dim2))
            # Output layer
            self.W3 = np.random.randn(hidden_dim2, output_dim)
            self.b3 = np.zeros((1, output_dim))
        else:
            # Output layer (directly from first hidden layer)
            self.W2 = np.random.randn(hidden_dim1, output_dim)
            self.b2 = np.zeros((1, output_dim))

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """Derivative of sigmoid."""
        s = self.sigmoid(z)
        return s * (1 - s)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, self.W2.shape[1] if self.hidden_dim2 is None else self.W3.shape[1])
        n_samples = X.shape[0]

        for i in range(self.n_iterations):
            # Forward pass
            Z1 = X.dot(self.W1) + self.b1
            A1 = self.sigmoid(Z1)
            if self.hidden_dim2 is not None:
                Z2 = A1.dot(self.W2) + self.b2
                A2 = self.sigmoid(Z2)
                Z3 = A2.dot(self.W3) + self.b3
                y_pred = Z3
            else:
                Z2 = A1.dot(self.W2) + self.b2
                y_pred = Z2

            # Compute mean squared error loss
            loss = np.mean((y_pred - y) ** 2)
            self.cost_history.append(loss)

            # Backward pass
            if self.hidden_dim2 is not None:
                dZ3 = 2 * (y_pred - y) / n_samples
                dW3 = A2.T.dot(dZ3)
                db3 = np.sum(dZ3, axis=0, keepdims=True)

                dA2 = dZ3.dot(self.W3.T)
                dZ2 = dA2 * self.sigmoid_derivative(Z2)
                dW2 = A1.T.dot(dZ2)
                db2 = np.sum(dZ2, axis=0, keepdims=True)

                dA1 = dZ2.dot(self.W2.T)
                dZ1 = dA1 * self.sigmoid_derivative(Z1)
                dW1 = X.T.dot(dZ1)
                db1 = np.sum(dZ1, axis=0, keepdims=True)

                # Update weights and biases
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2
                self.W3 -= self.learning_rate * dW3
                self.b3 -= self.learning_rate * db3
            else:
                dZ2 = 2 * (y_pred - y) / n_samples
                dW2 = A1.T.dot(dZ2)
                db2 = np.sum(dZ2, axis=0, keepdims=True)

                dA1 = dZ2.dot(self.W2.T)
                dZ1 = dA1 * self.sigmoid_derivative(Z1)
                dW1 = X.T.dot(dZ1)
                db1 = np.sum(dZ1, axis=0, keepdims=True)

                # Update weights and biases
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2

    def predict(self, X):
        X = np.array(X)
        Z1 = X.dot(self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        if self.hidden_dim2 is not None:
            Z2 = A1.dot(self.W2) + self.b2
            A2 = self.sigmoid(Z2)
            Z3 = A2.dot(self.W3) + self.b3
            return Z3
        else:
            Z2 = A1.dot(self.W2) + self.b2
            return Z2