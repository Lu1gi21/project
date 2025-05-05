import numpy as np


class SimpleNeuralNetwork:
    """
    A simple neural net with one or two hidden layers.
    """
    def __init__(self, n_in, h1, n_out, h2=None, lr=0.01, epochs=1000):
        # store the size of the optional second hidden layer
        self.h2 = h2
        # learning rate for gradient descent
        self.lr = lr
        # number of training epochs
        self.epochs = epochs
        # keep track of loss at each epoch
        self.losses = []

        # weights and biases for the first hidden layer
        self.w1 = np.random.randn(n_in, h1)
        self.b1 = np.zeros((1, h1))

        if h2 is not None:
            # if a second hidden layer is used
            self.w2 = np.random.randn(h1, h2)
            self.b2 = np.zeros((1, h2))
            # output layer weights/biases
            self.w3 = np.random.randn(h2, n_out)
            self.b3 = np.zeros((1, n_out))
        else:
            # only one hidden layer
            self.w2 = np.random.randn(h1, n_out)
            self.b2 = np.zeros((1, n_out))

    def sigmoid(self, z):
        """sigmoid activation function"""
        return 1 / (1 + np.exp(-z))

    def sigmoid_deriv(self, z):
        """derivative of sigmoid, for backprop"""
        s = self.sigmoid(z)
        return s * (1 - s)

    def fit(self, X, y):
        # make sure X and y are numpy arrays
        X = np.array(X)
        # reshape y to match output layer
        y = np.array(y).reshape(-1, self.w2.shape[1] if self.h2 is None else self.w3.shape[1])
        n = X.shape[0]

        for i in range(self.epochs):
            # --- forward pass ---
            z1 = X.dot(self.w1) + self.b1
            a1 = self.sigmoid(z1)
            if self.h2 is not None:
                z2 = a1.dot(self.w2) + self.b2
                a2 = self.sigmoid(z2)
                z3 = a2.dot(self.w3) + self.b3
                preds = z3
            else:
                z2 = a1.dot(self.w2) + self.b2
                preds = z2

            # --- loss ---
            loss = np.mean((preds - y) ** 2)
            self.losses.append(loss)

            # --- backward pass ---
            if self.h2 is not None:
                
                dz3 = 2 * (preds - y) / n
                dw3 = a2.T.dot(dz3)
                db3 = np.sum(dz3, axis=0, keepdims=True)

                da2 = dz3.dot(self.w3.T)
                dz2 = da2 * self.sigmoid_deriv(z2)
                dw2 = a1.T.dot(dz2)
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = dz2.dot(self.w2.T) 
                dz1 = da1 * self.sigmoid_deriv(z1)
                dw1 = X.T.dot(dz1)
                db1 = np.sum(dz1, axis=0, keepdims=True)

                # update weights and biases
                self.w1 -= self.lr * dw1
                self.b1 -= self.lr * db1
                self.w2 -= self.lr * dw2
                self.b2 -= self.lr * db2
                self.w3 -= self.lr * dw3
                self.b3 -= self.lr * db3
            else:
                # only one hidden layer
                dz2 = 2 * (preds - y) / n 
                dw2 = a1.T.dot(dz2)
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = dz2.dot(self.w2.T)
                dz1 = da1 * self.sigmoid_deriv(z1)
                dw1 = X.T.dot(dz1)
                db1 = np.sum(dz1, axis=0, keepdims=True)

                self.w1 -= self.lr * dw1
                self.b1 -= self.lr * db1
                self.w2 -= self.lr * dw2
                self.b2 -= self.lr * db2

    def predict(self, X):
        # just run a forward pass
        X = np.array(X)
        z1 = X.dot(self.w1) + self.b1
        a1 = self.sigmoid(z1)
        if self.h2 is not None:
            z2 = a1.dot(self.w2) + self.b2
            a2 = self.sigmoid(z2)
            z3 = a2.dot(self.w3) + self.b3
            return z3
        else:
            z2 = a1.dot(self.w2) + self.b2
            return z2