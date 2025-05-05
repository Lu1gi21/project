import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add project directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import models and utilities
from Luis.SLR.slr import SimpleLinearRegression
from Data.data import load_clean_vgsales, get_numeric_vgsales_columns
from Luis.ANN.ann import SimpleNeuralNetwork

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    """
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    # Shuffle indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    # Compute split index
    test_count = int(len(indices) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    # Split the data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test

def cooks_distance(X, y, model):
    """
    Calculate Cook's distance for each observation in a simple linear regression.
    """
    # Add intercept
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.mean(residuals ** 2)
    # Hat matrix diagonal
    H = X_b @ np.linalg.inv(X_b.T @ X_b) @ X_b.T
    leverage = np.diag(H)
    # Cook's distance formula
    cooks_d = (residuals ** 2) / (2 * mse) * (leverage / (1 - leverage) ** 2)
    return cooks_d

def plot_predictions(y_true, y_pred, title, filename=None):
    """
    Plot actual vs predicted values for regression.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, label='Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect prediction')
    plt.xlabel('Actual Global Sales')
    plt.ylabel('Predicted Global Sales')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(f"Luis/plots/{filename}")
    plt.close()

def plot_slr_fit(X, y, model, filename=None):
    """
    Plot data points and the regression line for SLR.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
    plt.plot(X, model.predict(X), color='red', label='SLR Fit')
    plt.xlabel('NA Sales')
    plt.ylabel('Global Sales')
    plt.title('Simple Linear Regression Fit')
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(f"Luis/plots/{filename}")
    plt.close()

def plot_ann_loss(ann_model, filename=None):

    plt.figure(figsize=(8, 6))
    plt.plot(ann_model.losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('ANN Training Loss Curve')
    plt.tight_layout()
    if filename:
        plt.savefig(f"Luis/plots/{filename}")
    plt.close()


def plot_cooks_distance(cooks_d, threshold=None, filename=None):
    """
    Plot Cook's distance for each observation in SLR.
    """
    plt.figure(figsize=(10, 6))
    plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt='o', basefmt=" ")
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.4f}')
    plt.xlabel('Observation Index')
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance for SLR")
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(f"Luis/plots/{filename}")
    plt.close()

def experiment_predict_global_sales():
    """
    Experiment to predict Global_Sales based on NA_Sales using Simple Linear Regression only.
    """
    # Load and prepare data
    df = load_clean_vgsales()
    df_numeric = get_numeric_vgsales_columns(df)
    
    # Prepare features 
    X = df_numeric['NA_Sales'].values.reshape(-1, 1)
    y = df_numeric['Global_Sales'].values
    
    # Fit model to ALL data to compute Cook's distance
    slr_model_all = SimpleLinearRegression()
    slr_model_all.fit(X, y)
    cooks_d = cooks_distance(X, y, slr_model_all)

    # Plot Cook's distance
    threshold = 4 / len(X)
    plot_cooks_distance(cooks_d, threshold=threshold, filename="cooks_distance_slr.png")

    # Remove outliers
    mask = cooks_d < threshold
    X = X[mask]
    y = y[mask]
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Simple Linear Regression model
    slr_model = SimpleLinearRegression()
    slr_model.fit(X_train, y_train)
    slr_score = slr_model.score(X_test, y_test)
    slr_predictions = slr_model.predict(X_test)
    
    print(f"SLR R² score: {slr_score:.4f}")
    print(f"SLR Coefficients: {slr_model.weights}")
    
    # Plot SLR fit and predictions
    plot_slr_fit(X_test, y_test, slr_model, filename="slr_fit.png")
    plot_predictions(y_test, slr_predictions, "SLR: Actual vs Predicted Global Sales", filename="slr_actual_vs_predicted.png")

def experiment_sales_by_year():
    """
    Experiment to predict Global_Sales based on NA_Sales, EU_Sales, JP_Sales, and Other_Sales using a simple ANN.
    """
    # Load and prepare data
    df = load_clean_vgsales()
    df_numeric = get_numeric_vgsales_columns(df)
    
    # Select features
    feature_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    X = df_numeric[feature_cols].values
    y = df_numeric['Global_Sales'].values
    
    # Fit model to ALL data to compute Cook's distance
    slr_model_all = SimpleLinearRegression()
    slr_model_all.fit(X, y)
    cooks_d = cooks_distance(X, y, slr_model_all)

    # Remove outliers 
    threshold = 4 / len(X)
    mask = cooks_d < threshold
    X = X[mask]
    y = y[mask]
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train_scaled = (X_train - X_train_mean) / X_train_std
    X_test_scaled = (X_test - X_train_mean) / X_train_std

    # --- ANN Model ---
    input_dim = X_train_scaled.shape[1]
    output_dim = 1

    # One hidden layer
    for hidden_dim in [4, 16, 32, 64]:
        ann_model = SimpleNeuralNetwork(input_dim, hidden_dim, output_dim, h2=None, lr=0.01, epochs=1000)
        ann_model.fit(X_train_scaled, y_train)
        ann_pred = ann_model.predict(X_test_scaled).flatten()
        ann_score = 1 - np.sum((ann_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        print(f"ANN (1 hidden, {hidden_dim} neurons) R² score: {ann_score:.4f}")
        
        # Plot actual vs predicted sales
        plot_predictions(y_test, ann_pred, f"ANN (1 hidden, {hidden_dim} neurons): Actual vs Predicted Global Sales", filename=f"ann1_hidden{hidden_dim}_actual_vs_predicted.png")
        
        # Plot training loss curve
        plot_ann_loss(ann_model, filename=f"ann1_hidden{hidden_dim}_loss_curve.png")

    # Two hidden layers
    for hidden_dim1, hidden_dim2 in [(16, 8), (32, 16), (64, 32)]:
        ann_model = SimpleNeuralNetwork(input_dim, hidden_dim1, output_dim, h2=hidden_dim2, lr=0.01, epochs=1000)
        ann_model.fit(X_train_scaled, y_train)
        ann_pred = ann_model.predict(X_test_scaled).flatten()
        ann_score = 1 - np.sum((ann_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        print(f"ANN (2 hidden, {hidden_dim1}-{hidden_dim2} neurons) R² score: {ann_score:.4f}")
        # Plot actual vs predicted sales
        plot_predictions(y_test, ann_pred, f"ANN (2 hidden, {hidden_dim1}-{hidden_dim2} neurons): Actual vs Predicted Global Sales", filename=f"ann2_hidden{hidden_dim1}_{hidden_dim2}_actual_vs_predicted.png")
        # Plot training loss curve
        plot_ann_loss(ann_model, filename=f"ann2_hidden{hidden_dim1}_{hidden_dim2}_loss_curve.png")


if __name__ == "__main__":
    print("Experiment 1: Predicting Global Sales from NA Sales")
    experiment_predict_global_sales()
    
    print("\nExperiment 2: Analyzing Global Sales by NA, EU, JP, and Other Sales")
    experiment_sales_by_year()
