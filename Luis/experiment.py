import numpy as np
import sys
import os

# Add project directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import models and data utilities
from Luis.SLR.slr import SimpleLinearRegression
from Luis.Gradient_Descent.gradientDescent import GradientDescentRegression
from Data.data import load_clean_vgsales, get_numeric_vgsales_columns

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.

    Args:
        X (np.ndarray): Feature array.
        y (np.ndarray): Target array.
        test_size (float): Proportion of the dataset to include in the test split (between 0 and 1).
        random_state (int, optional): Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test: Split data arrays.
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

    Args:
        X (np.ndarray): Feature array (n, 1).
        y (np.ndarray): Target array (n,).
        model (SimpleLinearRegression): Fitted regression model.

    Returns:
        np.ndarray: Cook's distance for each observation.
    """
    # Add intercept
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.mean(residuals ** 2)
    # Hat matrix diagonal (leverage)
    H = X_b @ np.linalg.inv(X_b.T @ X_b) @ X_b.T
    leverage = np.diag(H)
    # Cook's distance formula
    cooks_d = (residuals ** 2) / (2 * mse) * (leverage / (1 - leverage) ** 2)
    return cooks_d

def experiment_predict_global_sales():
    """
    Experiment to predict Global_Sales based on NA_Sales using both regression models.
    """
    # Load and prepare data
    df = load_clean_vgsales()
    df_numeric = get_numeric_vgsales_columns(df)
    
    # Prepare features and target
    X = df_numeric['NA_Sales'].values.reshape(-1, 1)
    y = df_numeric['Global_Sales'].values
    
    # Fit model to all data to compute Cook's distance
    slr_model_all = SimpleLinearRegression()
    slr_model_all.fit(X, y)
    cooks_d = cooks_distance(X, y, slr_model_all)

    # Remove outliers using Cook's distance
    threshold = 4 / len(X)
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
    
    # Train Gradient Descent model
    gd_model = GradientDescentRegression(learning_rate=0.01, n_iterations=1000)
    gd_model.fit(X_train, y_train)
    gd_score = gd_model.score(X_test, y_test)
    gd_predictions = gd_model.predict(X_test)
    
    print(f"SLR R² score: {slr_score:.4f}")
    print(f"GD R² score: {gd_score:.4f}")
    print(f"SLR Coefficients: {slr_model.weights}")
    print(f"GD Coefficients: {gd_model.weights}")


def experiment_sales_by_year():
    """
    Experiment to analyze the trend of Global_Sales over Year.
    """
    # Load and prepare data
    df = load_clean_vgsales()
    df_numeric = get_numeric_vgsales_columns(df)
    
    # Group by year and get average global sales
    yearly_sales = df_numeric.groupby('Year')['Global_Sales'].mean().reset_index()
    
    # Also get count of games per year for bubble size
    yearly_count = df_numeric.groupby('Year').size().reset_index(name='Count')
    yearly_data = yearly_sales.merge(yearly_count, on='Year')
    
    # Prepare features and target
    X = yearly_data['Year'].values.reshape(-1, 1)
    y = yearly_data['Global_Sales'].values
    
    # Fit model to all data to compute Cook's distance
    slr_model_all = SimpleLinearRegression()
    slr_model_all.fit(X, y)
    cooks_d = cooks_distance(X, y, slr_model_all)

    # Remove outliers using Cook's distance
    threshold = 4 / len(X)
    mask = cooks_d < threshold
    X = X[mask]
    y = y[mask]
    yearly_data = yearly_data[mask]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    yearly_data_train = yearly_data.iloc[X_train.flatten().argsort()]
    yearly_data_test = yearly_data.iloc[X_test.flatten().argsort()]

    # Train model
    model = SimpleLinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    predictions = model.predict(X_test)

    # Find peak sales year
    peak_idx = np.argmax(y_test)
    peak_year = X_test[peak_idx][0]
    peak_sales = y_test[peak_idx]

    intercept, slope = model.weights
    
    if slope > 0:
        trend_direction = f"Sales increasing by {slope:.4f}M per year"
    else:
        trend_direction = f"Sales decreasing by {abs(slope):.4f}M per year"

    print(f"R² score for predicting Global_Sales trend by Year: {score:.4f}")
    print(f"Coefficients: {model.weights}")
    print(f"Peak average sales year: {peak_year:.0f}, with {peak_sales:.4f}M average sales")

if __name__ == "__main__":
    print("Experiment 1: Predicting Global Sales from NA Sales")
    experiment_predict_global_sales()
    
    print("\nExperiment 2: Analyzing Sales Trends by Year")
    experiment_sales_by_year()
