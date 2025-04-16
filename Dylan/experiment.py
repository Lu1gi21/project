import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the KNN model and data utilities
from Dylan.kNN.kNN import KNearestNeighbors
from Data.data import load_clean_vgsales, get_numeric_vgsales_columns

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def experiment_knn_global_sales():
    """
    Experiment to predict Global_Sales based on NA_Sales using k-Nearest Neighbors regression.
    """
    # Load and prepare data
    df = load_clean_vgsales()
    df_numeric = get_numeric_vgsales_columns(df)

    # Prepare features and target
    X = df_numeric['NA_Sales'].values.reshape(-1, 1)
    y = df_numeric['Global_Sales'].values

    # Train kNN Regression model
    knn_model = KNearestNeighbors(k=5)
    knn_model.fit(X, y)
    knn_score = knn_model.score(X, y)
    knn_predictions = knn_model.predict(X)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(X, y, alpha=0.5, color='#3498db', label='Actual data')
    plt.scatter(X, knn_predictions, color='#e67e22', label='kNN Predictions (k=5)', s=10)
    
    plt.title('k-Nearest Neighbors Regression:\nPredicting Global Sales from NA Sales', fontsize=16, fontweight='bold')
    plt.xlabel('North America Sales (millions)', fontsize=14)
    plt.ylabel('Global Sales (millions)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Display model info
    plt.annotate(f"R² = {knn_score:.4f}\nk = 5", xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6),
                 fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

    print(f"R² score for kNN (k=5): {knn_score:.4f}")

if __name__ == "__main__":
    print("kNN Regression Experiment: Predicting Global Sales from NA Sales")
    experiment_knn_global_sales()
