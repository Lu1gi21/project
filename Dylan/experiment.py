import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the KNN model and data utilities
from Dylan.kNN.kNN import KNearestNeighbors
from Data.data import load_clean_vgsales, encode_categorical_column, get_numeric_vgsales_columns

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def compute_balanced_accuracy(y_true, y_pred):
    unique_classes = np.unique(y_true)
    recalls = []

    for cls in unique_classes:
        # True positives: correctly predicted as this class
        TP = np.sum((y_pred == cls) & (y_true == cls))
        # False negatives: actually this class but predicted as something else
        FN = np.sum((y_pred != cls) & (y_true == cls))

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        recalls.append(recall)

    balanced_acc = np.mean(recalls)
    return balanced_acc

def normalize_columns(df, columns):
    """
    Normalize given numeric columns using z-score standardization.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        columns (list): List of column names to normalize.
    
    Returns:
        pandas.DataFrame: DataFrame with normalized columns.
    """
    df = df.copy()
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col + '_norm'] = (df[col] - mean) / std
    return df

def train_test_split(X, y, test_ratio=0.2, seed=42):
    """
    Split features and target into training and testing sets.

    Args:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.
        test_ratio (float): Proportion of data to reserve for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_size = int(len(X) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def experiment_knn_nintendo_genre_prediction(k = 3):
    """
    Use kNN classification to predict Nintendo's most likely genre based on sales and metadata.
    """
    df = load_clean_vgsales()

    #df = df[df['Publisher'] == 'Nintendo']
    
    df = encode_categorical_column(df, 'Publisher', method='label')

    # Encode Genre as label
    df = encode_categorical_column(df, 'Genre', method='label')
    genre_mapping = df[['Genre', 'Genre_encoded']].drop_duplicates().set_index('Genre_encoded')['Genre'].to_dict()

    # Encode Platform and normalize Year
    df = encode_categorical_column(df, 'Platform', method='label')
    #df = normalize_columns(df, ['Year'])
    df = normalize_columns(df, ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])


    # Prepare features and labels
    feature_cols = ['Publisher_encoded', 'Year_norm', 'NA_Sales_norm', 'EU_Sales_norm', 'JP_Sales_norm', 'Other_Sales_norm']
    X = df[feature_cols].values
    y = df['Genre_encoded'].values
    df = df.reset_index(drop=True)  # ensure alignment
    all_indices = np.arange(len(df))


    X_train, X_test, y_train, y_test = train_test_split(X, y)

    test_size = len(X_test)
    test_indices = all_indices[-test_size:]

    model = KNearestNeighbors(k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    print(f"K value: {k}")
    print(f"Total training samples: {len(X_train)}")
    print(f"Total test samples (predictions made): {len(X_test)}")
    print(df['Genre'].value_counts())


    print(f"\nAccuracy: {accuracy:.4f}")
    balanced_acc = compute_balanced_accuracy(y_test, y_pred)
    print(f"Balanced Accuracy (manual): {balanced_acc:.4f}")

    # # Visualization
    # plt.figure(figsize=(10, 7))

    # x_feature = 'Year_norm'  # Instead of 'Year_norm'
    # y_feature = 'NA_Sales_norm'
    
    # correct = y_pred == y_test

    # # Track which genre labels have been used in legend
    # used_labels = set()
    # x_idx = feature_cols.index(x_feature)
    # y_idx = feature_cols.index(y_feature)

    # # Test set predictions
    # x_test_vals = df.iloc[test_indices]['Year'].values  # raw year  
    # y_test_vals = X_test[:, y_idx]
    # correct = y_pred == y_test
    # used_labels = set()

    # for i in range(len(x_test_vals)):
    #     genre_label = genre_mapping[y_pred[i]]
    #     color = sns.color_palette("tab10")[y_pred[i] % 10]
    #     #edge = 'black' if not correct[i] else 'none'
    #     edge = 'black'
    #     label = genre_label if genre_label not in used_labels else None

    #     plt.scatter(x_test_vals[i], y_test_vals[i],
    #                 color=color,
    #                 edgecolor=edge,
    #                 label=label,
    #                 alpha=1 if correct[i] else 0.8,
    #                 s=100 if correct[i] else 50,
    #                 marker='o' if correct[i] else 'x')  # circle for test

    #     if label:
    #         used_labels.add(genre_label)


    # # Labels and legend
    # plt.xlabel("Year")
    # #plt.ylim(-0.5, 20)
    # plt.ylabel("NA Sales (millions)")
    # plt.title("kNN Classification\n Accurate Predictions (Circles), Inaccurate Predictions (Xs)")
    # plt.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.show()




if __name__ == "__main__":
    #print("kNN Classification Experiment: High Seller Nintendo Racing Games")
    for k in [1, 3, 5, 10, 25, 50, 100]:
        print(f"\n--- k = {k} ---")
        experiment_knn_nintendo_genre_prediction(k=k)
    #experiment_knn_nintendo_genre_prediction(k=5)