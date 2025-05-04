import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the KNN model and data utilities
from Dylan.kNN.kNN import KNearestNeighbors
from Data.data import load_clean_vgsales, encode_categorical_column

# PCA for visualization
from sklearn.decomposition import PCA

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def compute_balanced_accuracy(y_true, y_pred):
    unique_classes = np.unique(y_true)
    recalls = []
    for cls in unique_classes:
        TP = np.sum((y_pred == cls) & (y_true == cls))
        FN = np.sum((y_pred != cls) & (y_true == cls))
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        recalls.append(recall)
    return np.mean(recalls)

def normalize_columns(df, columns):
    df = df.copy()
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col + '_norm'] = (df[col] - mean) / std
    return df

def clip_extreme_zscores(df, columns, z_thresh=3.0):
    for col in columns:
        norm_col = col + '_norm'
        df[norm_col] = df[norm_col].clip(lower=-z_thresh, upper=z_thresh)
    return df

def train_test_split(X, y, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_size = int(len(X) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def plot_pca(df, label_column='Genre', title='PCA of Nintendo Game Features by Genre'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='PCA1',
        y='PCA2',
        hue=label_column,
        palette='tab10',
        data=df,
        alpha=0.7,
        edgecolor=None
    )
    plt.title("2D PCA Projection of Games\nPC1 = Publisher, PC2 = Year/Sales (Color = Genre)")
    plt.xlabel("PC1: Strong influence of Publisher (encoded + normalized)")
    plt.ylabel("PC2: Variation by Release Year (minor sales influence)")
    plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def experiment_knn_nintendo_genre_prediction(k=3):
    df = load_clean_vgsales()
    #df = df[df['Publisher'] == 'Nintendo']

    df = encode_categorical_column(df, 'Publisher', method='label')
    df = encode_categorical_column(df, 'Platform', method='label')
    # Do not encode Genre – use string labels
    df = normalize_columns(df, ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])
    df = clip_extreme_zscores(df, ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])
    df = df.dropna()

    feature_cols = ['Publisher_encoded', 'Year_norm', 'NA_Sales_norm', 'EU_Sales_norm', 'JP_Sales_norm', 'Other_Sales_norm']
    X = df[feature_cols].values
    y = df['Genre'].values

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    # Print loadings
    print("\nPCA Loadings (Feature Contributions to PC1 and PC2):")
    for i, component in enumerate(pca.components_[:2]):
        print(f"\nPrincipal Component {i+1} loadings:")
        for feat, val in zip(feature_cols, component):
            print(f"{feat:20s}: {val:+.3f}")

    # Train/test split
    df = df.reset_index(drop=True)
    all_indices = np.arange(len(df))
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    test_size = len(X_test)
    test_indices = all_indices[-test_size:]

    # Train and predict with KNN
    model = KNearestNeighbors(k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    print(f"\nK value: {k}")
    print(f"Total training samples: {len(X_train)}")
    print(f"Total test samples (predictions made): {len(X_test)}")
    print(df['Genre'].value_counts())
    print(f"\nAccuracy: {accuracy:.4f}")
    balanced_acc = compute_balanced_accuracy(y_test, y_pred)
    print(f"Balanced Accuracy (manual): {balanced_acc:.4f}")

    # PCA plot
    # Cleave sample so that the graph doesn't have a bunch of data points.
    df_plot = df.sample(frac=0.05, random_state=42)  
    plot_pca(df_plot, label_column='Genre')

if __name__ == "__main__":
    experiment_knn_nintendo_genre_prediction(k=10)
