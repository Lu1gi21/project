import pandas as pd
import numpy as np
import sys
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Add module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Dylan.Decision_tree.Decision_tree import DecisionTreeClassifier
from Data.data import load_clean_vgsales

def label_encode(series):
    unique_vals = sorted(series.unique())
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    encoded = series.map(mapping)
    return encoded, mapping

def manual_train_test_split(X, y, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_size = int(len(X) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def simple_classification_report(y_true, y_pred, class_names):
    metrics = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    recall_values = []

    for actual, predicted in zip(y_true, y_pred):
        if actual == predicted:
            metrics[actual]["TP"] += 1
        else:
            metrics[actual]["FN"] += 1
            metrics[predicted]["FP"] += 1

    print("{:<12} {:<12} {:<12} {:<12}".format("Class", "Precision", "Recall", "F1-score"))

    for class_id, class_name in enumerate(class_names):
        TP = metrics[class_id]["TP"]
        FP = metrics[class_id]["FP"]
        FN = metrics[class_id]["FN"]

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        recall_values.append(recall)

        print("{:<12} {:<12.2f} {:<12.2f} {:<12.2f}".format(class_name, precision, recall, f1))

    balanced_accuracy = np.mean(recall_values)
    print("\nBalanced Accuracy: {:.4f}".format(balanced_accuracy))
    return balanced_accuracy

def plot_normalized_confusion_matrix(y_true, y_pred, class_names):
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))

    for i in range(num_classes):
        for j in range(num_classes):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black'))
            ax.text(j + 0.5, i + 0.5, f"{cm_percent[i, j]:.1f}%",
                    va='center', ha='center', fontsize=11)

    ax.set_xticks(np.arange(num_classes) + 0.5)
    ax.set_yticks(np.arange(num_classes) + 0.5)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label (TP / FP)")
    ax.set_ylabel("True Label (TP / FN)")
    ax.set_title("Normalized Confusion Matrix (%) — Row = Actual → TP & FN")
    ax.set_xlim(0, num_classes)
    ax.set_ylim(0, num_classes)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

# Load and prepare data once
df = load_clean_vgsales().dropna()

def get_dominant_region(row):
    sales = {'NA': row['NA_Sales'], 'EU': row['EU_Sales'], 'JP': row['JP_Sales']}
    return max(sales, key=sales.get)

df['Dominant_Region'] = df.apply(get_dominant_region, axis=1)

df['Publisher_encoded'], _ = label_encode(df['Publisher'])
df['Platform_encoded'], _ = label_encode(df['Platform'])
df['Genre_encoded'], _ = label_encode(df['Genre'])
df['Region_encoded'], region_map = label_encode(df['Dominant_Region'])

feature_cols = ['Publisher_encoded', 'Platform_encoded', 'Genre_encoded']
X = df[feature_cols].to_numpy()
y = df['Region_encoded'].to_numpy()
class_names = list(region_map.keys())

X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_ratio=0.2)

# Depths to test
depth_values = [1, 5, 10, 25, 50]
results = []

for depth in depth_values:
    print(f"\n--- Tree Depth: {depth} ---")
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}")
    balanced_acc = simple_classification_report(y_test, y_pred, class_names)
    results.append((depth, accuracy, balanced_acc))

# Optional: visualize confusion matrix of last run
plot_normalized_confusion_matrix(y_test, y_pred, class_names)

# Optional: print summary table
region_counts = df['Dominant_Region'].value_counts(normalize=True) * 100
print("Dominant Region Distribution (%):")
print(region_counts.round(2).to_string())
print("\nSummary Table:")
print("{:<8} {:<10} {:<10}".format("Depth", "Accuracy", "Balanced Acc"))
for depth, acc, bacc in results:
    print(f"{depth:<8} {acc:<10.4f} {bacc:<10.4f}")
