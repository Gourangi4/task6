# Task 6: K-Nearest Neighbors (KNN) Classification

## Objective
Understand and implement KNN for classification problems.

## Dataset
Iris dataset (Iris.csv) — 150 samples, 4 features, 3 classes (setosa, versicolor, virginica).

## Steps Performed
1. Loaded the dataset and dropped the Id column.
2. Encoded species labels and normalized features using StandardScaler.
3. Split data into train and test sets (80/20).
4. Trained KNeighborsClassifier for K values from 1 to 20 and compared accuracy.
5. Selected the best K based on test accuracy.
6. Evaluated the final model using accuracy and a confusion matrix.
7. Visualized the decision boundary using the first two features.

## Results
- Best K: 1
- Final Accuracy: 96.67%

## Files
- `knn_classification.py` — main script
- `Iris.csv` — dataset
- `accuracy_vs_k.png` — accuracy vs K plot
- `confusion_matrix.png` — confusion matrix plot
- `decision_boundary.png` — decision boundary plot

## Tools Used
Python, Pandas, Scikit-learn, Matplotlib

## What I Learned
Instance-based learning, Euclidean distance, and how to choose the right K value.
