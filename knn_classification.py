import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

df = pd.read_csv("Iris.csv")
df = df.drop(columns=["Id"])

feature_names = df.columns[:-1].tolist()
le = LabelEncoder()
y = le.fit_transform(df["Species"])
X = df[feature_names].values
target_names = le.classes_

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K={k}: Accuracy={acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), accuracies, marker='o')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.xticks(list(k_values))
plt.grid(True)
plt.savefig("accuracy_vs_k.png")
plt.close()

best_k = list(k_values)[int(np.argmax(accuracies))]
print(f"\nBest K: {best_k}")

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)
y_pred_final = final_knn.predict(X_test)

final_acc = accuracy_score(y_test, y_pred_final)
cm = confusion_matrix(y_test, y_pred_final)

print(f"\nFinal Accuracy (K={best_k}): {final_acc:.4f}")
print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap='Blues')
plt.title(f"Confusion Matrix (K={best_k})")
plt.savefig("confusion_matrix.png")
plt.close()

X_vis = X_scaled[:, :2]
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y, test_size=0.2, random_state=42, stratify=y
)

vis_knn = KNeighborsClassifier(n_neighbors=best_k)
vis_knn.fit(X_train_vis, y_train_vis)

x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = vis_knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=40)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title(f"KNN Decision Boundary (K={best_k})")
plt.legend(handles=scatter.legend_elements()[0], labels=list(target_names))
plt.savefig("decision_boundary.png")
plt.close()

print("\nSaved plots: accuracy_vs_k.png, confusion_matrix.png, decision_boundary.png")
