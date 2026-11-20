# ============================================================
# Project 2: Data Classification Using AI
# DecodeLabs - Industrial Training Kit (Batch 2026)
# Model used: IPO (Input -> Process -> Output)
# Dataset: Iris (150 samples, 3 classes, 4 features)
# Algorithm: K-Nearest Neighbors (KNN)
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, f1_score,
                             ConfusionMatrixDisplay)

# ============================================================
# STEP 1: INPUT - LOAD AND UNDERSTAND THE DATASET
# ============================================================
print("=" * 60)
print("STEP 1: LOAD AND UNDERSTAND THE DATASET")
print("=" * 60)

iris = load_iris()
X = iris.data      # features (4 measurements of each flower)
y = iris.target    # labels (0, 1, 2)

df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = [iris.target_names[i] for i in y]

print("Samples (rows)     :", X.shape[0])
print("Features (columns) :", X.shape[1])
print("Feature names      :", iris.feature_names)
print("Classes            :", iris.target_names.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nSamples in each class (balanced data):")
print(df["species"].value_counts())
print("\nMissing values:", int(df.isnull().sum().sum()))
print("\nBasic statistics:")
print(df.describe().round(2))

# ============================================================
# STEP 2: SPLIT DATA INTO TRAINING AND TESTING SETS (80% / 20%)
# shuffle=True removes order bias, stratify keeps classes balanced
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: TRAIN-TEST SPLIT (80% / 20%)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True, stratify=y, random_state=42
)
print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# ============================================================
# STEP 3: FEATURE SCALING (StandardScaler: mean = 0, variance = 1)
# Fit ONLY on training data, then apply to test data.
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Mean after scaling (train):", np.round(X_train_scaled.mean(axis=0), 2))
print("Std  after scaling (train):", np.round(X_train_scaled.std(axis=0), 2))

# ============================================================
# STEP 4: TUNE THE ENGINE - CHOOSING THE BEST "K" (elbow method)
# We use cross-validation on the TRAINING set only, so the test set
# stays locked and untouched for the final validation.
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: CHOOSING THE BEST K")
print("=" * 60)

k_values = range(1, 26)
error_rates = []
for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    cv_accuracy = cross_val_score(knn_temp, X_train_scaled, y_train, cv=5).mean()
    error_rates.append(1 - cv_accuracy)

best_k = list(k_values)[int(np.argmin(error_rates))]
print("Lowest cross-validation error: %.3f at K = %d" % (min(error_rates), best_k))

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), error_rates, marker="o")
plt.axvline(best_k, color="red", linestyle="--", label="Best K = %d" % best_k)
plt.title("Choosing K (Error Rate vs K Value)")
plt.xlabel("K Value")
plt.ylabel("Error Rate (5-fold cross-validation)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("k_value_plot.png", dpi=120, bbox_inches="tight")

# ============================================================
# STEP 5: PROCESS - TRAIN THE KNN MODEL
# Workflow: instantiate -> fit -> predict
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: TRAIN THE KNN MODEL (K = %d)" % best_k)
print("=" * 60)

model = KNeighborsClassifier(n_neighbors=best_k)    # INSTANTIATE (build the frame)
model.fit(X_train_scaled, y_train)             # FIT (memorize the map)
predictions = model.predict(X_test_scaled)     # PREDICT (apply logic)
print("Model trained and predictions made on the test set.")

# ============================================================
# STEP 6: OUTPUT - VALIDATE THE MODEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: OUTPUT VALIDATION")
print("=" * 60)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100))

cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix (rows = real, columns = predicted):")
print(cm)

print("\nClassification Report (Precision, Recall, F1 Score):")
print(classification_report(y_test, predictions,
                            target_names=iris.target_names))
print("Macro F1 Score: %.3f" % f1_score(y_test, predictions, average="macro"))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=iris.target_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - KNN (K = %d)" % best_k)
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")

# ============================================================
# STEP 7: COMPARE DIFFERENT ALGORITHMS (idea from the Conclusion)
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: COMPARING ALGORITHMS")
print("=" * 60)

models = {
    "KNN (K=%d)" % best_k: KNeighborsClassifier(n_neighbors=best_k),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
}
for name, clf in models.items():
    clf.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    print("%-20s Accuracy: %.2f%%" % (name, acc * 100))

# ============================================================
# STEP 8: TEST THE MODEL WITH COMPLETELY NEW DATA
# Values are: sepal length, sepal width, petal length, petal width (cm)
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: PREDICT NEW FLOWERS")
print("=" * 60)

new_flowers = np.array([
    [5.0, 3.5, 1.4, 0.2],   # small petals -> should be setosa
    [6.0, 2.9, 4.5, 1.5],   # medium petals -> should be versicolor
    [7.0, 3.1, 6.0, 2.1],   # large petals -> should be virginica
])
new_scaled = scaler.transform(new_flowers)   # use the SAME scaler
new_predictions = model.predict(new_scaled)

for flower, pred in zip(new_flowers, new_predictions):
    print("Measurements %s -> Predicted: %s" % (flower, iris.target_names[pred]))

print("\nDone! Plots saved: k_value_plot.png and confusion_matrix.png")
plt.show()
