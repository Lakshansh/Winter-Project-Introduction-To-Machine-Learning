import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Load the dataset
data_path = r"D:\Winter Projects 3rd sem\Machine Learning\Assignment3_Dataset\Updated_Data-Melbourne_F_fixed.csv"
data = pd.read_csv(data_path)

# Data Preprocessing
# Drop unnecessary columns if present
columns_to_drop = ['Year', 'Month', 'Day']
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors='ignore')

# Handle missing values
missing_summary = data.isnull().sum()
print("Missing Values Summary:")
print(missing_summary[missing_summary > 0])
data = data.fillna(data.mean())  # Fill missing values with column means

# Visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# Feature and Target Selection
# Assuming 'Average Outflow' is the target variable
data['Target'] = (data['Average Outflow'] > data['Average Outflow'].mean()).astype(int)
X = data.drop(['Serial No', 'Average Outflow', 'Target'], axis=1)
y = data['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression (From Scratch)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lr=0.01, iterations=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    
    for i in range(iterations):
        linear_model = np.dot(X, weights) + bias
        predictions = sigmoid(linear_model)
        
        dw = (1 / m) * np.dot(X.T, (predictions - y))
        db = (1 / m) * np.sum(predictions - y)
        
        weights -= lr * dw
        bias -= lr * db
        
    return weights, bias

weights, bias = logistic_regression(X_train_scaled, y_train.values)

def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    predictions = sigmoid(linear_model)
    return [1 if p > 0.5 else 0 for p in predictions]

y_pred_custom = predict(X_test_scaled, weights, bias)

# Logistic Regression (sklearn)
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred_logreg = log_reg.predict(X_test_scaled)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Random Forest Classifier with Hyperparameter Tuning
rf_params = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='f1')
rf.fit(X_train, y_train)
print("Best Random Forest Parameters:", rf.best_params_)
y_pred_rf = rf.best_estimator_.predict(X_test)

# Support Vector Machine with Hyperparameter Tuning
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm = GridSearchCV(SVC(), svm_params, cv=3, scoring='f1')
svm.fit(X_train_scaled, y_train)
print("Best SVM Parameters:", svm.best_params_)
y_pred_svm = svm.best_estimator_.predict(X_test_scaled)

# KNN from Scratch
def knn_from_scratch(X_train, y_train, X_test, k=5):
    distances = cdist(X_test, X_train, 'euclidean')
    neighbors = np.argsort(distances, axis=1)[:, :k]
    
    y_pred = []
    for i in range(neighbors.shape[0]):
        labels = y_train.iloc[neighbors[i]].values
        y_pred.append(np.bincount(labels).argmax())
    
    return np.array(y_pred)

# Use KNN from scratch
y_pred_knn_scratch = knn_from_scratch(X_train_scaled, y_train, X_test_scaled, k=5)

# Support Vector Machine from Scratch (Linear SVM)
class LinearSVM:
    def __init__(self, lr=0.01, lambda_param=0.01, epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        y = np.where(y == 0, -1, 1)

        for _ in range(self.epochs):
            for i in range(m):
                if y[i] * (np.dot(X[i], self.weights) + self.bias) < 1:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.lr * (-y[i])
                else:
                    self.weights -= self.lr * 2 * self.lambda_param * self.weights

    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias >= 0, 1, 0)

# Fit and predict using custom LinearSVM
svm_scratch = LinearSVM(lr=0.01, lambda_param=0.01, epochs=1000)
svm_scratch.fit(X_train_scaled, y_train.values)
y_pred_svm_scratch = svm_scratch.predict(X_test_scaled)

# Model Evaluation and Confusion Matrix Visualization
def evaluate_model(y_test, y_pred, model_name):
    print(f"\nModel: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()

# Evaluate all models
evaluate_model(y_test, y_pred_custom, "Logistic Regression (Scratch)")
evaluate_model(y_test, y_pred_logreg, "Logistic Regression (sklearn)")
evaluate_model(y_test, y_pred_knn, "K-Nearest Neighbors (sklearn)")
evaluate_model(y_test, y_pred_dt, "Decision Tree")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_svm, "Support Vector Machine (sklearn)")
evaluate_model(y_test, y_pred_knn_scratch, "K-Nearest Neighbors (Scratch)")
evaluate_model(y_test, y_pred_svm_scratch, "Support Vector Machine (Scratch)")

# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_train, palette='viridis', s=100, edgecolor='k')
plt.title('t-SNE Visualization')
plt.show()

# PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette='viridis', s=100, edgecolor='k')
plt.title('PCA Visualization')
plt.show()
