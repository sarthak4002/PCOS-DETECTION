
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:\\Users\\sarth\\OneDrive\\Desktop\\pcosdataset1.csv")

# Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split data into features and labels
X = df.drop(columns=['Label'])
y = df['Label']

# Introduce synthetic features (polynomial features)
X['BMI_Height'] = X['BMI'] * X['Height']
X['Weight_PulseRate'] = X['Weight'] * X['PulseRate']

# Introduce noise
np.random.seed(42)
noise = np.random.normal(0, 0.5, size=X.shape)  # Increased noise level
X_noisy = X + noise

# Introduce outliers
n_outliers = int(0.05 * X.shape[0])  # 5% outliers
outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, X.shape[1]))
X_noisy_with_outliers = np.vstack((X_noisy, outliers))
y_with_outliers = np.concatenate((y, np.random.choice([0, 1], size=n_outliers)))  # Random labels for outliers

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_noisy_with_outliers)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_with_outliers, test_size=0.2, random_state=42, stratify=y_with_outliers)

# Initialize KNN classifier
knn = KNeighborsClassifier()

# Hyperparameter tuning using Grid Search
param_grid = {'n_neighbors': range(1, 21), 'metric': ['euclidean', 'manhattan', 'minkowski']}
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best estimator from Grid Search
best_knn = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Perform k-fold cross-validation
kf = StratifiedKFold(n_splits=10)
cv_scores = cross_val_score(best_knn, X_scaled, y_with_outliers, cv=kf)
print("10-fold Cross-validation scores: ", cv_scores)
print(f"Average cross-validation score: {np.mean(cv_scores) * 100:.2f}%")

# Fit the KNN model on training data
best_knn.fit(X_train, y_train)

# Test the model on testing data
y_pred = best_knn.predict(X_test)

# Evaluate the KNN model on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Test Accuracy: {test_accuracy * 100:.2f}%")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate additional performance metrics
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'PCOS'], yticklabels=['Normal', 'PCOS'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC-AUC score
y_prob = best_knn.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.2f}")
