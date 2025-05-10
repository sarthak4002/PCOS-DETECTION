import os
import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize  # For multi-class ROC

# 1. Extract the zip file
zip_file_path = 'C:\\Users\\sarth\\Downloads\\archive (4).zip'
extracted_path = "C:\\Users\\sarth\\Downloads\\archive (4)"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)

# 2. Load and preprocess images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load images as grayscale
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to (128, 128) or any size suitable
            images.append(img.flatten())  # Flatten image to a 1D array
            labels.append(label)
    return images, labels

# Paths to your PCOS and normal images
pcos_images_path = "C:\\Users\\sarth\\Downloads\\archive (4)\\dataset\\pcos"
normal_images_path = "C:\\Users\\sarth\\Downloads\\archive (4)\\dataset\\normal"

# 3. Load and label the dataset
pcos_images, pcos_labels = load_images_from_folder(pcos_images_path, 'PCOS')
normal_images, normal_labels = load_images_from_folder(normal_images_path, 'Normal')

# Combine the data
X = np.array(pcos_images + normal_images)
y = np.array(pcos_labels + normal_labels)

# Encode labels (PCOS=1, Normal=0)
le = LabelEncoder()
y = le.fit_transform(y)

# Binarize labels for ROC curve
y_bin = label_binarize(y, classes=[0, 1])

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Function to plot ROC curve
def plot_roc_curve(y_test, y_score, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# 5. Apply Random Forest, SVM, and KNN
# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
y_score_rf = rf_clf.predict_proba(X_test)[:, 1]  # Probability estimates for ROC

print("Random Forest Accuracy: ", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix (Random Forest): \n", confusion_matrix(y_test, y_pred_rf))

# Plot ROC curve for Random Forest
plot_roc_curve(y_test, y_score_rf, "Random Forest")

# SVM
svm_clf = SVC(kernel='linear', random_state=42, probability=True)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
y_score_svm = svm_clf.predict_proba(X_test)[:, 1]  # Probability estimates for ROC

print("SVM Accuracy: ", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
print("Confusion Matrix (SVM): \n", confusion_matrix(y_test, y_pred_svm))

# Plot ROC curve for SVM
plot_roc_curve(y_test, y_score_svm, "SVM")

# KNN
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_test)
y_score_knn = knn_clf.predict_proba(X_test)[:, 1]  # Probability estimates for ROC

print("KNN Accuracy: ", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
print("Confusion Matrix (KNN): \n", confusion_matrix(y_test, y_pred_knn))

# Plot ROC curve for KNN
plot_roc_curve(y_test, y_score_knn, "KNN")

# Finalize and show ROC curves
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()
