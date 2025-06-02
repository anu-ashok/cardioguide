import os
import numpy as np
import cv2
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
normal_dir = "CardioGuide/Cardioguide/dataset/normal"
abnormal_dir = "CardioGuide/Cardioguide/dataset/abnormal"
image_size = (128, 128)

def load_images():
    X, y = [], []
    for label, folder in enumerate([normal_dir, abnormal_dir]):
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, image_size)
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y)

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {title}')
    plt.show()

# CNN Model
def train_cnn(X, y):
    X = X / 255.0
    y_cat = to_categorical(y, 2)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    model.save("models/cnn_model.h5")

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("[CNN] Classification Report:\n", classification_report(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred, "CNN")
    return model, accuracy_score(y_true, y_pred)

# XGBoost Feature Extraction
def extract_features(X):
    features = []
    for img in X:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        mean = np.mean(gray)
        std = np.std(gray)
        max_val = np.max(gray)
        min_val = np.min(gray)
        var = np.var(gray)
        combined = np.concatenate([hist, [mean, std, max_val, min_val, var]])
        features.append(combined)
    return np.array(features)

# XGBoost Model
def train_xgboost(X, y):
    X_feat = extract_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    joblib.dump(model, "models/xgboost_model.json")

    y_pred = model.predict(X_test)
    print("[XGBoost] Classification Report:\n", classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, "XGBoost")
    return model, accuracy_score(y_test, y_pred)

# GNN Feature Extraction (Simplified Dense Approach)
def extract_graph_features(X):
    features = []
    for img in X:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        std = np.std(gray)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
        features.append([mean, std, edge_density])
    return np.array(features)

# GNN Model (Dense MLP style for simplicity)
def train_gnn(X, y):
    X_feat = extract_graph_features(X)
    y_cat = to_categorical(y, 2)
    X_train, X_test, y_train, y_test = train_test_split(X_feat, y_cat, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(16, activation='relu', input_shape=(3,)),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)
    model.save("models/gnn_model.h5")

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("[GNN] Classification Report:\n", classification_report(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred, "GNN")
    return model, accuracy_score(y_true, y_pred)

# Load data
X, y = load_images()

# Train all models
cnn_model, acc_cnn = train_cnn(X, y)
xgb_model, acc_xgb = train_xgboost(X, y)
gnn_model, acc_gnn = train_gnn(X, y)

# Sample Prediction
sample = X[0]

cnn_pred = cnn_model.predict(np.expand_dims(sample / 255.0, axis=0))
xgb_pred = xgb_model.predict(extract_features(np.array([sample])))
gnn_pred = gnn_model.predict(extract_graph_features(np.array([sample])))

print("[CNN] Sample Prediction:", "Normal" if np.argmax(cnn_pred) == 0 else "Abnormal")
print("[XGBoost] Sample Prediction:", "Normal" if xgb_pred[0] == 0 else "Abnormal")
print("[GNN] Sample Prediction:", "Normal" if np.argmax(gnn_pred) == 0 else "Abnormal")

# Accuracy Comparison Plot
models = ['CNN', 'XGBoost', 'GNN']
accuracies = [acc_cnn, acc_xgb, acc_gnn]
plt.figure(figsize=(6, 4))
sns.barplot(x=models, y=accuracies, palette='Set2')
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.show()
