################################################

import os
import time
import warnings
import pickle
import glob

# Завантаження даних
with open('shaped.pickle', 'rb') as f:
    ab = pickle.load(f)
print("Shape of ab:", ab.shape)

###############################################
# Імпорт  бібліотек
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, islice
import scipy

from sklearn import decomposition, model_selection, metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

# аааааугментація даних
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model

###############################################
# 1. редукція розмірності за допомогою TruncatedSVD
start_time = time.time()
params = {'n_clusters': 4}
pca = decomposition.TruncatedSVD(n_components=10)
X_transformed = pca.fit_transform(ab)
end_time = time.time()
print("Time for SVD transformation: {:.2f} ms".format((end_time - start_time) * 1000))

###############################################
# 2. отримання 2D-простору за допомогою TSNE (для кластеризації та візуалізації)
start_time = time.time()
X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_transformed)
end_time = time.time()
print("Time for setting X_embedded (TSNE): {:.2f} ms".format((end_time - start_time) * 1000))

###############################################
# 3. кластеризація за допомогою KMeans (на TSNE-даних)
t0 = time.time()
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    kmeans = KMeans(n_clusters=params['n_clusters'], random_state=42)
    kmeans.fit(X_embedded)
    y_pred = kmeans.labels_.astype(int)
t1 = time.time()
print("Time for clustering: {:.2f} ms".format((t1 - t0) * 1000))
print("Shape of y_pred:", y_pred.shape)

# покращена візуалізація кластерів
plt.figure(figsize=(7, 6))
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(y_pred) + 1))))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=30, c=colors[y_pred], alpha=0.8)
plt.title("Cluster Visualization (KMeans)", fontsize=14)
plt.xlabel("Component 1", fontsize=12)
plt.ylabel("Component 2", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

###############################################
# 4. побудова автоенкодера
def create_dense_ae():
    hidden_dim = 60
    encoding_dim = 2
    # Кодувальник: вход має розмірність 10 (X_transformed)
    input_layer = Input(shape=(10,))
    flat = Flatten()(input_layer)
    hidden = Dense(hidden_dim, activation='relu')(flat)
    hidden2 = Dense(hidden_dim, activation='relu')(hidden)
    encoded = Dense(encoding_dim, activation='relu')(hidden2)
    # Декодувальник
    input_encoded = Input(shape=(encoding_dim,))
    hidden_encoded = Dense(hidden_dim, activation='sigmoid')(input_encoded)
    hidden_encoded2 = Dense(hidden_dim, activation='sigmoid')(hidden_encoded)
    flat_decoded = Dense(10, activation='sigmoid')(hidden_encoded2)
    decoded = Reshape((10,))(flat_decoded)
    encoder = Model(input_layer, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_layer, decoder(encoder(input_layer)), name="autoencoder")
    return encoder, decoder, autoencoder

###############################################
# 5. Перевірка фізичних пристроїв
print("Physical devices:", tf.config.list_physical_devices())

###############################################
# 6. Створення та компіляція автоенкодера (використовуємо GPU)
with tf.device("/device:GPU:0"):
    encoder, decoder, autoencoder = create_dense_ae()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
    autoencoder.summary()

###############################################
# 7. Навчання автоенкодера
# Використовуємо X_transformed (розмірність 10), що відповідає вхідним даним автоенкодера
X_train_ae, X_valid_ae, _, _ = model_selection.train_test_split(X_transformed, y_pred, test_size=0.3, random_state=42)
with tf.device("/device:GPU:0"):
    start_time = time.time()
    autoencoder.fit(X_train_ae, X_train_ae,
                    epochs=500,
                    batch_size=50,
                    shuffle=True,
                    validation_data=(X_valid_ae, X_valid_ae))
    end_time = time.time()
    print("Training time for autoencoder: {:.2f} ms".format((end_time - start_time) * 1000))

###############################################
# 8. Візуалізація 2D-латентного простору автоенкодера (за допомогою encoder)
colors_latent = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                            '#f781bf', '#a65628', '#984ea3',
                                            '#999999', '#e41a1c', '#dede00']),
                                     int(max(y_pred) + 1))))
start_time = time.time()
x_train_encoded = encoder.predict(X_train_ae, batch_size=50)
plt.figure(figsize=(7, 6))
plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1], c='blue', label='Train', alpha=0.6)
plt.title("2D Latent Space (Train Data)", fontsize=14)
plt.xlabel("Latent Dimension 1", fontsize=12)
plt.ylabel("Latent Dimension 2", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

x_valid_encoded = encoder.predict(X_valid_ae, batch_size=50)
plt.figure(figsize=(7, 6))
plt.scatter(x_valid_encoded[:, 0], x_valid_encoded[:, 1], c='red', label='Validation', alpha=0.6)
plt.title("2D Latent Space (Validation Data)", fontsize=14)
plt.xlabel("Latent Dimension 1", fontsize=12)
plt.ylabel("Latent Dimension 2", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
end_time = time.time()
print("Time for encoding visualization: {:.2f} ms".format((end_time - start_time) * 1000))

###############################################
# 9. Навчання класифікатора та порівняння ефективності з аугментацією

def evaluate_classifier(X_train, y_train, X_valid, y_valid, clf, clf_name="Classifier"):
    start = time.time()
    clf.fit(X_train, y_train)
    y_pred_clf = clf.predict(X_valid)
    acc = metrics.accuracy_score(y_valid, y_pred_clf)
    elapsed = time.time() - start
    print(f"{clf_name} accuracy: {acc:.4f} (Elapsed time: {elapsed*1000:.2f} ms)")
    return acc

# Розбиття даних для класифікації (використовуємо X_embedded та y_pred)
X_train_cls, X_valid_cls, y_train_cls, y_valid_cls = model_selection.train_test_split(X_embedded, y_pred, test_size=0.3, random_state=42)

# Базовий класифікатор – Logistic Regression
clf_baseline = LogisticRegression(max_iter=1000, random_state=42)
acc_baseline = evaluate_classifier(X_train_cls, y_train_cls, X_valid_cls, y_valid_cls, clf_baseline, "Baseline Logistic Regression")
results = {"Baseline": acc_baseline}

def augment_and_evaluate(aug, aug_name):
    X_train_aug, y_train_aug = aug.fit_resample(X_train_cls, y_train_cls)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    acc = evaluate_classifier(X_train_aug, y_train_aug, X_valid_cls, y_valid_cls, clf, aug_name)
    return acc

ros = RandomOverSampler(random_state=42)
results["RandomOverSampler"] = augment_and_evaluate(ros, "RandomOverSampler")

smote = SMOTE(random_state=42)
results["SMOTE"] = augment_and_evaluate(smote, "SMOTE")

adasyn = ADASYN(random_state=42)
results["ADASYN"] = augment_and_evaluate(adasyn, "ADASYN")

###############################################
# 10. Візуалізація результатів класифікатора
methods = list(results.keys())
accuracies = [results[m] for m in methods]

plt.figure(figsize=(8, 6))
bars = plt.bar(methods, accuracies, color=['gray', 'skyblue', 'lightgreen', 'salmon'])
plt.ylim(0, 1)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Classifier Performance Comparison", fontsize=14)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{acc:.4f}", ha='center', fontsize=12)
plt.tight_layout()
plt.show()

###############################################

