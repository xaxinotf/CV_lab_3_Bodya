################################################
### GPU_old.py – Використання GPU

import os
import time
import warnings
import pickle
import glob

# НЕ вимикаємо GPU – TensorFlow автоматично використовує GPU, якщо він доступний.
# Якщо потрібні специфічні налаштування для GPU, перевір, що драйвери та CUDA/cuDNN встановлені.

# Завантаження даних з shaped.pickle (переконайся, що файл знаходиться у потрібній папці)
with open('shaped.pickle', 'rb') as f:
    ab = pickle.load(f)

# Перевірка форми даних
print("Shape of ab:", ab.shape)

###############################################
# Імпорт необхідних бібліотек
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn import decomposition, ensemble
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model

###############################################
# Допоміжні функції для навчання класифікаторів
def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, valid_y)

def train_model2(classifier, feature_vector_train, label, feature_vector_valid, valid_Y, is_neural_net=False):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, valid_Y)

###############################################
# Підготовка даних та редукція розмірності за допомогою TruncatedSVD
start_time = time.time()
# Базові параметри
default_base = {
    'quantile': 0.3,
    'eps': 0.3,
    'damping': 0.9,
    'preference': -200,
    'n_neighbors': 10,  # можливо більше
    'n_clusters': 4,    # фактична кількість кластерів
    'min_samples': 20,
    'xi': 0.05,
    'min_cluster_size': 0.1
}
params = default_base.copy()

pca = decomposition.TruncatedSVD(n_components=10)
pca.fit(ab)
transformed_ = pca.transform(ab)
end_time = time.time()
print("Time for SVD transformation: {:.2f} ms".format((end_time - start_time) * 1000))

###############################################
# Використовуємо трансформовані дані як X_embedded (TSNE-код закоментовано)
start_time = time.time()
# Якщо бажаєш використовувати TSNE, розкоментуй наступний рядок:
# X_embedded = TSNE(n_components=2).fit_transform(transformed_)
X_embedded = transformed_
end_time = time.time()
print("Time for setting X_embedded: {:.2f} ms".format((end_time - start_time) * 1000))

###############################################
# Кластеризація за допомогою KMeans
two_means = KMeans(n_clusters=params['n_clusters'])
clustering_algorithms = (('KMeans', two_means),)
plot_num = 1

t0 = time.time()
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="the number of connected components of the connectivity matrix is [0-9]{1,2} > 1. Completing it to avoid stopping the tree early.",
        category=UserWarning)
    warnings.filterwarnings(
        "ignore",
        message="Graph is not fully connected, spectral embedding may not work as expected.",
        category=UserWarning)

    for name, algorithm in clustering_algorithms:
        algorithm.fit(X_embedded)
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X_embedded)

        plt.subplot(1, len(clustering_algorithms), plot_num)
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        colors = np.append(colors, ["#000000"])
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=2, color=colors[y_pred])
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1
        break
t1 = time.time()
print("Time for clustering: {:.2f} ms".format((t1 - t0) * 1000))

###############################################
# Візуалізація результату кластеризації
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(y_pred) + 1))))
plt.figure(figsize=(6, 6))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=2, c=colors[y_pred])
plt.colorbar()
plt.title("Cluster visualization")
plt.show()

print("Shape of y_pred:", y_pred.shape)

###############################################
# Додатковий блок: класифікатори (опціонально)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

###############################################
# Розбиття даних для навчання
start_time = time.time()
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X_embedded, y_pred, random_state=42)
end_time = time.time()
print("Time for train/test split: {:.2f} ms".format((end_time - start_time) * 1000))

###############################################
# Побудова глибокої автоенкодерної моделі (shallow neural network)
def create_dense_ae():
    # Розмірність кодованого простору
    hidden_dim = 60
    encoding_dim = 2

    # Кодувальник
    input_layer = Input(shape=(10,))  # 10 - розмірність вхідного шару
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
# Перевірка наявних пристроїв
print("Physical devices:")
print(tf.config.list_physical_devices())

###############################################
# Створення та компіляція автоенкодера (використовуємо GPU)
with tf.device("/device:GPU:0"):
    encoder, decoder, autoencoder = create_dense_ae()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

###############################################
# Вивід архітектури моделі
autoencoder.summary()

###############################################
# Навчання автоенкодера
with tf.device("/device:GPU:0"):
    start_time = time.time()
    autoencoder.fit(train_x, train_x,
                    epochs=500,
                    batch_size=50,
                    shuffle=True,
                    validation_data=(valid_x, valid_x))
    end_time = time.time()
    print("Training time for autoencoder: {:.2f} ms".format((end_time - start_time) * 1000))

###############################################
# Візуалізація 2D-латентного простору автоенкодера
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(valid_y) + 1))))

start_time = time.time()
x_train_encoded = encoder.predict(train_x, batch_size=500)
plt.figure(figsize=(6, 6))
plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1], c=colors[train_y])
plt.colorbar()
plt.title("Encoded train data")
plt.show()

x_test_encoded = encoder.predict(valid_x, batch_size=500)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=colors[valid_y])
plt.colorbar()
plt.title("Encoded validation data")
plt.show()
end_time = time.time()
print("Time for encoding visualization: {:.2f} ms".format((end_time - start_time) * 1000))

###############################################
# Аугментація даних за допомогою RandomOverSampler
from imblearn.over_sampling import RandomOverSampler

start_time = time.time()
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_embedded, y_pred)
end_time = time.time()
print("Time for oversampling: {:.2f} ms".format((end_time - start_time) * 1000))

###############################################
# (За потреби можна додати додатковий код для порівняння ефективності класифікаторів з аугментацією та без неї)
