import os
import time
import warnings
import pickle
import numpy as np
from itertools import cycle, islice
import platform

from sklearn import decomposition, model_selection, metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

# Аугментація даних
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

import tensorflow as tf
# перейменовуємо Input з Keras, щоб уникнути конфлікту з Dash
from tensorflow.keras.layers import Input as KInput, Dense, Flatten, Reshape
from tensorflow.keras.models import Model

# Dash imports для візуалізації результатів
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_table
import plotly.graph_objs as go


# =====================
# Функція для побудови автоенкодера
def create_dense_ae(input_dim=10, hidden_dim=60, encoding_dim=2):
    # Кодувальник
    input_layer = KInput(shape=(input_dim,))
    flat = Flatten()(input_layer)
    hidden = Dense(hidden_dim, activation='relu')(flat)
    hidden2 = Dense(hidden_dim, activation='relu')(hidden)
    encoded = Dense(encoding_dim, activation='relu')(hidden2)
    # Декодувальник
    input_encoded = KInput(shape=(encoding_dim,))
    hidden_encoded = Dense(hidden_dim, activation='sigmoid')(input_encoded)
    hidden_encoded2 = Dense(hidden_dim, activation='sigmoid')(hidden_encoded)
    flat_decoded = Dense(input_dim, activation='sigmoid')(hidden_encoded2)
    decoded = Reshape((input_dim,))(flat_decoded)
    encoder = Model(input_layer, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_layer, decoder(encoder(input_layer)), name="autoencoder")
    return encoder, decoder, autoencoder


# =====================
# Функція для запуску повного пайплайна на заданому пристрої ("cpu" або "gpu")
def run_pipeline(device_choice="cpu"):
    results = {}  # словник для зберігання результатів
    timings = {}

    # Якщо пристрій CPU – відключаємо GPU
    if device_choice.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device_tf = "/cpu:0"
    else:
        device_tf = "/device:GPU:0"

    # 1. Завантаження даних
    with open('shaped.pickle', 'rb') as f:
        ab = pickle.load(f)
    results["Розмір даних"] = ab.shape

    # 2. Редукція розмірності за допомогою TruncatedSVD (10 вимірів)
    t0 = time.time()
    pca = decomposition.TruncatedSVD(n_components=10)
    X_transformed = pca.fit_transform(ab)
    t1 = time.time()
    timings["SVD (мс)"] = (t1 - t0) * 1000

    # 3. Отримання 2D-простору за допомогою TSNE (для кластеризації та візуалізації)
    t0 = time.time()
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_transformed)
    t1 = time.time()
    timings["TSNE (мс)"] = (t1 - t0) * 1000

    # 4. Кластеризація за допомогою KMeans (на TSNE-даних)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(X_embedded)
        y_pred = kmeans.labels_.astype(int)
    t1 = time.time()
    timings["KMeans (мс)"] = (t1 - t0) * 1000
    results["Розмір y_pred"] = y_pred.shape

    # 5. Побудова та навчання автоенкодера (навчаємо на X_transformed, тобто 10D)
    encoder, decoder, autoencoder = create_dense_ae(input_dim=10)
    with tf.device(device_tf):
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
        # Розбиття даних для автоенкодера
        X_train_ae, X_valid_ae, _, _ = model_selection.train_test_split(X_transformed, y_pred, test_size=0.3,
                                                                        random_state=42)
        t0 = time.time()
        history = autoencoder.fit(X_train_ae, X_train_ae,
                                  epochs=500,
                                  batch_size=50,
                                  shuffle=True,
                                  validation_data=(X_valid_ae, X_valid_ae),
                                  verbose=0)
        t1 = time.time()
        timings["Навчання автоенкодера (мс)"] = (t1 - t0) * 1000
        results["Втрата (валидація)"] = history.history["val_loss"][-1]
        results["Точність (валидація)"] = history.history["val_accuracy"][-1]

    # Обчислюємо середній час ітерації (за 500 епох, можна вважати що друга ітерація стабільна)
    results["Середній час ітерації (мс)"] = timings["Навчання автоенкодера (мс)"] / 500

    # 6. Отримання 2D-латентного простору (використовуємо encoder)
    x_train_encoded = encoder.predict(X_train_ae, batch_size=50)
    results["Розмір латентного простору"] = x_train_encoded.shape

    # 7. Навчання класифікатора та аугментація
    def evaluate_classifier(X_tr, y_tr, X_val, y_val, clf, clf_name="Класифікатор"):
        start = time.time()
        clf.fit(X_tr, y_tr)
        y_pred_clf = clf.predict(X_val)
        acc = metrics.accuracy_score(y_val, y_pred_clf)
        elapsed = time.time() - start
        return acc, elapsed * 1000

    X_train_cls, X_valid_cls, y_train_cls, y_valid_cls = model_selection.train_test_split(X_embedded, y_pred,
                                                                                          test_size=0.3,
                                                                                          random_state=42)
    clf_baseline = LogisticRegression(max_iter=1000, random_state=42)
    acc_baseline, elapsed_baseline = evaluate_classifier(X_train_cls, y_train_cls, X_valid_cls, y_valid_cls,
                                                         clf_baseline, "Базовий LR")
    results["Точність (Базовий)"] = acc_baseline
    timings["Час (Базовий, мс)"] = elapsed_baseline

    # Функція для аугментації
    def augment_and_evaluate(aug, aug_name):
        X_tr_aug, y_tr_aug = aug.fit_resample(X_train_cls, y_train_cls)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        acc, elapsed = evaluate_classifier(X_tr_aug, y_tr_aug, X_valid_cls, y_valid_cls, clf, aug_name)
        return acc, elapsed

    ros = RandomOverSampler(random_state=42)
    acc_ros, time_ros = augment_and_evaluate(ros, "ROS")
    smote = SMOTE(random_state=42)
    acc_smote, time_smote = augment_and_evaluate(smote, "SMOTE")
    adasyn = ADASYN(random_state=42)
    acc_adasyn, time_adasyn = augment_and_evaluate(adasyn, "ADASYN")

    results["Точність (ROS)"] = acc_ros
    results["Точність (SMOTE)"] = acc_smote
    results["Точність (ADASYN)"] = acc_adasyn

    timings["Час (ROS, мс)"] = time_ros
    timings["Час (SMOTE, мс)"] = time_smote
    timings["Час (ADASYN, мс)"] = time_adasyn

    return {"Пристрій": device_choice.upper(), "Результати": results, "Часи": timings,
            "X_embedded": X_embedded, "y_pred": y_pred}


# =====================
# Отримання даних про систему
os_version = f"{platform.system()} {platform.release()}"
tf_version = tf.__version__
# Виробники (для звіту – без деталей)
cpu_manufacturer = "AMD"  # Припустимо, що це AMD
gpu_manufacturer = "NVIDIA"  # Припустимо, що це NVIDIA

# =====================
# Запускаємо пайплайни для CPU та GPU
print("Запуск на CPU...")
cpu_output = run_pipeline(device_choice="cpu")
print("Пайплайн CPU завершено.")
print("Запуск на GPU...")
gpu_output = run_pipeline(device_choice="gpu")
print("Пайплайн GPU завершено.")

# Для консольного виводу – виводимо обидва результати
print("\n=== Результати CPU ===")
print(cpu_output["Результати"])
print(cpu_output["Часи"])
print("\n=== Результати GPU ===")
print(gpu_output["Результати"])
print(gpu_output["Часи"])

# =====================
# Технічні характеристики (HTML розмітка для гарного відображення)
cpu_specs = html.Div([
    html.H3("Технічні характеристики CPU"),
    html.Ul([
        html.Li("ЦП: AMD Ryzen 5 5600 6-Core Processor"),
        html.Li("Базова швидкість: 3,50 ГГц"),
        html.Li("Сокетів: 1"),
        html.Li("Ядра: 6"),
        html.Li("Логічних процесорів: 12"),
        html.Li("Віртуалізація: Включено"),
        html.Li("Кеш L1: 384 КБ"),
        html.Li("Кеш L2: 3,0 МБ"),
        html.Li("Кеш L3: 32,0 МБ"),
        html.Li("Використання: 2%"),
        html.Li("Швидкість: 3,77 ГГц"),
        html.Li("Час роботи: 2:23:50:13"),
        html.Li("Процеси: 209"),
        html.Li("Потоки: 3624"),
        html.Li("Дескриптори: 100352")
    ], style={'fontSize': '16px', 'lineHeight': '1.6'})
], style={'backgroundColor': '#f9f9f9', 'padding': '10px', 'border': '1px solid #ccc', 'marginBottom': '20px'})

gpu_specs = html.Div([
    html.H3("Технічні характеристики GPU"),
    html.Ul([
        html.Li("Графічний процесор: NVIDIA GeForce RTX 3070 palit"),
        html.Li("Версія драйвера: 32.0.15.6094"),
        html.Li("Дата розробки: 14.08.2024"),
        html.Li("Версія DirectX: 12 (FL 12.1)"),
        html.Li("Фізичне розташування: PCI-шина 38, пристрій 0, функція 0"),
        html.Li("Використання: 4%"),
        html.Li("Виділена пам'ять: 0,7/8,0 ГБ"),
        html.Li("Загальна пам'ять: 0,1/16,0 ГБ"),
        html.Li("Оперативна пам'ять: 0,7/24,0 ГБ")
    ], style={'fontSize': '16px', 'lineHeight': '1.6'})
], style={'backgroundColor': '#f9f9f9', 'padding': '10px', 'border': '1px solid #ccc', 'marginBottom': '20px'})

# =====================
# Формування звіту (HTML)
report_details = html.Div([
    html.H3("Додатковий звіт"),
    html.Ul([
        html.Li(f"Операційна система: {os_version}", style={'fontSize': '16px'}),
        html.Li(f"Версія TensorFlow: {tf_version}", style={'fontSize': '16px'}),
        html.Li(f"Виробник CPU: {cpu_manufacturer}", style={'fontSize': '16px'}),
        html.Li(f"Виробник GPU: {gpu_manufacturer}", style={'fontSize': '16px'}),
        html.Li(f"Середній час ітерації навчання автоенкодера (мс):", style={'fontSize': '16px'}),
        html.Ul([
            html.Li(f"CPU: {cpu_output['Результати']['Середній час ітерації (мс)']:.2f} мс",
                    style={'fontSize': '16px'}),
            html.Li(f"GPU: {gpu_output['Результати']['Середній час ітерації (мс)']:.2f} мс", style={'fontSize': '16px'})
        ])
    ], style={'lineHeight': '1.6'})
], style={'backgroundColor': '#e8f5e9', 'padding': '10px', 'border': '1px solid #4caf50', 'marginBottom': '20px'})

# =====================
# Таблиця метрик часу (Dash DataTable)
time_table = dash_table.DataTable(
    id='time-table',
    columns=[{"name": key, "id": key} for key in cpu_output["Часи"].keys()],
    data=[{**{"Пристрій": "CPU"}, **cpu_output["Часи"]},
          {**{"Пристрій": "GPU"}, **gpu_output["Часи"]}],
    style_cell={'textAlign': 'center', 'fontSize': '14px'},
    style_header={'fontWeight': 'bold'}
)

# Таблиця показників ефективності (Dash DataTable)
acc_table = dash_table.DataTable(
    id='acc-table',
    columns=[{"name": key, "id": key} for key in cpu_output["Результати"].keys() if
             key not in ["Розмір даних", "Розмір y_pred", "Розмір латентного простору"]],
    data=[{**{"Пристрій": "CPU"}, **{k: v for k, v in cpu_output["Результати"].items() if
                                     k not in ["Розмір даних", "Розмір y_pred", "Розмір латентного простору"]}},
          {**{"Пристрій": "GPU"}, **{k: v for k, v in gpu_output["Результати"].items() if
                                     k not in ["Розмір даних", "Розмір y_pred", "Розмір латентного простору"]}}],
    style_cell={'textAlign': 'center', 'fontSize': '14px'},
    style_header={'fontWeight': 'bold'}
)

# =====================
# Dash-додаток для порівняння CPU та GPU
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("CV lab 3 Bogdan Cheban TTP-42 (CPU vs GPU)", style={'textAlign': 'center', 'marginBottom': '20px'}),

    html.Div([
        html.Div(cpu_specs,
                 style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
        html.Div(gpu_specs, style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ]),

    html.Hr(style={'marginTop': '20px'}),

    report_details,

    html.Hr(style={'marginTop': '20px'}),

    html.H2("Метрики часу (мс)", style={'textAlign': 'center'}),
    dcc.Graph(
        id='time-bar',
        figure={
            'data': [
                go.Bar(name='CPU', x=list(cpu_output["Часи"].keys()), y=list(cpu_output["Часи"].values())),
                go.Bar(name='GPU', x=list(gpu_output["Часи"].keys()), y=list(gpu_output["Часи"].values()))
            ],
            'layout': go.Layout(title='Час роботи (мс) для різних етапів', barmode='group', font=dict(size=14))
        }
    ),
    html.Br(),
    html.H2("Таблиця метрик часу", style={'textAlign': 'center'}),
    time_table,

    html.Hr(style={'marginTop': '20px'}),

    html.H2("Показники ефективності (точність)", style={'textAlign': 'center'}),
    dcc.Graph(
        id='acc-bar',
        figure={
            'data': [
                go.Bar(name='CPU', x=['Базовий', 'ROS', 'SMOTE', 'ADASYN'],
                       y=[cpu_output["Результати"]["Точність (Базовий)"],
                          cpu_output["Результати"]["Точність (ROS)"],
                          cpu_output["Результати"]["Точність (SMOTE)"],
                          cpu_output["Результати"]["Точність (ADASYN)"]]),
                go.Bar(name='GPU', x=['Базовий', 'ROS', 'SMOTE', 'ADASYN'],
                       y=[gpu_output["Результати"]["Точність (Базовий)"],
                          gpu_output["Результати"]["Точність (ROS)"],
                          gpu_output["Результати"]["Точність (SMOTE)"],
                          gpu_output["Результати"]["Точність (ADASYN)"]])
            ],
            'layout': go.Layout(title='Точність класифікатора', barmode='group', yaxis=dict(range=[0, 1]),
                                font=dict(size=14))
        }
    ),
    html.Br(),
    html.H2("Таблиця показників ефективності", style={'textAlign': 'center'}),
    acc_table,

    html.Hr(style={'marginTop': '20px'}),

    html.H2("Кластеризація (TSNE)", style={'textAlign': 'center'}),
    dcc.Graph(
        id='cluster-scatter',
        figure={
            'data': [
                go.Scatter(
                    x=cpu_output["X_embedded"][:, 0],
                    y=cpu_output["X_embedded"][:, 1],
                    mode='markers',
                    marker=dict(
                        color=cpu_output["y_pred"],
                        colorscale='Viridis',
                        showscale=True,
                        size=8
                    ),
                    name='CPU'
                ),
                go.Scatter(
                    x=gpu_output["X_embedded"][:, 0],
                    y=gpu_output["X_embedded"][:, 1],
                    mode='markers',
                    marker=dict(
                        color=gpu_output["y_pred"],
                        colorscale='Cividis',
                        showscale=True,
                        size=8
                    ),
                    name='GPU'
                )
            ],
            'layout': go.Layout(title='TSNE кластеризація (CPU vs GPU)', xaxis=dict(title='Компонента 1'),
                                yaxis=dict(title='Компонента 2'), font=dict(size=14))
        }
    ),

    html.Hr(style={'marginTop': '20px'}),

    html.H2("Autoencoder: Втрата та Точність (Валідація)", style={'textAlign': 'center'}),
    dcc.Graph(
        id='autoencoder-performance',
        figure={
            'data': [
                go.Scatter(
                    x=["Втрата", "Точність"],
                    y=[cpu_output["Результати"]["Втрата (валидація)"],
                       cpu_output["Результати"]["Точність (валидація)"]],
                    mode='markers+lines',
                    name='CPU Autoencoder',
                    marker=dict(size=12)
                ),
                go.Scatter(
                    x=["Втрата", "Точність"],
                    y=[gpu_output["Результати"]["Втрата (валидація)"],
                       gpu_output["Результати"]["Точність (валидація)"]],
                    mode='markers+lines',
                    name='GPU Autoencoder',
                    marker=dict(size=12)
                )
            ],
            'layout': go.Layout(title='Показники Autoencoder', yaxis=dict(automargin=True), font=dict(size=14))
        }
    )
])

if __name__ == '__main__':
    app.run(debug=True)
