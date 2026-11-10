#%%
import os
import time
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#%% =========================
# Configuração
# =========================
TARGET_DIMS = [8, 16, 32, 64, 128, 256]
REPEATS = 10  # número de execuções para boxplot

#%% =========================
# Carregar CWRU
# =========================
df = pd.DataFrame(columns=['DE_data', 'fault'])

for root, dirs, files in os.walk("C:\\Users\\Arthur\\Desktop\\load_12K"):
    for file_name in files:
        path = os.path.join(root, file_name)
        mat = scipy.io.loadmat(path)
        key_name = list(mat.keys())[3]
        DE_data = mat.get(key_name)
        fault = np.full((len(DE_data), 1), file_name[:-4])

        df_temp = pd.DataFrame({
            'DE_data': np.ravel(DE_data),
            'fault': np.ravel(fault)
        })

        df = pd.concat([df, df_temp], axis=0)

signal_values = df['DE_data'].values
labels = df['fault'].values

#%% =========================
# Modelo
# =========================
def train_model(X_train, X_test, y_train, y_test):

    model = Sequential([
        Dense(256, activation='tanh', input_shape=(X_train.shape[1],)),
        Dense(128, activation='tanh'),
        Dense(64, activation='tanh'),
        Dense(32, activation='tanh'),
        Dense(y_train.shape[1], activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='accuracy',
        patience=20,
        restore_best_weights=True
    )

    start = time.time()
    model.fit(X_train, y_train, epochs=100, verbose=0, callbacks=[early_stop])
    train_time = time.time() - start

    _, acc = model.evaluate(X_test, y_test, verbose=0)

    return acc, train_time

#%% =========================
# Loop experimental (RAW + FFT)
# =========================
results_dict = {}

for dim in TARGET_DIMS:

    print(f"\n>>> Dimensão {dim}")

    results_dict[dim] = {
        "raw_acc": [], "fft_acc": [],
        "raw_time": [], "fft_time": []
    }

    for mode in ["raw", "fft"]:

        for r in range(REPEATS):

            X_blocks = []
            y_blocks = []

            for i in range(0, len(signal_values)-dim, dim):

                block = signal_values[i:i+dim]
                label = labels[i]

                if mode == "fft":
                    block = np.abs(np.fft.fft(block))

                X_blocks.append(block)
                y_blocks.append(label)

            X_blocks = np.array(X_blocks)
            y_blocks = np.array(y_blocks)

            scaler = MinMaxScaler()
            X_blocks = scaler.fit_transform(X_blocks)

            encoder = LabelEncoder()
            y_enc = to_categorical(encoder.fit_transform(y_blocks))

            X_train, X_test, y_train, y_test = train_test_split(
                X_blocks, y_enc,
                test_size=0.3,
                random_state=42
            )

            acc, train_time = train_model(X_train, X_test, y_train, y_test)

            if mode == "raw":
                results_dict[dim]["raw_acc"].append(acc)
                results_dict[dim]["raw_time"].append(train_time)
            else:
                results_dict[dim]["fft_acc"].append(acc)
                results_dict[dim]["fft_time"].append(train_time)

#%% =========================
# BOXPLOT - ACURÁCIA
# =========================
plt.figure(figsize=(10,6))

dims = sorted(results_dict.keys())

raw_mean = [np.mean(results_dict[d]["raw_acc"]) for d in dims]
fft_mean = [np.mean(results_dict[d]["fft_acc"]) for d in dims]

raw_std = [np.std(results_dict[d]["raw_acc"]) for d in dims]
fft_std = [np.std(results_dict[d]["fft_acc"]) for d in dims]

plt.errorbar(dims, raw_mean, yerr=raw_std, fmt='o-', label='Raw Data', capsize=5)
plt.errorbar(dims, fft_mean, yerr=fft_std, fmt='s-', label='FFT', capsize=5)

plt.xlabel("Dimension")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Dimension")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
#%% =========================
# BARPLOT - TEMPO MÉDIO
# =========================
mean_raw_time = [np.mean(results_dict[d]["raw_time"]) for d in dims]
mean_fft_time = [np.mean(results_dict[d]["fft_time"]) for d in dims]

x = np.arange(len(dims))

plt.figure(figsize=(10,6))
plt.bar(x - 0.2, mean_raw_time, width=0.4, label="Raw")
plt.bar(x + 0.2, mean_fft_time, width=0.4, label="FFT")

plt.xticks(x, dims)
plt.xlabel("Dimensão")
plt.ylabel("Tempo de Treino (s)")
plt.title("Processing Time")

plt.legend()
plt.show()

#%% =========================
# BOXPLOT - TEMPO (OPCIONAL)
# =========================
plt.figure(figsize=(12,6))

raw_time_data = [results_dict[d]["raw_time"] for d in dims]
fft_time_data = [results_dict[d]["fft_time"] for d in dims]

# RAW
plt.boxplot(
    raw_time_data,
    positions=x - 0.2,
    widths=0.35,
    patch_artist=True,
    boxprops=dict(facecolor='steelblue', alpha=0.7),
    medianprops=dict(color='black', linewidth=2),
)

# FFT
plt.boxplot(
    fft_time_data,
    positions=x + 0.2,
    widths=0.35,
    patch_artist=True,
    boxprops=dict(facecolor='orange', alpha=0.7),
    medianprops=dict(color='black', linewidth=2),
)

plt.xticks(x, dims)
plt.xlabel("Dimension")
plt.ylabel("Time (s)")
plt.title("Processing Time Distribution")

plt.legend(handles=legend_elements)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()