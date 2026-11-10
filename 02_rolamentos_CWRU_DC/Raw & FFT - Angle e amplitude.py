#%%
# =============================================================================
# Raw & FFT - Angle e amplitude.py
#
# raw e fft direto na MLP; amplitude e angle aplicados sobre o módulo da FFT do bloco
# =============================================================================

import os
import gc
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
from tensorflow.keras import backend as K

import pennylane as qml

#%% =========================
# Configuração
# =========================
TARGET_DIMS = [8, 16, 32, 64, 128, 256]  # raw, fft, amplitude e angle rodam em todas
REPEATS = 1            # retreinos da MLP para raw / fft / amplitude / angle

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

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

signal_values = df['DE_data'].values.astype(float)
labels = df['fault'].values

#%% =========================
# Modelo (MLP) - com limpeza de memória a cada chamada
# =========================
def train_model(X_train, X_test, y_train, y_test):

    K.clear_session()

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

    del model
    gc.collect()

    return acc, train_time

#%% =========================
# Blocagem do sinal (igual ao original)
# =========================
def make_blocks(dim):
    X_blocks = []
    y_blocks = []
    for i in range(0, len(signal_values) - dim, dim):
        X_blocks.append(signal_values[i:i + dim])
        y_blocks.append(labels[i])
    return np.array(X_blocks), np.array(y_blocks)

#%% =========================
# Amplitude Encoding (fórmula fechada, validada contra circuito real)
#
# Vetor normalizado do bloco carregado nas amplitudes de um estado
# quântico de log2(dim) qubits. Feature clássica = <Z_i> de cada qubit.
# =========================
def _z_sign_matrix(n_qubits):
    dim = 2 ** n_qubits
    idx = np.arange(dim)
    bits = (idx[:, None] >> (n_qubits - 1 - np.arange(n_qubits))) & 1
    return 1 - 2 * bits

def encode_amplitude(X_blocks):
    dim = X_blocks.shape[1]
    n_qubits = int(np.log2(dim))
    assert 2 ** n_qubits == dim, "Amplitude encoding requer dimensão potência de 2"

    norms = np.linalg.norm(X_blocks, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    probs = (X_blocks / norms) ** 2

    S = _z_sign_matrix(n_qubits)
    return probs @ S

def _validate_amplitude_encoding():
    n_qubits = 3
    dim = 2 ** n_qubits
    x = rng.normal(size=dim)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(vec):
        qml.AmplitudeEmbedding(vec, wires=range(n_qubits), normalize=True)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    z_circuit = np.array(circuit(x))
    z_closed = encode_amplitude(x.reshape(1, -1))[0]

    assert np.allclose(z_circuit, z_closed, atol=1e-6), (
        f"Amplitude encoding: fórmula fechada diverge do circuito "
        f"({z_circuit} vs {z_closed})"
    )

#%% =========================
# Angle (Qubit) Encoding (fórmula fechada, validada contra circuito real)
#
# Cada valor do bloco vira o ângulo de uma rotação Ry independente em um
# qubit próprio (estado separável, sem emaranhamento). Dados reescalados
# para [0, pi]. <Z_i> = cos(theta_i).
# =========================
def encode_angle(X_blocks):
    x_min, x_max = X_blocks.min(), X_blocks.max()
    if x_max == x_min:
        theta = np.zeros_like(X_blocks)
    else:
        theta = np.pi * (X_blocks - x_min) / (x_max - x_min)
    return np.cos(theta)

def _validate_angle_encoding():
    n_qubits = 4
    theta = rng.uniform(0, np.pi, size=n_qubits)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(angles):
        for i, a in enumerate(angles):
            qml.RY(a, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    z_circuit = np.array(circuit(theta))
    z_closed = np.cos(theta)

    assert np.allclose(z_circuit, z_closed, atol=1e-6), (
        "Angle encoding: fórmula fechada diverge do circuito"
    )

print("Validando amplitude encoding (fórmula fechada vs circuito PennyLane)...")
_validate_amplitude_encoding()
print("OK: amplitude encoding validado.\n")

print("Validando angle encoding (fórmula fechada vs circuito PennyLane)...")
_validate_angle_encoding()
print("OK: angle encoding validado.\n")

#%% =========================
# Loop experimental - raw, fft, amplitude, angle
#
# =========================
results_dict = {}

for dim in TARGET_DIMS:

    print(f"\n>>> Dimensão {dim}")

    X_blocks, y_blocks_all = make_blocks(dim)

    modes = ["raw", "fft", "amplitude", "angle"]

    results_dict[dim] = {m: {"acc": [], "encode_time": [], "train_time": [], "total_time": []} for m in modes}

    for mode in modes:

        for r in range(REPEATS):
            print(f"    [{mode}] repetição {r + 1}/{REPEATS}")

            X_b, y_b = X_blocks, y_blocks_all

            t0 = time.time()
            if mode == "raw":
                X_features = X_b.copy()
            elif mode == "fft":
                X_features = np.abs(np.fft.fft(X_b, axis=1))
            elif mode == "amplitude":
                fft_block = np.abs(np.fft.fft(X_b, axis=1))
                X_features = encode_amplitude(fft_block)
            else:  # angle
                fft_block = np.abs(np.fft.fft(X_b, axis=1))
                X_features = encode_angle(fft_block)
            encode_time = time.time() - t0

            scaler = MinMaxScaler()
            X_features = scaler.fit_transform(X_features)

            label_enc = LabelEncoder()
            y_enc = to_categorical(label_enc.fit_transform(y_b))

            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y_enc, test_size=0.3, random_state=RANDOM_STATE
            )

            acc, train_time = train_model(X_train, X_test, y_train, y_test)

            results_dict[dim][mode]["acc"].append(acc)
            results_dict[dim][mode]["encode_time"].append(encode_time)
            results_dict[dim][mode]["train_time"].append(train_time)
            results_dict[dim][mode]["total_time"].append(encode_time + train_time)

#%% =========================
# GRÁFICO - ACURÁCIA vs DIMENSÃO
# =========================
plt.figure(figsize=(10, 6))
dims = sorted(results_dict.keys())
styles = {"raw": "o-", "fft": "s-", "amplitude": "^-", "angle": "d-"}

for mode, style in styles.items():
    xs, means, stds = [], [], []
    for d in dims:
        vals = results_dict[d].get(mode, {}).get("acc", [])
        if vals:
            xs.append(d)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
    if xs:
        plt.errorbar(xs, means, yerr=stds, fmt=style, label=mode, capsize=5)

plt.xlabel("Dimension")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Dimension - Raw / FFT / Amplitude / Angle (sobre módulo da FFT)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("accuracy_vs_dimension_v7.png", dpi=150, bbox_inches="tight")
plt.show()

#%% =========================
# GRÁFICO - TEMPO TOTAL (encoding + treino)
# =========================
plt.figure(figsize=(10, 6))
n_modes = len(styles)
width = 0.8 / n_modes
x = np.arange(len(dims))

for i, mode in enumerate(styles):
    means = []
    for d in dims:
        vals = results_dict[d].get(mode, {}).get("total_time", [])
        means.append(np.mean(vals) if vals else np.nan)
    plt.bar(x + (i - n_modes / 2) * width, means, width=width, label=mode)

plt.xticks(x, dims)
plt.xlabel("Dimensão")
plt.ylabel("Tempo Total: encoding + treino (s)")
plt.title("Processing Time")
plt.legend()
plt.savefig("time_vs_dimension_v7.png", dpi=150, bbox_inches="tight")
plt.show()

#%% =========================
# Exportar resultados para CSV
# =========================
rows = []
for d in dims:
    for mode in styles:
        vals = results_dict[d].get(mode)
        if vals:
            for acc, enc_t, tr_t, tot_t in zip(
                vals["acc"], vals["encode_time"], vals["train_time"], vals["total_time"]
            ):
                rows.append({
                    "dim": d, "mode": mode, "accuracy": acc,
                    "encode_time": enc_t, "train_time": tr_t, "total_time": tot_t,
                })

pd.DataFrame(rows).to_csv("results_raw_fft_amplitude_angle.csv", index=False)
print("Resultados salvos em results_raw_fft_amplitude_angle.csv")
# %%
