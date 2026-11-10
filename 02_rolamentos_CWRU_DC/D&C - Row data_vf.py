#%%
# =============================================================================
# D&C - Row data_vf.py
#
# Pré-processamento do bloco (raw / fft) x encoding quântico (amplitude / angle / dc) -> MLP
# =============================================================================

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

import pennylane as qml

#%% =========================
# Configuração
# =========================
MAX_DC_QUBITS = 15    # limite de qubits simuláveis para o D&C (2**MAX_DC_QUBITS amplitudes)

# Com MAX_DC_QUBITS = 15: dims 8 e 16 cabem (7 e 15 qubits); dim=32 precisaria
# de 31 qubits (2^31 amplitudes, ~34 GB) e fica de fora.
TARGET_DIMS = [d for d in [8, 16, 32, 64, 128, 256] if d - 1 <= MAX_DC_QUBITS]

REPEATS = 10          # retrainos da MLP para raw / fft / amplitude / angle
REPEATS_DC = 5        # retrainos da MLP para D&C (encoding mais caro computacionalmente)
MAX_SAMPLES_DC = 300  # subamostragem de blocos só para o modo D&C

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
# Modelo (MLP) - o mesmo para todas as codificações
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
# Blocagem do sinal (igual ao v3)
# =========================
def make_blocks(dim):
    X_blocks = []
    y_blocks = []
    for i in range(0, len(signal_values) - dim, dim):
        X_blocks.append(signal_values[i:i + dim])
        y_blocks.append(labels[i])
    return np.array(X_blocks), np.array(y_blocks)

#%% =========================
# Codificação 1: Raw Data
# =========================
def encode_raw(X_blocks):
    return X_blocks.copy()

#%% =========================
# Codificação 2: Raw Data + FFT (módulo)
# =========================
def encode_fft(X_blocks):
    return np.abs(np.fft.fft(X_blocks, axis=1))

#%% =========================
# Codificação 3: Amplitude Encoding
#
# =========================
def _z_sign_matrix(n_qubits):
    """Matriz (2**n_qubits, n_qubits) com +1/-1 conforme o bit do índice
    da base computacional (wire 0 = bit mais significativo, convenção
    usada pelo PennyLane)."""
    dim = 2 ** n_qubits
    idx = np.arange(dim)
    bits = (idx[:, None] >> (n_qubits - 1 - np.arange(n_qubits))) & 1
    return 1 - 2 * bits  # bit 0 -> +1 ; bit 1 -> -1

def encode_amplitude(X_blocks):
    dim = X_blocks.shape[1]
    n_qubits = int(np.log2(dim))
    assert 2 ** n_qubits == dim, "Amplitude encoding requer dimensão potência de 2"

    norms = np.linalg.norm(X_blocks, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    probs = (X_blocks / norms) ** 2  # |amplitude_j|^2, shape (n_amostras, dim)

    S = _z_sign_matrix(n_qubits)     # (dim, n_qubits)
    return probs @ S                 # (n_amostras, n_qubits)

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
# Codificação 4: Angle Encoding (qubit encoding do artigo)
#
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

#%% =========================
# Codificação 5: Divide & Conquer (Araujo et al. 2021, Sci Rep 11:6329)
#
# Implementação do Algoritmo 1 (gen_angles) e Algoritmo 3 (circuito com
# rotações Ry + portas CSWAP) do artigo. Carrega um vetor real de dimensão
# N usando N-1 qubits, com informação emaranhada nos qubits "ancilla".
#
# =========================
def _parent(k): return (k - 1) // 2
def _left(k):   return 2 * k + 1
def _right(k):  return 2 * k + 2

def gen_angles(x):
    """Algoritmo 1 do artigo. x: vetor real de tamanho N (potência de 2).
    Retorna vetor de ângulos de tamanho N-1, em ordem de heap binário
    (raiz no índice 0, depois nível a nível)."""
    N = len(x)
    if N == 1:
        return np.array([])

    new_x = np.sqrt(x[0::2] ** 2 + x[1::2] ** 2)
    inner_angles = gen_angles(new_x)

    angles = np.zeros(N // 2)
    for k in range(N // 2):
        if new_x[k] != 0:
            ratio = np.clip(x[2 * k + 1] / new_x[k], -1.0, 1.0)
            if x[2 * k] > 0:
                angles[k] = 2 * np.arcsin(ratio)
            else:
                angles[k] = 2 * np.pi - 2 * np.arcsin(ratio)
        else:
            angles[k] = 0.0

    return np.concatenate([inner_angles, angles])

def _dc_circuit_ops(angles, n_wires):
    """Algoritmo 3 do artigo: inicializa cada qubit com Ry(angle[k]) e
    depois combina os estados bottom-up com portas CSWAP."""
    for k in range(n_wires):
        qml.RY(angles[k], wires=k)

    actual = _parent(n_wires - 1)
    while actual >= 0:
        left_idx = _left(actual)
        right_idx = _right(actual)
        while right_idx < n_wires:
            qml.CSWAP(wires=[actual, left_idx, right_idx])
            left_idx = _left(left_idx)
            right_idx = _left(right_idx)
        actual -= 1

def encode_dc(X_blocks):
    dim = X_blocks.shape[1]
    n_wires = dim - 1

    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(x):
        angles = gen_angles(np.asarray(x, dtype=float))
        _dc_circuit_ops(angles, n_wires)
        return [qml.expval(qml.PauliZ(w)) for w in range(n_wires)]

    return np.array([circuit(x) for x in X_blocks])

# -----------------------------------------------------------------------
#   - "raw"           : bloco bruto -> MLP direto            (baseline v3)
#   - "fft"           : módulo da FFT do bloco -> MLP direto (baseline v3)
#   - "raw_amplitude" : bloco bruto -> amplitude encoding -> MLP
#   - "raw_angle"     : bloco bruto -> angle encoding -> MLP
#   - "raw_dc"        : bloco bruto -> D&C encoding -> MLP
#   - "fft_amplitude" : módulo da FFT -> amplitude encoding -> MLP
#   - "fft_angle"     : módulo da FFT -> angle encoding -> MLP
#   - "fft_dc"        : módulo da FFT -> D&C encoding -> MLP
# -----------------------------------------------------------------------
PREPROCESSORS = {
    "raw": encode_raw,
    "fft": encode_fft,
}

QUANTUM_ENCODERS = {
    "amplitude": encode_amplitude,
    "angle": encode_angle,
    "dc": encode_dc,
}

#%% =========================
# Validação das codificações quânticas (roda uma vez, no início)
# =========================
print("Validando amplitude encoding (fórmula fechada vs circuito PennyLane)...")
_validate_amplitude_encoding()
print("Validando angle encoding (fórmula fechada vs circuito PennyLane)...")
_validate_angle_encoding()
print("OK: codificações validadas.\n")

#%% =========================
# Loop experimental
# =========================
results_dict = {}

# nomes de todos os modos testados, na ordem em que aparecem nos gráficos
ALL_MODES = ["raw", "fft"] + [
    f"{prep}_{enc}" for prep in PREPROCESSORS for enc in QUANTUM_ENCODERS
]

for dim in TARGET_DIMS:

    print(f"\n>>> Dimensão {dim}")
    X_blocks_full, y_blocks_full = make_blocks(dim)

    # cada modo guarda: acc, encode_time (custo de gerar as features -
    # FFT/amplitude/angle/dc), train_time (custo de treinar a MLP) e
    # total_time (encode_time + train_time)
    results_dict[dim] = {
        m: {"acc": [], "encode_time": [], "train_time": [], "total_time": []}
        for m in ALL_MODES
    }

    for prep_name, prep_fn in PREPROCESSORS.items():

        t0 = time.time()
        block_features_full = prep_fn(X_blocks_full)
        prep_time = time.time() - t0

        # --- baseline direto (sem encoding quântico) ---
        for r in range(REPEATS):
            X_features = block_features_full
            y_blocks = y_blocks_full

            scaler = MinMaxScaler()
            X_features = scaler.fit_transform(X_features)

            label_enc = LabelEncoder()
            y_enc = to_categorical(label_enc.fit_transform(y_blocks))

            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y_enc, test_size=0.3, random_state=RANDOM_STATE
            )

            acc, train_time = train_model(X_train, X_test, y_train, y_test)

            results_dict[dim][prep_name]["acc"].append(acc)
            results_dict[dim][prep_name]["encode_time"].append(prep_time)
            results_dict[dim][prep_name]["train_time"].append(train_time)
            results_dict[dim][prep_name]["total_time"].append(prep_time + train_time)

        # --- encodings quânticos aplicados sobre o bloco pré-processado ---
        for enc_name, enc_fn in QUANTUM_ENCODERS.items():

            mode = f"{prep_name}_{enc_name}"
            n_repeats = REPEATS_DC if enc_name == "dc" else REPEATS

            for r in range(n_repeats):

                if enc_name == "dc" and len(block_features_full) > MAX_SAMPLES_DC:
                    sel = rng.choice(len(block_features_full), size=MAX_SAMPLES_DC, replace=False)
                    block_features, y_blocks = block_features_full[sel], y_blocks_full[sel]
                else:
                    block_features, y_blocks = block_features_full, y_blocks_full

                t0 = time.time()
                X_features = enc_fn(block_features)
                encode_time = time.time() - t0

                scaler = MinMaxScaler()
                X_features = scaler.fit_transform(X_features)

                label_enc = LabelEncoder()
                y_enc = to_categorical(label_enc.fit_transform(y_blocks))

                X_train, X_test, y_train, y_test = train_test_split(
                    X_features, y_enc, test_size=0.3, random_state=RANDOM_STATE
                )

                acc, train_time = train_model(X_train, X_test, y_train, y_test)

                results_dict[dim][mode]["acc"].append(acc)
                results_dict[dim][mode]["encode_time"].append(encode_time)
                results_dict[dim][mode]["train_time"].append(train_time)
                results_dict[dim][mode]["total_time"].append(encode_time + train_time)

#%% =========================
# GRÁFICO - ACURÁCIA vs DIMENSÃO (todas as codificações)
# =========================
plt.figure(figsize=(10, 6))
dims = sorted(results_dict.keys())
styles = {
    "raw": "o-", "fft": "s-",
    "raw_amplitude": "^-", "raw_angle": "d-", "raw_dc": "x--",
    "fft_amplitude": "^:", "fft_angle": "d:", "fft_dc": "x:",
}

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
plt.title("Accuracy vs Dimension - Raw/FFT (baseline) x Amplitude/Angle/D&C encoding")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("accuracy_vs_dimension.png", dpi=150, bbox_inches="tight")
plt.show()

#%% =========================
# GRÁFICO - TEMPO TOTAL (encoding + treino da MLP)
#
# =========================
plt.figure(figsize=(12, 6))
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
plt.title("Processing Time (encode_time + train_time)")
plt.legend(fontsize=8)
plt.savefig("time_vs_dimension.png", dpi=150, bbox_inches="tight")
plt.show()

#%% =========================
# GRÁFICO - QUEBRA encode_time x train_time (só os 3 encodings quânticos)
#
# =========================
quantum_modes = [m for m in styles if m not in ("raw", "fft")]

plt.figure(figsize=(12, 6))
width2 = 0.8 / len(quantum_modes)
x2 = np.arange(len(dims))

for i, mode in enumerate(quantum_modes):
    encode_means = []
    train_means = []
    for d in dims:
        vals = results_dict[d].get(mode, {})
        encode_means.append(np.mean(vals["encode_time"]) if vals.get("encode_time") else np.nan)
        train_means.append(np.mean(vals["train_time"]) if vals.get("train_time") else np.nan)
    pos = x2 + (i - len(quantum_modes) / 2) * width2
    plt.bar(pos, encode_means, width=width2, label=f"{mode} (encode)", color=f"C{i}", alpha=0.9)
    plt.bar(pos, train_means, width=width2, bottom=encode_means, label=f"{mode} (train)", color=f"C{i}", alpha=0.4)

plt.xticks(x2, dims)
plt.xlabel("Dimensão")
plt.ylabel("Tempo (s)")
plt.title("Encode time vs Train time - Amplitude / Angle / D&C")
plt.legend(fontsize=7, ncol=2)
plt.savefig("encode_vs_train_time.png", dpi=150, bbox_inches="tight")
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

pd.DataFrame(rows).to_csv("results_raw_fft_amplitude_angle_dc.csv", index=False)
print("Resultados salvos em results_raw_fft_amplitude_angle_dc.csv")
# %%
