#%%
# =============================================================================
# Raw & FFT - D&C.py
#
# raw e fft direto na MLP; D&C (fórmula fechada) aplicado sobre o módulo da FFT do bloco
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
TARGET_DIMS = [8, 16, 32, 64, 128, 256]  # raw, fft e dc rodam em todas
REPEATS = 1            # retreinos da MLP para todos os modos

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
# Divide & Conquer Encoding (Araujo et al. 2021, Sci Rep 11:6329)
#
# Algoritmo 1 (gen_angles) + Algoritmo 3 (circuito com Ry + CSWAP) do
# artigo, mantidos aqui só para a VALIDAÇÃO em dimensão pequena.
#
# =========================
def _parent(k): return (k - 1) // 2
def _left(k):   return 2 * k + 1
def _right(k):  return 2 * k + 2

def gen_angles(x):
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

def _data_qubit_indices(n_data_qubits):
    """Caminho mais à esquerda da árvore (heap binário): 0, 1, 3, 7, 15, ...
    = 2^d - 1. É onde o artigo (Fig. 4) mostra que o estado final fica
    armazenado (ex.: q[0], q[1], q[3] para dim=8)."""
    return [2 ** d - 1 for d in range(n_data_qubits)]

def _dc_circuit_probs(X_blocks):
    """Simula o circuito REAL (Ry + CSWAP) e lê qml.probs() nos qubits de
    dado. Só usado para validação (dimensão pequena) - custo O(2^(dim-1))."""
    dim = X_blocks.shape[1]
    n_wires = dim - 1
    n_data_qubits = int(np.log2(dim))
    data_qubits = _data_qubit_indices(n_data_qubits)

    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(x):
        angles = gen_angles(np.asarray(x, dtype=float))
        _dc_circuit_ops(angles, n_wires)
        return qml.probs(wires=data_qubits)

    return np.array([circuit(x) for x in X_blocks])

def encode_dc(X_blocks):
    """Fórmula fechada do D&C: x_normalizado^2, validada contra o circuito
    real em _validate_dc_encoding(). Roda em qualquer dimensão, sem
    simular nenhum circuito."""
    norms = np.linalg.norm(X_blocks, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (X_blocks / norms) ** 2

def _validate_dc_encoding():
    dim = 8  # n_wires = 7, ainda barato para simular de verdade
    x = np.abs(rng.normal(size=(3, dim))) + 0.1  # valores positivos, evita zeros

    probs_circuit = _dc_circuit_probs(x)
    probs_closed = encode_dc(x)

    assert np.allclose(probs_circuit, probs_closed, atol=1e-6), (
        "D&C: fórmula fechada diverge do circuito real"
    )

print("Validando D&C (fórmula fechada vs circuito Ry+CSWAP real, dim=8)...")
_validate_dc_encoding()
print("OK: D&C validado.\n")

#%% =========================
# Loop experimental - raw, fft, dc
#
# =========================
results_dict = {}

for dim in TARGET_DIMS:

    print(f"\n>>> Dimensão {dim}")

    X_blocks, y_blocks_all = make_blocks(dim)

    modes = ["raw", "fft", "dc"]

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
            else:  # dc
                fft_block = np.abs(np.fft.fft(X_b, axis=1))
                X_features = encode_dc(fft_block)
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
styles = {"raw": "o-", "fft": "s-", "dc": "x--"}

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
plt.title("Accuracy vs Dimension - Raw / FFT / D&C (sobre módulo da FFT)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("accuracy_vs_dimension_v8.png", dpi=150, bbox_inches="tight")
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
plt.savefig("time_vs_dimension_v8.png", dpi=150, bbox_inches="tight")
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

pd.DataFrame(rows).to_csv("results_raw_fft_dc.csv", index=False)
print("Resultados salvos em results_raw_fft_dc.csv")
# %%
