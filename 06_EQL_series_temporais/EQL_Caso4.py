# %% ============================= IMPORTAÇÕES =============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------- ELM MULTI-SAÍDA ----------
class ELMRegressorMulti:
    def __init__(self, n_hidden=500, alpha=1e-3, random_state=42):
        self.n_hidden = n_hidden
        self.alpha = alpha
        np.random.seed(random_state)

    def fit(self, X, Y):
        N, d = X.shape
        self.W = np.random.randn(d, self.n_hidden)
        self.b = np.random.randn(self.n_hidden)
        H = np.tanh(X @ self.W + self.b)
        I = np.eye(self.n_hidden)
        self.beta = np.linalg.solve((H.T @ H + self.alpha * I), H.T @ Y)
        return self

    def predict(self, X):
        H = np.tanh(X @ self.W + self.b)
        return H @ self.beta


# %% ========================== CONFIGURAÇÕES ==============================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

EPOCHS = 60
BS = 64

early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)


# %% ========================== CARREGAMENTO ===============================
CSV_PATH = "EQL_STI_EQL_PBS.csv"   # troque para o arquivo desejado

df = pd.read_csv(CSV_PATH)
y_raw = df["power"].values.astype(float)

N = len(y_raw)
split = int(0.8 * N)

y_train = y_raw[:split]
y_test  = y_raw[split:]

scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1,1)).reshape(-1)
y_test_scaled  = scaler.transform(y_test.reshape(-1,1)).reshape(-1)


# %% =========================== FUNÇÕES ÚTEIS ==============================

def report_reg(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    med  = np.median(np.abs(y_true - y_pred))

    # tolerância ±3%
    tol = 0.03 * np.abs(y_true)
    acc = np.mean(np.abs(y_true - y_pred) <= tol)

    print(f"\n=== {name} ===")
    print(f"MAE       : {mae:.6e}")
    print(f"RMSE      : {rmse:.6e}")
    print(f"R²        : {r2:.4f}")
    print(f"Median AE : {med:.6e}")
    print(f"TolAcc±3% : {acc:.3f}")


def inv_scale(x):
    """inverte normalização e achata"""
    return scaler.inverse_transform(x.reshape(-1,1)).reshape(-1)


# %% ========================= CASO 4 — Dataset =============================

WINDOW4 = 15
H4 = 2
STRIDE4 = 2

series_all_scaled = np.concatenate([y_train_scaled, y_test_scaled])

def make_supervised_multi_stride(series, window, horizon, stride):
    X, Y, idx_first_target = [], [], []
    s = series.reshape(-1)
    max_start = len(s) - window - horizon + 1

    for i in range(0, max_start, stride):
        X.append(s[i:i+window])
        Y.append(s[i+window:i+window+horizon])
        idx_first_target.append(i+window)

    return np.array(X), np.array(Y), np.array(idx_first_target)

# treino
X4_train, Y4_train, _ = make_supervised_multi_stride(
    y_train_scaled, WINDOW4, H4, STRIDE4
)

# all -> teste
X4_all, Y4_all, idx_first = make_supervised_multi_stride(
    series_all_scaled, WINDOW4, H4, STRIDE4
)

mask_test = idx_first >= split
X4_test_scaled = X4_all[mask_test]
Y4_test_scaled = Y4_all[mask_test]
idx_first_test = idx_first[mask_test]

print("X4_train:", X4_train.shape, "| Y4_train:", Y4_train.shape)
print("X4_test :", X4_test_scaled.shape, "| Y4_test:", Y4_test_scaled.shape)

# prepara tensores para RNN/TCN
X4_train_rnn = X4_train.reshape(-1, WINDOW4, 1)
X4_test_rnn  = X4_test_scaled.reshape(-1, WINDOW4, 1)

y4_true = inv_scale(Y4_test_scaled.ravel())

idx_targets = []
for i in idx_first_test:
    idx_targets.append(i - split)
    idx_targets.append(i + 1 - split)
idx_targets = np.array(idx_targets)


# %% ========================= MODELOS — CASO 4 =============================

# MLP
mlp_c4 = Sequential([
    Dense(64, activation="relu", input_shape=(WINDOW4,)),
    Dense(32, activation="relu"),
    Dense(H4)
])
mlp_c4.compile(optimizer="adam", loss="mse")
mlp_c4.fit(X4_train, Y4_train, epochs=EPOCHS, batch_size=BS,
           validation_split=0.2, verbose=1, callbacks=[early, reduce])

# RNN
rnn_c4 = Sequential([
    SimpleRNN(64, activation="tanh", input_shape=(WINDOW4,1)),
    Dense(32, activation="relu"),
    Dense(H4)
])
rnn_c4.compile(optimizer="adam", loss="mse")
rnn_c4.fit(X4_train_rnn, Y4_train, epochs=EPOCHS, batch_size=BS,
           validation_split=0.2, verbose=1, callbacks=[early, reduce])

# LSTM
lstm_c4 = Sequential([
    LSTM(64, activation="tanh", input_shape=(WINDOW4,1)),
    Dense(32, activation="relu"),
    Dense(H4)
])
lstm_c4.compile(optimizer="adam", loss="mse")
lstm_c4.fit(X4_train_rnn, Y4_train, epochs=EPOCHS, batch_size=BS,
            validation_split=0.2, verbose=1, callbacks=[early, reduce])

# GRU
gru_c4 = Sequential([
    GRU(64, activation="tanh", input_shape=(WINDOW4,1)),
    Dense(32, activation="relu"),
    Dense(H4)
])
gru_c4.compile(optimizer="adam", loss="mse")
gru_c4.fit(X4_train_rnn, Y4_train, epochs=EPOCHS, batch_size=BS,
           validation_split=0.2, verbose=1, callbacks=[early, reduce])

# TCN
tcn_c4 = Sequential([
    Conv1D(64, kernel_size=3, padding="causal", activation="relu", input_shape=(WINDOW4,1)),
    Conv1D(32, kernel_size=3, padding="causal", activation="relu"),
    GlobalAveragePooling1D(),
    Dense(H4)
])
tcn_c4.compile(optimizer="adam", loss="mse")
tcn_c4.fit(X4_train_rnn, Y4_train, epochs=EPOCHS, batch_size=BS,
           validation_split=0.2, verbose=1, callbacks=[early, reduce])

# ELM
elm_c4 = ELMRegressorMulti(n_hidden=600, alpha=1e-2, random_state=SEED)
elm_c4.fit(X4_train, Y4_train)


# %% ============================ PREVISÕES ================================

y4_mlp_hat  = inv_scale( mlp_c4.predict(X4_test_scaled, verbose=0).ravel() )
y4_rnn_hat  = inv_scale( rnn_c4.predict(X4_test_rnn,   verbose=0).ravel() )
y4_lstm_hat = inv_scale(lstm_c4.predict(X4_test_rnn,   verbose=0).ravel() )
y4_gru_hat  = inv_scale( gru_c4.predict(X4_test_rnn,   verbose=0).ravel() )
y4_tcn_hat  = inv_scale( tcn_c4.predict(X4_test_rnn,   verbose=0).ravel() )
y4_elm_hat  = inv_scale( elm_c4.predict(X4_test_scaled).ravel() )


# %% ============================ MÉTRICAS ================================

print("\n=== CASO 4 — H=2, janela=15, stride=2 ===")
report_reg("MLP",  y4_true, y4_mlp_hat)
report_reg("RNN",  y4_true, y4_rnn_hat)
report_reg("LSTM", y4_true, y4_lstm_hat)
report_reg("GRU",  y4_true, y4_gru_hat)
report_reg("TCN",  y4_true, y4_tcn_hat)
report_reg("ELM",  y4_true, y4_elm_hat)


# %% ============================= PLOT ==========================
import matplotlib.pyplot as plt

preds = {
    "MLP":  y4_mlp_hat,
    "RNN":  y4_rnn_hat,
    "LSTM": y4_lstm_hat,
    "GRU":  y4_gru_hat,
    "TCN":  y4_tcn_hat,
    "ELM":  y4_elm_hat,
}

colors = {
    "MLP":  "#1f77b4",
    "RNN":  "#ff7f0e",
    "LSTM": "#2ca02c",
    "GRU":  "#d62728",
    "TCN":  "#9467bd",
    "ELM":  "#7fa6c9"
}

plt.figure(figsize=(12,5))

for i, (name, pred) in enumerate(preds.items()):
    plt.subplot(2, 3, i+1)
    
    # Curva real
    plt.plot(idx_targets, y4_true, label="True", color="black", linewidth=1.2)
    
    # Curva prevista
    plt.plot(idx_targets, pred, label=name, color=colors[name], linewidth=1.2)
    
    plt.title(f"{name} — Caso 4 (h=2, stride=2)")
    plt.xlabel("Índice (teste, pulando 2)")
    plt.ylabel("Potência")
    plt.grid(alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()


