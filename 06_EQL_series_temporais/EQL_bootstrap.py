# %% ===================== Imports & Setup =====================
import os, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers

# Reprodutibilidade
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ===================== Carregamento & Split =====================
CSV_PATH = "EQL_STI_EQL_PBS.csv"  # troque para qualquer dos 4 arquivos
df = pd.read_csv(CSV_PATH)
y_raw = df["power"].values.astype(float)

TEST_RATIO = 0.2
N = len(y_raw)
split = int((1 - TEST_RATIO) * N)
H_TEST = N - split          # tamanho real do teste
WINDOW = 15                 # janela para one-step
H_DIR  = min(200, H_TEST)   # horizonte direto treinado (evita explodir memória)

# ===== MODO RÁPIDO (apenas para a recursiva) =====
FAST_MODE = True
H_TEST_ROLL = min(H_TEST, 3000) if FAST_MODE else H_TEST

print(f"Arquivo: {os.path.basename(CSV_PATH)} | Total={N} | Train={split} | "
      f"Test={H_TEST} | WINDOW={WINDOW} | H_DIR={H_DIR} | H_TEST_ROLL={H_TEST_ROLL}")

# Scaling sem leakage
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_raw[:split].reshape(-1,1)).reshape(-1)
y_test_scaled  = scaler.transform(y_raw[split:].reshape(-1,1)).reshape(-1)
y_test_orig    = y_raw[split:]  # avaliação na escala original

# %% ===================== Helpers de Dataset =====================
def make_supervised(series, window):
    X, y = [], []
    s = series.reshape(-1)
    for i in range(len(s) - window):
        X.append(s[i:i+window]); y.append(s[i+window])
    return np.array(X), np.array(y)

def make_supervised_multi(series, window, horizon):
    X, Y = [], []
    s = series.reshape(-1)
    for i in range(len(s) - window - horizon + 1):
        X.append(s[i:i+window]); Y.append(s[i+window:i+window+horizon])
    return np.array(X), np.array(Y)

# One-step (teacher forcing)
X_train_1s, y_train_1s = make_supervised(y_train_scaled, WINDOW)
bridge = y_train_scaled[-WINDOW:]
test_with_bridge = np.concatenate([bridge, y_test_scaled])
X_test_1s, y_test_1s = make_supervised(test_with_bridge, WINDOW)

# Shapes p/ RNN/TCN
X_train_1s_rnn = X_train_1s.reshape(-1, WINDOW, 1)
X_test_1s_rnn  = X_test_1s.reshape(-1, WINDOW, 1)

# One-shot (direta): janela -> vetor de H_DIR
X_train_dir, Y_train_dir = make_supervised_multi(y_train_scaled, WINDOW, H_DIR)
X_last_window = y_train_scaled[-WINDOW:].reshape(1, WINDOW)
X_last_window_rnn = X_last_window.reshape(1, WINDOW, 1)

# %% ===================== Callbacks =====================
EPOCHS = 60
BS = 256
early  = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)

# %% ===================== MODELOS ONE-STEP (saída escalar) =====================
# MLP 1-step
mlp = Sequential([
    Dense(64, activation="relu", input_shape=(WINDOW,)),
    Dense(32, activation="relu"),
    Dense(1)
])
mlp.compile(optimizer="adam", loss="mse")
mlp.fit(X_train_1s, y_train_1s, epochs=EPOCHS, batch_size=BS, verbose=1,
        validation_split=0.2, callbacks=[early, reduce])

# RNN 1-step
rnn = Sequential([
    SimpleRNN(64, activation="tanh", input_shape=(WINDOW,1)),
    Dense(1)
])
rnn.compile(optimizer="adam", loss="mse")
rnn.fit(X_train_1s_rnn, y_train_1s, epochs=EPOCHS, batch_size=BS, verbose=1,
        validation_split=0.2, callbacks=[early, reduce])

# LSTM 1-step
lstm = Sequential([
    LSTM(64, activation="tanh", input_shape=(WINDOW,1)),
    Dense(1)
])
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(X_train_1s_rnn, y_train_1s, epochs=EPOCHS, batch_size=BS, verbose=1,
         validation_split=0.2, callbacks=[early, reduce])

# GRU 1-step
gru = Sequential([
    GRU(64, activation="tanh", input_shape=(WINDOW,1)),
    Dense(1)
])
gru.compile(optimizer="adam", loss="mse")
gru.fit(X_train_1s_rnn, y_train_1s, epochs=EPOCHS, batch_size=BS, verbose=1,
        validation_split=0.2, callbacks=[early, reduce])

# TCN 1-step
def build_tcn_1s(input_shape):
    return Sequential([
        layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu", input_shape=input_shape),
        layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu"),
        layers.GlobalAveragePooling1D(),
        Dense(1)
    ])

tcn = build_tcn_1s((WINDOW,1))
tcn.compile(optimizer="adam", loss="mse")
tcn.fit(X_train_1s_rnn, y_train_1s, epochs=EPOCHS, batch_size=BS, verbose=1,
        validation_split=0.2, callbacks=[early, reduce])

# ELM 1-step
class ELMRegressor:
    def __init__(self, n_hidden=600, alpha=1e-2, random_state=42):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.random_state = np.random.RandomState(random_state)
        self.W = None; self.b = None
        self.out = Ridge(alpha=self.alpha)
    def _hidden(self, X):
        return np.tanh(X @ self.W + self.b)
    def fit(self, X, y):
        self.W = self.random_state.normal(0, 1, size=(X.shape[1], self.n_hidden))
        self.b = self.random_state.normal(0, 1, size=(self.n_hidden,))
        Hm = self._hidden(X)
        self.out.fit(Hm, y)
        return self
    def predict(self, X):
        return self.out.predict(self._hidden(X))

elm_1s = ELMRegressor(n_hidden=600, alpha=1e-2, random_state=SEED).fit(X_train_1s, y_train_1s)

# %% ===================== MODELOS DIRETOS (saída vetor H_DIR) =====================
def build_mlp_dir():
    m = Sequential([
        Dense(128, activation="relu", input_shape=(WINDOW,)),
        Dense(64, activation="relu"),
        Dense(H_DIR)
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def build_rnn_dir():
    m = Sequential([
        SimpleRNN(128, activation="tanh", input_shape=(WINDOW,1)),
        Dense(64, activation="relu"),
        Dense(H_DIR)
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def build_lstm_dir():
    m = Sequential([
        LSTM(128, activation="tanh", input_shape=(WINDOW,1)),
        Dense(64, activation="relu"),
        Dense(H_DIR)
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def build_gru_dir():
    m = Sequential([
        GRU(128, activation="tanh", input_shape=(WINDOW,1)),
        Dense(64, activation="relu"),
        Dense(H_DIR)
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def build_tcn_dir(input_shape, H):
    m = Sequential([
        layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu", input_shape=input_shape),
        layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu"),
        layers.GlobalAveragePooling1D(),
        Dense(H)
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

class ELMRegressorMulti:
    def __init__(self, n_hidden=600, alpha=1e-2, random_state=42):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.random_state = np.random.RandomState(random_state)
        self.W = None; self.b = None
        self.out = Ridge(alpha=self.alpha)
    def _hidden(self, X):
        return np.tanh(X @ self.W + self.b)
    def fit(self, X, Y):
        self.W = self.random_state.normal(0, 1, size=(X.shape[1], self.n_hidden))
        self.b = self.random_state.normal(0, 1, size=(self.n_hidden,))
        Hm = self._hidden(X)
        self.out.fit(Hm, Y)
        return self
    def predict(self, X):
        return self.out.predict(self._hidden(X))

# Treino base (sem bootstrap) dos diretos
mlp_dir  = build_mlp_dir()
mlp_dir.fit(X_train_dir, Y_train_dir, epochs=EPOCHS, batch_size=128, verbose=1,
            validation_split=0.2, callbacks=[early, reduce])

rnn_dir  = build_rnn_dir()
rnn_dir.fit(X_train_dir.reshape(-1, WINDOW, 1), Y_train_dir, epochs=EPOCHS, batch_size=128, verbose=1,
            validation_split=0.2, callbacks=[early, reduce])

lstm_dir = build_lstm_dir()
lstm_dir.fit(X_train_dir.reshape(-1, WINDOW, 1), Y_train_dir, epochs=EPOCHS, batch_size=128, verbose=1,
             validation_split=0.2, callbacks=[early, reduce])

gru_dir  = build_gru_dir()
gru_dir.fit(X_train_dir.reshape(-1, WINDOW, 1), Y_train_dir, epochs=EPOCHS, batch_size=128, verbose=1,
            validation_split=0.2, callbacks=[early, reduce])

tcn_dir  = build_tcn_dir((WINDOW,1), H_DIR)
tcn_dir.fit(X_train_dir.reshape(-1, WINDOW, 1), Y_train_dir, epochs=EPOCHS, batch_size=128, verbose=1,
            validation_split=0.2, callbacks=[early, reduce])

elm_dir = ELMRegressorMulti(n_hidden=600, alpha=1e-2, random_state=SEED).fit(X_train_dir, Y_train_dir)

# %% ===================== Direta em blocos (cobre todo o teste) =====================
def _safe_predict(model, x):
    try:
        return model.predict(x, verbose=0)
    except TypeError:
        return model.predict(x)

def predict_direct_blocks(model, seed_window_scaled, total_steps, block_size=200, is_rnn=False):
    out = []
    seq = seed_window_scaled.copy().tolist()
    steps_left = total_steps
    while steps_left > 0:
        b = min(block_size, steps_left)
        x = (np.array(seq[-WINDOW:]).reshape(1, WINDOW, 1)
             if is_rnn else np.array(seq[-WINDOW:]).reshape(1, WINDOW))
        yhat_block = _safe_predict(model, x).reshape(-1)[:b]
        out.extend(yhat_block)
        seq.extend(yhat_block)  # propaga erro ENTRE blocos
        steps_left -= b
    return np.array(out)

seed_scaled = y_train_scaled[-WINDOW:]

yhat_dir_mlp_s  = predict_direct_blocks(mlp_dir,  seed_scaled, H_TEST, block_size=H_DIR, is_rnn=False)
yhat_dir_rnn_s  = predict_direct_blocks(rnn_dir,  seed_scaled, H_TEST, block_size=H_DIR, is_rnn=True)
yhat_dir_lstm_s = predict_direct_blocks(lstm_dir, seed_scaled, H_TEST, block_size=H_DIR, is_rnn=True)
yhat_dir_gru_s  = predict_direct_blocks(gru_dir,  seed_scaled, H_TEST, block_size=H_DIR, is_rnn=True)
yhat_dir_tcn_s  = predict_direct_blocks(tcn_dir,  seed_scaled, H_TEST, block_size=H_DIR, is_rnn=True)
yhat_dir_elm_s  = predict_direct_blocks(elm_dir,  seed_scaled, H_TEST, block_size=H_DIR, is_rnn=False)

def inv(x): 
    return scaler.inverse_transform(x.reshape(-1,1)).reshape(-1)

yhat_dir_mlp  = inv(yhat_dir_mlp_s)
yhat_dir_rnn  = inv(yhat_dir_rnn_s)
yhat_dir_lstm = inv(yhat_dir_lstm_s)
yhat_dir_gru  = inv(yhat_dir_gru_s)
yhat_dir_tcn  = inv(yhat_dir_tcn_s)
yhat_dir_elm  = inv(yhat_dir_elm_s)

# %% ===================== Recursiva RÁPIDA =====================
tf.keras.backend.set_floatx('float32')

def rollout_fast_keras(model, seed_scaled_np, steps, window=50, is_rnn=False):
    seed = tf.convert_to_tensor(seed_scaled_np, dtype=tf.float32)
    @tf.function
    def _roll(seed, steps):
        out = tf.TensorArray(dtype=tf.float32, size=steps)
        seq = seed
        def cond(i, seq, out): return i < steps
        def body(i, seq, out):
            x = seq[-window:]
            x = tf.reshape(x, (1, window, 1) if is_rnn else (1, window))
            y = model(x, training=False)
            y = tf.reshape(y, ())
            out = out.write(i, y)
            seq = tf.concat([seq, [y]], axis=0)
            return i+1, seq, out
        i0 = tf.constant(0)
        _, _, out = tf.while_loop(
            cond, body, [i0, seed, out],
            shape_invariants=[tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape(None)],
            parallel_iterations=1, swap_memory=True
        )
        return out.stack()
    return _roll(seed, tf.constant(steps, dtype=tf.int32)).numpy()

def rollout_fast_sklearn(model, seed_scaled_np, steps, window=50):
    seq = np.empty(window + steps, dtype=np.float32)
    seq[:window] = seed_scaled_np.astype(np.float32)
    out = np.empty(steps, dtype=np.float32)
    x_flat = np.empty((1, window), dtype=np.float32)
    for i in range(steps):
        x_flat[0, :] = seq[i:i+window]
        yhat = model.predict(x_flat).reshape(-1)[0]
        out[i] = yhat
        seq[i+window] = yhat
    return out

seed_scaled = y_train_scaled[-WINDOW:]

yhat_rec_mlp_s  = rollout_fast_keras(mlp,  seed_scaled, H_TEST_ROLL, window=WINDOW, is_rnn=False)
yhat_rec_rnn_s  = rollout_fast_keras(rnn,  seed_scaled, H_TEST_ROLL, window=WINDOW, is_rnn=True)
yhat_rec_lstm_s = rollout_fast_keras(lstm, seed_scaled, H_TEST_ROLL, window=WINDOW, is_rnn=True)
yhat_rec_gru_s  = rollout_fast_keras(gru,  seed_scaled, H_TEST_ROLL, window=WINDOW, is_rnn=True)
yhat_rec_tcn_s  = rollout_fast_keras(tcn,  seed_scaled, H_TEST_ROLL, window=WINDOW, is_rnn=True)
yhat_rec_elm_s  = rollout_fast_sklearn(elm_1s, seed_scaled, H_TEST_ROLL, window=WINDOW)

yhat_rec_mlp  = inv(yhat_rec_mlp_s)
yhat_rec_rnn  = inv(yhat_rec_rnn_s)
yhat_rec_lstm = inv(yhat_rec_lstm_s)
yhat_rec_gru  = inv(yhat_rec_gru_s)
yhat_rec_tcn  = inv(yhat_rec_tcn_s)
yhat_rec_elm  = inv(yhat_rec_elm_s)

y_test_rec_slice = y_test_orig[:H_TEST_ROLL]

# %% ===================== Avaliação básica sem bootstrap =====================
def report_reg(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    med  = np.median(np.abs(y_true - y_pred))
    y_t_s = scaler.transform(y_true.reshape(-1,1)).reshape(-1)
    y_p_s = scaler.transform(y_pred.reshape(-1,1)).reshape(-1)
    tol   = np.mean(np.abs(y_t_s - y_p_s) < 0.03)
    print(f"\n=== {name} ===")
    print(f"MAE       : {mae:.6e}")
    print(f"RMSE      : {rmse:.6e}")
    print(f"R²        : {r2:.4f}")
    print(f"Median AE : {med:.6e}")
    print(f"TolAcc±3% : {tol:.3f}")

# Diretas
report_reg("Direta MLP",  y_test_orig, yhat_dir_mlp)
report_reg("Direta RNN",  y_test_orig, yhat_dir_rnn)
report_reg("Direta LSTM", y_test_orig, yhat_dir_lstm)
report_reg("Direta GRU",  y_test_orig, yhat_dir_gru)
report_reg("Direta TCN",  y_test_orig, yhat_dir_tcn)
report_reg("Direta ELM",  y_test_orig, yhat_dir_elm)

# Recursivas
report_reg("Recursiva MLP",  y_test_rec_slice, yhat_rec_mlp)
report_reg("Recursiva RNN",  y_test_rec_slice, yhat_rec_rnn)
report_reg("Recursiva LSTM", y_test_rec_slice, yhat_rec_lstm)
report_reg("Recursiva GRU",  y_test_rec_slice, yhat_rec_gru)
report_reg("Recursiva TCN",  y_test_rec_slice, yhat_rec_tcn)
report_reg("Recursiva ELM",  y_test_rec_slice, yhat_rec_elm)

# %% ===================== Plots básicos =====================
# Direta
plt.figure(figsize=(11,4))
x = np.arange(len(y_test_orig))
plt.plot(x, y_test_orig, label="True (test)", color="#1f77b4", linewidth=1)
plt.plot(x, yhat_dir_mlp,  label="MLP",  color="#ff7f0e", alpha=0.9)
plt.plot(x, yhat_dir_rnn,  label="RNN",  color="#2ca02c", alpha=0.9)
plt.plot(x, yhat_dir_lstm, label="LSTM", color="#d62728", alpha=0.9)
plt.plot(x, yhat_dir_gru,  label="GRU",  color="#9467bd", alpha=0.9)
plt.plot(x, yhat_dir_tcn,  label="TCN",  color="#8c564b", alpha=0.9)
plt.plot(x, yhat_dir_elm,  label="ELM",  color="#e377c2", alpha=0.9)
plt.title(f"Previsão Direta — {os.path.basename(CSV_PATH)}")
plt.xlabel("Índice (região de teste)"); plt.ylabel("Potência (escala original)")
plt.legend(ncol=1, loc="upper left", frameon=True)
plt.tight_layout(); plt.savefig("previsao_direta.png", dpi=300, bbox_inches="tight"); plt.show()

# Recursiva
plt.figure(figsize=(11,4))
xr = np.arange(len(y_test_rec_slice))
plt.plot(xr, y_test_rec_slice, label="True (test)", color="#1f77b4", linewidth=1)
plt.plot(xr, yhat_rec_mlp,  label="MLP",  color="#ff7f0e", alpha=0.9)
plt.plot(xr, yhat_rec_rnn,  label="RNN",  color="#2ca02c", alpha=0.9)
plt.plot(xr, yhat_rec_lstm, label="LSTM", color="#d62728", alpha=0.9)
plt.plot(xr, yhat_rec_gru,  label="GRU",  color="#9467bd", alpha=0.9)
plt.plot(xr, yhat_rec_tcn,  label="TCN", color="#8c564b", alpha=0.9)
plt.plot(xr, yhat_rec_elm,  label="ELM",  color="#e377c2", alpha=0.9)
plt.title(f"Previsão Recursiva — {os.path.basename(CSV_PATH)}")
plt.xlabel("Índice (início do teste)"); plt.ylabel("Potência (escala original)")
plt.legend(ncol=1, loc="upper left", frameon=True)
plt.tight_layout(); plt.savefig("previsao_recursiva_slice.png", dpi=300, bbox_inches="tight"); plt.show()

# %% ===================== BOOTSTRAP TIPO B (séries completas) =====================
# 1) bootstrap da série de treino (moving block bootstrap)
# 2) reconstrói janelas
# 3) re-treina os 6 modelos diretos em cada réplica
# 4) prevê toda a região de teste
# 5) gera média e IC das previsões

def moving_block_bootstrap(series, block_len, random_state=None):
    """
    Bootstrap em blocos para séries temporais.
    Gera uma série bootstrap com o mesmo tamanho da original,
    preservando alguma dependência local.
    """
    rng = np.random.default_rng(random_state)
    n = len(series)
    n_blocks = int(np.ceil(n / block_len))
    max_start = n - block_len
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    out = []
    for s in starts:
        out.extend(series[s:s+block_len])
    return np.array(out[:n])

N_BOOT = 20           # número de réplicas bootstrap (ajuste conforme o tempo disponível)
BLOCK_LEN = 50       # tamanho do bloco (pode ser ~ WINDOW ou um pouco maior)

# Matrizes para guardar previsões bootstrap (escala original)
boot_preds_mlp  = np.zeros((N_BOOT, H_TEST))
boot_preds_rnn  = np.zeros((N_BOOT, H_TEST))
boot_preds_lstm = np.zeros((N_BOOT, H_TEST))
boot_preds_gru  = np.zeros((N_BOOT, H_TEST))
boot_preds_tcn  = np.zeros((N_BOOT, H_TEST))
boot_preds_elm  = np.zeros((N_BOOT, H_TEST))

for b in range(N_BOOT):
    print(f"\n>>> Bootstrap {b+1}/{N_BOOT}")
    # 1) Série de treino bootstrap (em escala normalizada)
    y_train_boot = moving_block_bootstrap(y_train_scaled, BLOCK_LEN, random_state=SEED + b)

    # 2) Reconstruir janelas diretas (janela -> vetor H_DIR)
    X_train_dir_b, Y_train_dir_b = make_supervised_multi(y_train_boot, WINDOW, H_DIR)

    # 3) Re-treinar modelos diretos nesta réplica
    mlp_b  = build_mlp_dir()
    rnn_b  = build_rnn_dir()
    lstm_b = build_lstm_dir()
    gru_b  = build_gru_dir()
    tcn_b  = build_tcn_dir((WINDOW,1), H_DIR)
    elm_b  = ELMRegressorMulti(n_hidden=600, alpha=1e-2, random_state=SEED + b)

    EPOCHS_BOOT = 40

    mlp_b.fit(X_train_dir_b, Y_train_dir_b, epochs=EPOCHS_BOOT, batch_size=128,
              verbose=0, validation_split=0.2, callbacks=[early, reduce])

    X_train_dir_b_rnn = X_train_dir_b.reshape(-1, WINDOW, 1)
    rnn_b.fit(X_train_dir_b_rnn, Y_train_dir_b, epochs=EPOCHS_BOOT, batch_size=128,
              verbose=0, validation_split=0.2, callbacks=[early, reduce])

    lstm_b.fit(X_train_dir_b_rnn, Y_train_dir_b, epochs=EPOCHS_BOOT, batch_size=128,
               verbose=0, validation_split=0.2, callbacks=[early, reduce])

    gru_b.fit(X_train_dir_b_rnn, Y_train_dir_b, epochs=EPOCHS_BOOT, batch_size=128,
              verbose=0, validation_split=0.2, callbacks=[early, reduce])

    tcn_b.fit(X_train_dir_b_rnn, Y_train_dir_b, epochs=EPOCHS_BOOT, batch_size=128,
              verbose=0, validation_split=0.2, callbacks=[early, reduce])

    elm_b.fit(X_train_dir_b, Y_train_dir_b)

    # 4) Prever toda a região de teste para cada modelo
    seed_boot = y_train_boot[-WINDOW:]

    yhat_mlp_s  = predict_direct_blocks(mlp_b,  seed_boot, H_TEST, block_size=H_DIR, is_rnn=False)
    yhat_rnn_s  = predict_direct_blocks(rnn_b,  seed_boot, H_TEST, block_size=H_DIR, is_rnn=True)
    yhat_lstm_s = predict_direct_blocks(lstm_b, seed_boot, H_TEST, block_size=H_DIR, is_rnn=True)
    yhat_gru_s  = predict_direct_blocks(gru_b,  seed_boot, H_TEST, block_size=H_DIR, is_rnn=True)
    yhat_tcn_s  = predict_direct_blocks(tcn_b,  seed_boot, H_TEST, block_size=H_DIR, is_rnn=True)
    yhat_elm_s  = predict_direct_blocks(elm_b,  seed_boot, H_TEST, block_size=H_DIR, is_rnn=False)

    boot_preds_mlp[b, :]  = inv(yhat_mlp_s)
    boot_preds_rnn[b, :]  = inv(yhat_rnn_s)
    boot_preds_lstm[b, :] = inv(yhat_lstm_s)
    boot_preds_gru[b, :]  = inv(yhat_gru_s)
    boot_preds_tcn[b, :]  = inv(yhat_tcn_s)
    boot_preds_elm[b, :]  = inv(yhat_elm_s)

# %% ===================== Estatísticas do Bootstrap =====================
def summarize_bootstrap(name, boot_matrix, y_true):
    """
    boot_matrix: shape (N_BOOT, H_TEST)
    """
    mean_pred = boot_matrix.mean(axis=0)
    lower = np.percentile(boot_matrix, 5, axis=0)
    upper = np.percentile(boot_matrix, 95, axis=0)

    print(f"\n### {name} — Métrica na média das réplicas ###")
    report_reg(name + " (média bootstrap)", y_true, mean_pred)
    return mean_pred, lower, upper

mean_mlp,  lo_mlp,  up_mlp  = summarize_bootstrap("Direta MLP (bootstrap)",  boot_preds_mlp,  y_test_orig)
mean_rnn,  lo_rnn,  up_rnn  = summarize_bootstrap("Direta RNN (bootstrap)",  boot_preds_rnn,  y_test_orig)
mean_lstm, lo_lstm, up_lstm = summarize_bootstrap("Direta LSTM (bootstrap)", boot_preds_lstm, y_test_orig)
mean_gru,  lo_gru,  up_gru  = summarize_bootstrap("Direta GRU (bootstrap)",  boot_preds_gru,  y_test_orig)
mean_tcn,  lo_tcn,  up_tcn  = summarize_bootstrap("Direta TCN (bootstrap)",  boot_preds_tcn,  y_test_orig)
mean_elm,  lo_elm,  up_elm  = summarize_bootstrap("Direta ELM (bootstrap)",  boot_preds_elm,  y_test_orig)

# %% ===================== Plot Exemplo de IC (ex.: LSTM) =====================
plt.figure(figsize=(11,4))
x = np.arange(len(y_test_orig))
plt.plot(x, y_test_orig, label="True (test)", color="#1f77b4", linewidth=1)
plt.plot(x, mean_lstm, label="LSTM (média bootstrap)", color="#d62728", linewidth=1.2)
plt.fill_between(x, lo_lstm, up_lstm, color="#d62728", alpha=0.2, label="IC 90% (bootstrap)")
plt.title(f"Bootstrap — LSTM Direto — {os.path.basename(CSV_PATH)}")
plt.xlabel("Índice (região de teste)")
plt.ylabel("Potência (escala original)")
plt.legend(loc="upper left", frameon=True)
plt.tight_layout()
plt.savefig("bootstrap_lstm_direto_ic90.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
