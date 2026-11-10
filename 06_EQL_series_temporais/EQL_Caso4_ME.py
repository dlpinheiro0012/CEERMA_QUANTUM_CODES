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

# %% ===================== Configuração do Caso 4 =====================
CSV_PATH = "EQL_STI_EQL_Ptotal.csv"  # troque para qualquer arquivo
WINDOW4  = 15                        # tamanho da janela (do PACF)
HORIZON4 = 2                         # prevemos 2 passos à frente
STRIDE4  = 2                         # janelas avançam de 2 em 2
TEST_RATIO = 0.2

# %% ===================== Carregamento & Split =====================
df = pd.read_csv(CSV_PATH)
y_raw = df["power"].values.astype(float)

N = len(y_raw)
split = int((1 - TEST_RATIO) * N)
H_TEST = N - split

print(f"Arquivo: {os.path.basename(CSV_PATH)} | Total={N} | Train={split} | Test={H_TEST}")

# Scaling sem leakage
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_raw[:split].reshape(-1,1)).reshape(-1)
y_test_scaled  = scaler.transform(y_raw[split:].reshape(-1,1)).reshape(-1)
y_test_orig    = y_raw[split:]  # apenas para referência

# utilitário para voltar à escala original
def inv_arr(x):
    return scaler.inverse_transform(x.reshape(-1,1)).reshape(-1)

# %% ===================== Helpers de Dataset (Caso 4) =====================
def make_supervised_h2_stride(series, window, horizon=2, stride=2):
    """
    Para cada índice i, gera:
      X[i] = série[i : i+window]
      Y[i] = série[i+window : i+window+horizon]
    e avança i de 'stride' em 'stride'.
    """
    X, Y = [], []
    s = series.reshape(-1)
    i = 0
    while i + window + horizon <= len(s):
        X.append(s[i:i+window])
        Y.append(s[i+window:i+window+horizon])
        i += stride
    return np.array(X), np.array(Y)

# Conjuntos de treino (apenas sobre a região de treino)
X4_train_flat, Y4_train = make_supervised_h2_stride(
    y_train_scaled, WINDOW4, horizon=HORIZON4, stride=STRIDE4
)

# Região de teste com “ponte” (última janela do treino + teste)
bridge = y_train_scaled[-WINDOW4:]
test_with_bridge = np.concatenate([bridge, y_test_scaled])

X4_test_flat, Y4_test = make_supervised_h2_stride(
    test_with_bridge, WINDOW4, horizon=HORIZON4, stride=STRIDE4
)

# Para métrica vamos usar apenas o PRIMEIRO passo do horizonte
y4_true_s = Y4_test[:, 0]      # escala normalizada
y4_true   = inv_arr(y4_true_s) # escala original

print("X4_train_flat:", X4_train_flat.shape)
print("Y4_train     :", Y4_train.shape)
print("X4_test_flat :", X4_test_flat.shape)
print("Y4_test      :", Y4_test.shape)

# Versões 3D para RNN/GRU/LSTM/TCN
X4_train_seq = X4_train_flat.reshape(-1, WINDOW4, 1)
X4_test_seq  = X4_test_flat.reshape(-1, WINDOW4, 1)

# %% ===================== Callbacks =====================
EPOCHS = 60
BS = 256
early  = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)

# %% ===================== Modelo ELM multi-output =====================
class ELMRegressorMulti:
    def __init__(self, n_hidden=600, alpha=1e-2, random_state=42):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.random_state = np.random.RandomState(random_state)
        self.W = None
        self.b = None
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

# %% ===================== MODELOS (Caso 4, saída com 2 neurônios) =====================

# MLP
mlp4 = Sequential([
    Dense(128, activation="relu", input_shape=(WINDOW4,)),
    Dense(64, activation="relu"),
    Dense(HORIZON4)
])
mlp4.compile(optimizer="adam", loss="mse")
mlp4.fit(X4_train_flat, Y4_train, epochs=EPOCHS, batch_size=BS,
         validation_split=0.2, callbacks=[early, reduce], verbose=1)

# RNN
rnn4 = Sequential([
    SimpleRNN(128, activation="tanh", input_shape=(WINDOW4,1)),
    Dense(64, activation="relu"),
    Dense(HORIZON4)
])
rnn4.compile(optimizer="adam", loss="mse")
rnn4.fit(X4_train_seq, Y4_train, epochs=EPOCHS, batch_size=BS,
         validation_split=0.2, callbacks=[early, reduce], verbose=1)

# LSTM
lstm4 = Sequential([
    LSTM(128, activation="tanh", input_shape=(WINDOW4,1)),
    Dense(64, activation="relu"),
    Dense(HORIZON4)
])
lstm4.compile(optimizer="adam", loss="mse")
lstm4.fit(X4_train_seq, Y4_train, epochs=EPOCHS, batch_size=BS,
          validation_split=0.2, callbacks=[early, reduce], verbose=1)

# GRU
gru4 = Sequential([
    GRU(128, activation="tanh", input_shape=(WINDOW4,1)),
    Dense(64, activation="relu"),
    Dense(HORIZON4)
])
gru4.compile(optimizer="adam", loss="mse")
gru4.fit(X4_train_seq, Y4_train, epochs=EPOCHS, batch_size=BS,
         validation_split=0.2, callbacks=[early, reduce], verbose=1)

# TCN
def build_tcn_case4(input_shape, horizon):
    return Sequential([
        layers.Conv1D(64, kernel_size=3, padding="causal",
                      activation="relu", input_shape=input_shape),
        layers.Conv1D(32, kernel_size=3, padding="causal",
                      activation="relu"),
        layers.GlobalAveragePooling1D(),
        Dense(horizon)
    ])

tcn4 = build_tcn_case4((WINDOW4,1), HORIZON4)
tcn4.compile(optimizer="adam", loss="mse")
tcn4.fit(X4_train_seq, Y4_train, epochs=EPOCHS, batch_size=BS,
         validation_split=0.2, callbacks=[early, reduce], verbose=1)

# ELM
elm4 = ELMRegressorMulti(n_hidden=600, alpha=1e-2, random_state=SEED)
elm4.fit(X4_train_flat, Y4_train)

# %% ===================== Avaliação Direta (sem bootstrap) =====================
def report_reg_case4(name, y_true_orig, y_pred_s_firststep):
    y_pred_orig = inv_arr(y_pred_s_firststep)
    mae  = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = math.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2   = r2_score(y_true_orig, y_pred_orig)
    med  = np.median(np.abs(y_true_orig - y_pred_orig))

    y_t_s = scaler.transform(y_true_orig.reshape(-1,1)).reshape(-1)
    y_p_s = scaler.transform(y_pred_orig.reshape(-1,1)).reshape(-1)
    tol   = np.mean(np.abs(y_t_s - y_p_s) < 0.03)

    print(f"\n=== Caso 4 — {name} ===")
    print(f"MAE       : {mae:.6e}")
    print(f"RMSE      : {rmse:.6e}")
    print(f"R²        : {r2:.4f}")
    print(f"Median AE : {med:.6e}")
    print(f"TolAcc±3% : {tol:.3f}")

# Previsões em escala normalizada, usando só o primeiro passo do horizonte
mlp4_pred_s  = mlp4.predict(X4_test_flat, verbose=0)[:,0]
rnn4_pred_s  = rnn4.predict(X4_test_seq,  verbose=0)[:,0]
lstm4_pred_s = lstm4.predict(X4_test_seq, verbose=0)[:,0]
gru4_pred_s  = gru4.predict(X4_test_seq,  verbose=0)[:,0]
tcn4_pred_s  = tcn4.predict(X4_test_seq,  verbose=0)[:,0]
elm4_pred_s  = elm4.predict(X4_test_flat)[:,0]

report_reg_case4("MLP",  y4_true, mlp4_pred_s)
report_reg_case4("RNN",  y4_true, rnn4_pred_s)
report_reg_case4("LSTM", y4_true, lstm4_pred_s)
report_reg_case4("GRU",  y4_true, gru4_pred_s)
report_reg_case4("TCN",  y4_true, tcn4_pred_s)
report_reg_case4("ELM",  y4_true, elm4_pred_s)

# %% ===================== Bootstrap de Máxima Entropia (Caso 4) =====================

def meboot_window(x, B=100, seed=None):
    """
    Bootstrap de máxima entropia (versão simplificada) para uma janela 1D.
    x: array 1D (janela, em escala normalizada)
    B: número de réplicas bootstrap
    retorna: array (B, len(x)) com as janelas bootstrapeadas
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, n, size=(B, n))
        return x[idx]

    rng = np.random.RandomState(seed)

    xs = np.sort(x)
    z = np.empty(n + 1)
    z[1:n] = (xs[:-1] + xs[1:]) / 2.0
    z[0] = xs[0] - (xs[1] - xs[0])
    z[n] = xs[-1] + (xs[-1] - xs[-2])

    idx_sorted = np.argsort(x)
    out = np.empty((B, n))

    for b in range(B):
        u = np.sort(rng.rand(n))
        y_sorted = np.empty(n)
        for j, uj in enumerate(u):
            k = int(np.floor(uj * n))
            if k == n:
                k = n - 1
            u_local = uj * n - k
            y_sorted[j] = z[k] + u_local * (z[k + 1] - z[k])

        y_star = np.empty(n)
        for j, pos in enumerate(idx_sorted):
            y_star[pos] = y_sorted[j]

        out[b] = y_star

    return out

def bootstrap_case4_me(model, X_flat, window, B=100, is_rnn=False):
    """
    Faz bootstrap de máxima entropia janela a janela (Caso 4, horizonte=2, stride=2)
    e retorna média e IC 90% das previsões para o PRIMEIRO passo do horizonte.
    """
    N = X_flat.shape[0]
    mean_pred = np.empty(N)
    ci_low = np.empty(N)
    ci_high = np.empty(N)

    for i in range(N):
        w = X_flat[i]           # janela (1D, normalizada)
        boots = meboot_window(w, B=B)  # (B, window)

        if is_rnn:
            X_boot = boots.reshape(B, window, 1)
        else:
            X_boot = boots

        y_boot = model.predict(X_boot, verbose=0)[:, 0]  # primeiro passo

        mean_pred[i] = y_boot.mean()
        ci_low[i]    = np.quantile(y_boot, 0.05)
        ci_high[i]   = np.quantile(y_boot, 0.95)

    return mean_pred, ci_low, ci_high

# Número de réplicas bootstrap
B_ME = 10

#%% ----- MLP -----
mlp4_mean_s, mlp4_lo_s, mlp4_hi_s = bootstrap_case4_me(
    mlp4, X4_test_flat, window=WINDOW4, B=B_ME, is_rnn=False
)

#%% ----- RNN -----
rnn4_mean_s, rnn4_lo_s, rnn4_hi_s = bootstrap_case4_me(
    rnn4, X4_test_flat, window=WINDOW4, B=B_ME, is_rnn=True
)

#%% ----- LSTM -----
lstm4_mean_s, lstm4_lo_s, lstm4_hi_s = bootstrap_case4_me(
    lstm4, X4_test_flat, window=WINDOW4, B=B_ME, is_rnn=True
)

#%% ----- GRU -----
gru4_mean_s, gru4_lo_s, gru4_hi_s = bootstrap_case4_me(
    gru4, X4_test_flat, window=WINDOW4, B=B_ME, is_rnn=True
)

#%% ----- TCN -----
tcn4_mean_s, tcn4_lo_s, tcn4_hi_s = bootstrap_case4_me(
    tcn4, X4_test_flat, window=WINDOW4, B=B_ME, is_rnn=True
)

#%%
class ELMWrapper:
    def __init__(self, base_model):
        self.base_model = base_model

    def predict(self, X, verbose=0):
        # ignora o verbose e chama o ELM normal
        return self.base_model.predict(X)


#%% ----- ELM -----
elm4_wrapped = ELMWrapper(elm4)

elm4_mean_s, elm4_lo_s, elm4_hi_s = bootstrap_case4_me(
    elm4_wrapped, X4_test_flat, window=WINDOW4, B=B_ME, is_rnn=False
)

#%% Convertendo médias e ICs para escala original
mlp4_mean  = inv_arr(mlp4_mean_s)
mlp4_lo    = inv_arr(mlp4_lo_s)
mlp4_hi    = inv_arr(mlp4_hi_s)

rnn4_mean  = inv_arr(rnn4_mean_s)
rnn4_lo    = inv_arr(rnn4_lo_s)
rnn4_hi    = inv_arr(rnn4_hi_s)

lstm4_mean = inv_arr(lstm4_mean_s)
lstm4_lo   = inv_arr(lstm4_lo_s)
lstm4_hi   = inv_arr(lstm4_hi_s)

gru4_mean  = inv_arr(gru4_mean_s)
gru4_lo    = inv_arr(gru4_lo_s)
gru4_hi    = inv_arr(gru4_hi_s)

tcn4_mean  = inv_arr(tcn4_mean_s)
tcn4_lo    = inv_arr(tcn4_lo_s)
tcn4_hi    = inv_arr(tcn4_hi_s)

elm4_mean  = inv_arr(elm4_mean_s)
elm4_lo    = inv_arr(elm4_lo_s)
elm4_hi    = inv_arr(elm4_hi_s)

# %% ===================== Métricas usando a MÉDIA bootstrap (sem IC) =====================
def report_reg_case4_boot(name, y_true_orig, y_pred_orig):
    mae  = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = math.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2   = r2_score(y_true_orig, y_pred_orig)
    med  = np.median(np.abs(y_true_orig - y_pred_orig))

    y_t_s = scaler.transform(y_true_orig.reshape(-1,1)).reshape(-1)
    y_p_s = scaler.transform(y_pred_orig.reshape(-1,1)).reshape(-1)
    tol   = np.mean(np.abs(y_t_s - y_p_s) < 0.03)

    print(f"\n=== Caso 4 — {name} (ME-bootstrap) ===")
    print(f"MAE       : {mae:.6e}")
    print(f"RMSE      : {rmse:.6e}")
    print(f"R²        : {r2:.4f}")
    print(f"Median AE : {med:.6e}")
    print(f"TolAcc±3% : {tol:.3f}")

report_reg_case4_boot("MLP",  y4_true, mlp4_mean)
report_reg_case4_boot("RNN",  y4_true, rnn4_mean)
report_reg_case4_boot("LSTM", y4_true, lstm4_mean)
report_reg_case4_boot("GRU",  y4_true, gru4_mean)
report_reg_case4_boot("TCN",  y4_true, tcn4_mean)
report_reg_case4_boot("ELM",  y4_true, elm4_mean)

# %% ===================== Plots separados por modelo (sem IC, só real x previsão) =====================
x_axis = np.arange(len(y4_true))

models_boot_mean = {
    "MLP":  mlp4_mean,
    "RNN":  rnn4_mean,
    "LSTM": lstm4_mean,
    "GRU":  gru4_mean,
    "TCN":  tcn4_mean,
    "ELM":  elm4_mean,
}

colors = {
    "MLP":  "#ff7f0e",
    "RNN":  "#2ca02c",
    "LSTM": "#d62728",
    "GRU":  "#9467bd",
    "TCN":  "#8c564b",
    "ELM":  "#e377c2",
}

for name, m_mean in models_boot_mean.items():
    plt.figure(figsize=(11,4))
    plt.plot(x_axis, y4_true, label="True (test)", color="#1f77b4", linewidth=1)
    plt.plot(x_axis, m_mean, label=f"{name} — média ME-bootstrap",
             color=colors[name], linewidth=1.2)
    plt.title(f"Caso 4 — ME-bootstrap (h=2, stride=2) — {name} — {os.path.basename(CSV_PATH)}")
    plt.xlabel("Índice (região de teste, stride=2)")
    plt.ylabel("Potência (escala original)")
    plt.legend()
    plt.tight_layout()
    # plt.savefig(f"caso4_meboot_light_{name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
#%%

