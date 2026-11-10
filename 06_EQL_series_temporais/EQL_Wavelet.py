#%% ===================== Imports & Setup =====================
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers

from sklearn.linear_model import Ridge

import pywt

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

#%% ===================== Configurações =====================
CSV_PATH   = "EQL_STI_EQL_PBS.csv"   # troque
COL_TARGET = "power"

TEST_RATIO = 0.2
WINDOW     = 15
EPOCHS     = 100
BATCH      = 256

# Wavelet
WAVELET     = "db4"
LEVEL       = None     # None = máximo permitido pelo tamanho da janela
THRESH_MODE = "soft"   # "soft" ou "hard"

# Plot
N_PLOT = 30000  # pontos do teste para plot

#%% ===================== Métricas =====================
def median_ae(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.median(np.abs(y_true - y_pred))

def report_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    med  = median_ae(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"MAE        : {mae:.6e}")
    print(f"RMSE       : {rmse:.6e}")
    print(f"R²         : {r2:.4f}")
    print(f"Median AE  : {med:.6e}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MedianAE": med}

#%% ===================== Wavelet denoise (por janela) =====================
def _mad(x):
    x = np.asarray(x)
    return np.median(np.abs(x - np.median(x)))

def wavelet_denoise_1d(x, wavelet="db4", level=None, mode="soft"):
    """
    Denoise de um vetor 1D com wavelet thresholding.
    Retorna sinal reconstruído com o mesmo tamanho de x.
    """
    x = np.asarray(x).reshape(-1)
    n = len(x)

    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)

    # sigma via MAD do último detalhe
    detail = coeffs[-1]
    sigma = _mad(detail) / 0.6745 if len(detail) > 0 else 0.0

    # threshold universal
    thr = sigma * np.sqrt(2.0 * np.log(max(n, 2)))

    cA = coeffs[0]
    new_coeffs = [cA]
    for cD in coeffs[1:]:
        new_coeffs.append(pywt.threshold(cD, value=thr, mode=mode))

    x_rec = pywt.waverec(new_coeffs, wavelet=wavelet)

    # ajusta tamanho
    if len(x_rec) > n:
        x_rec = x_rec[:n]
    elif len(x_rec) < n:
        x_rec = np.pad(x_rec, (0, n - len(x_rec)), mode="edge")

    return x_rec

def make_wavelet_supervised(y_raw_segment, window, wavelet="db4", level=None, mode="soft"):
    """
    Cria dataset supervisionado:
      X_t = janela denoised (causal) construída a partir da janela raw [t-window+1..t]
      y_t = próximo ponto bruto y_raw[t+1]

    Retorna:
      X (n, window)
      y (n,)
    """
    y = np.asarray(y_raw_segment).reshape(-1)
    X, Y = [], []
    # t vai até len(y)-2 para existir y[t+1]
    for t in range(window - 1, len(y) - 1):
        raw_win = y[t-window+1:t+1]
        den_win = wavelet_denoise_1d(raw_win, wavelet=wavelet, level=level, mode=mode)
        X.append(den_win)        # entrada = janela denoised
        Y.append(y[t+1])         # alvo = próximo ponto BRUTO
    return np.array(X), np.array(Y)

#%% ===================== Modelos =====================
def build_mlp(window):
    return Sequential([
        Dense(64, activation="relu", input_shape=(window,)),
        Dense(32, activation="relu"),
        Dense(1)
    ])

def build_rnn(window):
    return Sequential([
        SimpleRNN(64, activation="tanh", input_shape=(window,1)),
        Dense(1)
    ])

def build_lstm(window):
    return Sequential([
        LSTM(64, activation="tanh", input_shape=(window,1)),
        Dense(1)
    ])

def build_gru(window):
    return Sequential([
        GRU(64, activation="tanh", input_shape=(window,1)),
        Dense(1)
    ])

def build_tcn(window):
    return Sequential([
        layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu", input_shape=(window,1)),
        layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu"),
        layers.GlobalAveragePooling1D(),
        Dense(1)
    ])

class ELMRegressor:
    def __init__(self, n_hidden=600, alpha=1e-2, random_state=42):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.rs = np.random.RandomState(random_state)
        self.W = None
        self.b = None
        self.out = Ridge(alpha=self.alpha)

    def _hidden(self, X):
        return np.tanh(X @ self.W + self.b)

    def fit(self, X, y):
        self.W = self.rs.normal(0, 1, size=(X.shape[1], self.n_hidden))
        self.b = self.rs.normal(0, 1, size=(self.n_hidden,))
        H = self._hidden(X)
        self.out.fit(H, y)
        return self

    def predict(self, X):
        H = self._hidden(X)
        return self.out.predict(H)

#%% ===================== Callbacks =====================
early  = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)

def compile_and_fit(model, Xtr, ytr, Xva, yva):
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=1,
        callbacks=[early, reduce]
    )
    return model

#%% ===================== Carregar série e split (Caso 2, sem bootstrap) =====================
df = pd.read_csv(CSV_PATH)
y_all = df[COL_TARGET].values.astype(float)

N = len(y_all)
split = int((1 - TEST_RATIO) * N)

# Treino: somente passado
y_train_raw = y_all[:split]

# Teste: inclui um pouco de histórico antes do split para permitir janelas
y_test_raw  = y_all[split - WINDOW:]  # inclui WINDOW pontos antes

print(f"Dataset: {os.path.basename(CSV_PATH)} | Total={N} | Train={len(y_train_raw)} | Test(seg)={len(y_test_raw)}")

#%% ===================== Construir datasets Wavelet (X denoised, y bruto) =====================
X_train_rawden, y_train_next = make_wavelet_supervised(
    y_train_raw, window=WINDOW, wavelet=WAVELET, level=LEVEL, mode=THRESH_MODE
)
X_test_rawden,  y_test_next  = make_wavelet_supervised(
    y_test_raw,  window=WINDOW, wavelet=WAVELET, level=LEVEL, mode=THRESH_MODE
)

print(f"X_train: {X_train_rawden.shape}, y_train: {y_train_next.shape}")
print(f"X_test : {X_test_rawden.shape},  y_test : {y_test_next.shape}")

#%% ===================== Escalonar (comparável) =====================
# Para comparar com outros pré-processamentos que avaliam em y_raw:
# - escalamos Y (alvo) usando scaler fitado no treino BRUTO
# - escalamos X (entrada denoised) usando o MESMO scaler (só para consistência de escala)
scaler_y = MinMaxScaler()
scaler_y.fit(y_train_raw.reshape(-1,1))

X_train = scaler_y.transform(X_train_rawden.reshape(-1,1)).reshape(-1, WINDOW)
X_test  = scaler_y.transform(X_test_rawden.reshape(-1,1)).reshape(-1, WINDOW)

y_train = scaler_y.transform(y_train_next.reshape(-1,1)).reshape(-1)
y_test  = scaler_y.transform(y_test_next.reshape(-1,1)).reshape(-1)

X_train_rnn = X_train.reshape(-1, WINDOW, 1)
X_test_rnn  = X_test.reshape(-1, WINDOW, 1)

#%% ===================== Treinar 6 modelos (prever y_bruto one-step) =====================
pred_scaled = {}

mlp = compile_and_fit(build_mlp(WINDOW), X_train, y_train, X_test, y_test)
pred_scaled["MLP"] = mlp.predict(X_test, verbose=0).reshape(-1)

rnn = compile_and_fit(build_rnn(WINDOW), X_train_rnn, y_train, X_test_rnn, y_test)
pred_scaled["RNN"] = rnn.predict(X_test_rnn, verbose=0).reshape(-1)

lstm = compile_and_fit(build_lstm(WINDOW), X_train_rnn, y_train, X_test_rnn, y_test)
pred_scaled["LSTM"] = lstm.predict(X_test_rnn, verbose=0).reshape(-1)

gru = compile_and_fit(build_gru(WINDOW), X_train_rnn, y_train, X_test_rnn, y_test)
pred_scaled["GRU"] = gru.predict(X_test_rnn, verbose=0).reshape(-1)

tcn = compile_and_fit(build_tcn(WINDOW), X_train_rnn, y_train, X_test_rnn, y_test)
pred_scaled["TCN"] = tcn.predict(X_test_rnn, verbose=0).reshape(-1)

elm = ELMRegressor(n_hidden=600, alpha=1e-2, random_state=SEED)
elm.fit(X_train, y_train)
pred_scaled["ELM"] = elm.predict(X_test).reshape(-1)

#%% ===================== Voltar para o domínio original BRUTO =====================
y_true_raw = y_test_next  # já está no bruto original (não escalado)

pred_raw = {}
for m, p in pred_scaled.items():
    pred_raw[m] = scaler_y.inverse_transform(p.reshape(-1,1)).reshape(-1)

#%% ===================== Métricas =====================
results = {}
for m in ["MLP","RNN","LSTM","GRU","TCN","ELM"]:
    results[m] = report_metrics(f"WAVELET(X) -> y_raw + {m}", y_true_raw, pred_raw[m])

#%% ===================== Plots =====================
MODEL_ORDER = ["MLP","RNN","LSTM","GRU","TCN","ELM"]

# Comparação geral
plt.figure(figsize=(12,5))
n = min(N_PLOT, len(y_true_raw))
plt.plot(y_true_raw[:n], label="True (raw)", linewidth=1.4)
for m in MODEL_ORDER:
    plt.plot(pred_raw[m][:n], label=m, alpha=0.85)
plt.title(f"Wavelet (entrada denoised, alvo bruto) — {os.path.basename(CSV_PATH)}")
plt.xlabel("Índice (teste)")
plt.ylabel(COL_TARGET)
plt.legend()
plt.tight_layout()
plt.show()

# Erro acumulado
plt.figure(figsize=(12,5))
for m in MODEL_ORDER:
    err = np.abs(y_true_raw - pred_raw[m])
    plt.plot(np.cumsum(err), label=m, alpha=0.9)
plt.title("Wavelet — Erro acumulado (cumsum |erro|) no y_bruto")
plt.xlabel("Índice (teste)")
plt.ylabel("cumsum(|erro|)")
plt.legend()
plt.tight_layout()
plt.show()

# Histograma de erros
plt.figure(figsize=(12,5))
for m in MODEL_ORDER:
    e = (y_true_raw - pred_raw[m])
    plt.hist(e, bins=120, alpha=0.25, density=True, label=m)
plt.title("Wavelet — Distribuição dos erros (y_true - y_pred) no y_bruto")
plt.xlabel("Erro")
plt.ylabel("Densidade")
plt.legend()
plt.tight_layout()
plt.show()

# Scatter
plt.figure(figsize=(7,7))
n_sc = min(25000, len(y_true_raw))
for m in MODEL_ORDER:
    plt.scatter(y_true_raw[:n_sc], pred_raw[m][:n_sc], s=6, alpha=0.12, label=m)
plt.title("Wavelet — Dispersão y_pred vs y_true (y_bruto)")
plt.xlabel("y_true (raw)")
plt.ylabel("y_pred (raw)")
plt.legend()
plt.tight_layout()
plt.show()

# Derivada discreta (Δ)
plt.figure(figsize=(12,5))
dy_true = np.diff(y_true_raw)
plt.plot(dy_true, label="True Δ", linewidth=1.5)
for m in MODEL_ORDER:
    plt.plot(np.diff(pred_raw[m]), label=f"{m} Δ", alpha=0.75)
plt.title("Wavelet — Derivada discreta (Δ) no y_bruto — Verdade vs modelos")
plt.xlabel("Índice")
plt.ylabel("Δ (y_t - y_{t-1})")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()
