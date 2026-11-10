#%% ===================== Imports & Setup =====================
import os, math, warnings
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

# Reprodutibilidade
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

#%% ===================== Configurações =====================
CSV_PATH   = "EQL_STI_EQL_PBS.csv"   # troque aqui
COL_TARGET = "power"

TEST_RATIO = 0.2
WINDOW     = 15
EPOCHS     = 100
BATCH      = 256

#%% ===================== Funções auxiliares =====================
def median_ae(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.median(np.abs(y_true - y_pred))

def report_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    med = median_ae(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"MAE        : {mae:.6e}")
    print(f"RMSE       : {rmse:.6e}")
    print(f"R²         : {r2:.4f}")
    print(f"Median AE  : {med:.6e}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MedianAE": med}

def make_supervised(series, window):
    """Cria X,y one-step: janela -> próximo ponto"""
    X, y = [], []
    s = series.reshape(-1)
    for i in range(len(s) - window):
        X.append(s[i:i+window])
        y.append(s[i+window])
    return np.array(X), np.array(y)

def centered_diff_1(y):
    """
    d1_t = (y_{t+1} - y_{t-1})/2, definido para t=1..N-2
    Retorna vetor de tamanho N-2 alinhado com t=1..N-2
    """
    y = np.asarray(y).reshape(-1)
    return (y[2:] - y[:-2]) / 2.0

def centered_diff_2(y):
    """
    d2_t = y_{t+1} - 2y_t + y_{t-1}, definido para t=1..N-2
    Retorna vetor de tamanho N-2 alinhado com t=1..N-2
    """
    y = np.asarray(y).reshape(-1)
    return (y[2:] - 2*y[1:-1] + y[:-2])

# Reconstruções (one-step ahead)
def recon_from_d1(d1_hat, y_tm1):
    """
    d1_t = (y_{t+1} - y_{t-1})/2  =>  y_{t+1} = y_{t-1} + 2*d1_t
    Aqui usamos y_tm1 = y_{t-1} real conhecido.
    """
    return y_tm1 + 2.0 * d1_hat

def recon_from_d2(d2_hat, y_t, y_tm1):
    """
    d2_t = y_{t+1} - 2*y_t + y_{t-1}  =>  y_{t+1} = 2*y_t - y_{t-1} + d2_t
    """
    return 2.0*y_t - y_tm1 + d2_hat

# Modelos
def build_mlp(window):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(window,)),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    return model

def build_rnn(window):
    model = Sequential([
        SimpleRNN(64, activation="tanh", input_shape=(window,1)),
        Dense(1)
    ])
    return model

def build_lstm(window):
    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(window,1)),
        Dense(1)
    ])
    return model

def build_gru(window):
    model = Sequential([
        GRU(64, activation="tanh", input_shape=(window,1)),
        Dense(1)
    ])
    return model

def build_tcn(window):
    model = Sequential([
        layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu", input_shape=(window,1)),
        layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu"),
        layers.GlobalAveragePooling1D(),
        Dense(1)
    ])
    return model

from sklearn.linear_model import Ridge
class ELMRegressor:
    def __init__(self, n_hidden=600, alpha=1e-2, random_state=42):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.random_state = np.random.RandomState(random_state)
        self.W = None
        self.b = None
        self.out = Ridge(alpha=self.alpha)

    def _hidden(self, X):
        return np.tanh(X @ self.W + self.b)

    def fit(self, X, y):
        self.W = self.random_state.normal(0, 1, size=(X.shape[1], self.n_hidden))
        self.b = self.random_state.normal(0, 1, size=(self.n_hidden,))
        H = self._hidden(X)
        self.out.fit(H, y)
        return self

    def predict(self, X):
        H = self._hidden(X)
        return self.out.predict(H)

#%% ===================== Carregar série e split =====================
df = pd.read_csv(CSV_PATH)
y_raw = df[COL_TARGET].values.astype(float)

N = len(y_raw)
split = int((1 - TEST_RATIO) * N)

y_train_raw = y_raw[:split]
y_test_raw  = y_raw[split - WINDOW:]   # mantém histórico pro teste

print(f"Dataset: {os.path.basename(CSV_PATH)} | Total={N} | Train={len(y_train_raw)} | Test(raw seg)={len(y_test_raw)}")

#%% ===================== Pré-processamento: d1 e d2 (centradas) =====================
# Criamos d1/d2 em cada segmento (treino e teste) separadamente
# (isso evita vazamento de info do futuro do TESTE pro TREINO)

d1_train = centered_diff_1(y_train_raw)   # tamanho len(train)-2
d2_train = centered_diff_2(y_train_raw)

d1_test  = centered_diff_1(y_test_raw)    # tamanho len(test_raw)-2
d2_test  = centered_diff_2(y_test_raw)


#%% ===================== Escalonar d1 e d2 (separado) =====================
scaler_d1 = MinMaxScaler()
scaler_d2 = MinMaxScaler()

d1_train_s = scaler_d1.fit_transform(d1_train.reshape(-1,1)).reshape(-1)
d1_test_s  = scaler_d1.transform(d1_test.reshape(-1,1)).reshape(-1)

d2_train_s = scaler_d2.fit_transform(d2_train.reshape(-1,1)).reshape(-1)
d2_test_s  = scaler_d2.transform(d2_test.reshape(-1,1)).reshape(-1)

#%% ===================== Criar janelas em d1 e d2 =====================
X1_tr, y1_tr = make_supervised(d1_train_s, WINDOW)
X1_te, y1_te = make_supervised(d1_test_s,  WINDOW)

X2_tr, y2_tr = make_supervised(d2_train_s, WINDOW)
X2_te, y2_te = make_supervised(d2_test_s,  WINDOW)

# Para redes recorrentes/TCN:
X1_tr_rnn = X1_tr.reshape(-1, WINDOW, 1)
X1_te_rnn = X1_te.reshape(-1, WINDOW, 1)

X2_tr_rnn = X2_tr.reshape(-1, WINDOW, 1)
X2_te_rnn = X2_te.reshape(-1, WINDOW, 1)

print(f"d1: X_train={X1_tr.shape}, X_test={X1_te.shape}")
print(f"d2: X_train={X2_tr.shape}, X_test={X2_te.shape}")

#%% ===================== Callbacks =====================
early  = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)

def compile_and_fit_keras(model, Xtr, ytr, Xva, yva, name="model"):
    model.compile(optimizer="adam", loss="mse")
    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=1,
        callbacks=[early, reduce]
    )
    return hist

#%% ===================== Treinar e avaliar para um cenário (d1 ou d2) =====================
def run_scenario_diff(diff_name, Xtr, ytr, Xte, yte, Xtr_rnn, Xte_rnn,
                      inv_scaler, recon_fn):
    """
    diff_name: "d1" ou "d2"
    inv_scaler: scaler para desfazer escala do diff previsto
    recon_fn: função que reconstrói y_hat (original) a partir do diff_hat (original scale)
    """
    results = {}
    preds_yhat = {}  # y_hat em escala original do y (não escalado)

    # ========= 1) MLP =========
    mlp = build_mlp(WINDOW)
    compile_and_fit_keras(mlp, Xtr, ytr, Xte, yte, "MLP")
    d_hat = mlp.predict(Xte, verbose=0).reshape(-1)
    d_hat = inv_scaler.inverse_transform(d_hat.reshape(-1,1)).reshape(-1)

    # ========= 2) RNN =========
    rnn = build_rnn(WINDOW)
    compile_and_fit_keras(rnn, Xtr_rnn, ytr, Xte_rnn, yte, "RNN")
    d_hat_rnn = rnn.predict(Xte_rnn, verbose=0).reshape(-1)
    d_hat_rnn = inv_scaler.inverse_transform(d_hat_rnn.reshape(-1,1)).reshape(-1)

    # ========= 3) LSTM =========
    lstm = build_lstm(WINDOW)
    compile_and_fit_keras(lstm, Xtr_rnn, ytr, Xte_rnn, yte, "LSTM")
    d_hat_lstm = lstm.predict(Xte_rnn, verbose=0).reshape(-1)
    d_hat_lstm = inv_scaler.inverse_transform(d_hat_lstm.reshape(-1,1)).reshape(-1)

    # ========= 4) GRU =========
    gru = build_gru(WINDOW)
    compile_and_fit_keras(gru, Xtr_rnn, ytr, Xte_rnn, yte, "GRU")
    d_hat_gru = gru.predict(Xte_rnn, verbose=0).reshape(-1)
    d_hat_gru = inv_scaler.inverse_transform(d_hat_gru.reshape(-1,1)).reshape(-1)

    # ========= 5) TCN =========
    tcn = build_tcn(WINDOW)
    compile_and_fit_keras(tcn, Xtr_rnn, ytr, Xte_rnn, yte, "TCN")
    d_hat_tcn = tcn.predict(Xte_rnn, verbose=0).reshape(-1)
    d_hat_tcn = inv_scaler.inverse_transform(d_hat_tcn.reshape(-1,1)).reshape(-1)

    # ========= 6) ELM =========
    elm = ELMRegressor(n_hidden=600, alpha=1e-2, random_state=SEED)
    elm.fit(Xtr, ytr)
    d_hat_elm = elm.predict(Xte).reshape(-1)
    d_hat_elm = inv_scaler.inverse_transform(d_hat_elm.reshape(-1,1)).reshape(-1)

    # --- Reconstrução em y original (alinhamento temporal) ---
    # t_local = (i + W) + 1
    # y_true = y_test_raw[t_local + 1]
    #
    # para d1: y_{t-1} = y_test_raw[t_local - 1]
    # para d2: y_t = y_test_raw[t_local], y_{t-1} = y_test_raw[t_local - 1]

    M = len(y_test_raw)
    n_samples = len(Xte)

    t_local = np.arange(n_samples) + WINDOW + 1  # vetor com t_local para cada amostra
    y_true = y_test_raw[t_local + 1]             # y_{t+1}

    if diff_name == "d1":
        y_tm1 = y_test_raw[t_local - 1]
        yhat_mlp  = recon_fn(d_hat,      y_tm1)
        yhat_rnn  = recon_fn(d_hat_rnn,  y_tm1)
        yhat_lstm = recon_fn(d_hat_lstm, y_tm1)
        yhat_gru  = recon_fn(d_hat_gru,  y_tm1)
        yhat_tcn  = recon_fn(d_hat_tcn,  y_tm1)
        yhat_elm  = recon_fn(d_hat_elm,  y_tm1)
    else:  # d2
        y_t   = y_test_raw[t_local]
        y_tm1 = y_test_raw[t_local - 1]
        yhat_mlp  = recon_fn(d_hat,      y_t, y_tm1)
        yhat_rnn  = recon_fn(d_hat_rnn,  y_t, y_tm1)
        yhat_lstm = recon_fn(d_hat_lstm, y_t, y_tm1)
        yhat_gru  = recon_fn(d_hat_gru,  y_t, y_tm1)
        yhat_tcn  = recon_fn(d_hat_tcn,  y_t, y_tm1)
        yhat_elm  = recon_fn(d_hat_elm,  y_t, y_tm1)

    preds_yhat["MLP"]  = yhat_mlp
    preds_yhat["RNN"]  = yhat_rnn
    preds_yhat["LSTM"] = yhat_lstm
    preds_yhat["GRU"]  = yhat_gru
    preds_yhat["TCN"]  = yhat_tcn
    preds_yhat["ELM"]  = yhat_elm

    # --- Métricas finais em y original ---
    for k, yhat in preds_yhat.items():
        results[k] = report_metrics(f"{diff_name.upper()} + {k}", y_true, yhat)

    return y_true, preds_yhat, results

#%% ===================== Rodar Cenário d1 =====================
y_true_d1, preds_d1, results_d1 = run_scenario_diff(
    diff_name="d1",
    Xtr=X1_tr, ytr=y1_tr, Xte=X1_te, yte=y1_te,
    Xtr_rnn=X1_tr_rnn, Xte_rnn=X1_te_rnn,
    inv_scaler=scaler_d1,
    recon_fn=recon_from_d1
)

#%% ===================== Rodar Cenário d2 =====================
y_true_d2, preds_d2, results_d2 = run_scenario_diff(
    diff_name="d2",
    Xtr=X2_tr, ytr=y2_tr, Xte=X2_te, yte=y2_te,
    Xtr_rnn=X2_tr_rnn, Xte_rnn=X2_te_rnn,
    inv_scaler=scaler_d2,
    recon_fn=recon_from_d2
)

#%% ===================== Plots =====================
# Paleta consistente (matplotlib default cycle)
MODEL_ORDER = ["MLP","RNN","LSTM","GRU","TCN","ELM"]

def plot_zoom(title, y_true, preds, start=10000, end=10200):
    plt.figure(figsize=(12,5))
    plt.plot(y_true[start:end], label="True (test)", linewidth=1.5)
    for m in MODEL_ORDER:
        plt.plot(preds[m][start:end], label=m, alpha=0.9)
    plt.title(title)
    plt.xlabel("Índice da amostra")
    plt.ylabel(COL_TARGET)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cumsum(title, y_true, preds):
    plt.figure(figsize=(12,5))
    for m in MODEL_ORDER:
        err = np.abs(y_true - preds[m])
        plt.plot(np.cumsum(err), label=m, alpha=0.9)
    plt.title(title)
    plt.xlabel("Índice da amostra (teste)")
    plt.ylabel("cumsum(|erro|)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_err_hist(title, y_true, preds, bins=120):
    plt.figure(figsize=(12,5))
    for m in MODEL_ORDER:
        e = (y_true - preds[m])
        plt.hist(e, bins=bins, alpha=0.25, density=True, label=m)
    plt.title(title)
    plt.xlabel("Erro (y_true - y_pred)")
    plt.ylabel("Densidade")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_scatter(title, y_true, preds, nmax=25000):
    n = min(len(y_true), nmax)
    plt.figure(figsize=(7,7))
    for m in MODEL_ORDER:
        plt.scatter(y_true[:n], preds[m][:n], s=6, alpha=0.12, label=m)
    plt.title(title)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_derivative(title, y_true, preds):
    # Derivada discreta do sinal original reconstruído (Δy = y_t - y_{t-1})
    plt.figure(figsize=(12,5))
    dy_true = np.diff(y_true)
    plt.plot(dy_true, label="True Δ", linewidth=1.5)
    for m in MODEL_ORDER:
        dy_pred = np.diff(preds[m])
        plt.plot(dy_pred, label=f"{m} Δ", alpha=0.75)
    plt.title(title)
    plt.xlabel("Índice")
    plt.ylabel("Δ (y_t - y_{t-1})")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

#%% --- Plots para d1 ---
plot_zoom("Cenário d1 (1ª diferença centrada) — Zoom no teste [10000:10200]", y_true_d1, preds_d1, 10000, 10200)
plot_cumsum("Cenário d1 — Erro acumulado (cumsum)", y_true_d1, preds_d1)
plot_err_hist("Cenário d1 — Distribuição dos erros (y_true - y_pred)", y_true_d1, preds_d1)
plot_scatter("Cenário d1 — Dispersão y_pred vs y_true", y_true_d1, preds_d1)
plot_derivative("Cenário d1 — Derivada da série (Δ) — Verdade vs modelos", y_true_d1, preds_d1)

#%% --- Plots para d2 ---
plot_zoom("Cenário d2 (2ª diferença centrada) — Zoom no teste [10000:10200]", y_true_d2, preds_d2, 10000, 10200)
plot_cumsum("Cenário d2 — Erro acumulado (cumsum)", y_true_d2, preds_d2)
plot_err_hist("Cenário d2 — Distribuição dos erros (y_true - y_pred)", y_true_d2, preds_d2)
plot_scatter("Cenário d2 — Dispersão y_pred vs y_true", y_true_d2, preds_d2)
plot_derivative("Cenário d2 — Derivada da série (Δ) — Verdade vs modelos", y_true_d2, preds_d2)
