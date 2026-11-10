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

MD_LAGS    = [1, 5, 15]

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

#%% ===================== Janelas (one-step) =====================
def make_supervised(series, window):
    """
    X: janelas [t-window+1 .. t]
    y: próximo ponto [t+1]  (one-step)
    """
    X, y = [], []
    s = series.reshape(-1)
    for i in range(len(s) - window):
        X.append(s[i:i+window])
        y.append(s[i+window])
    return np.array(X), np.array(y)

#%% ===================== Moving Difference (MD) =====================
def moving_difference(y, k):
    """
    MD_k(t) = y_t - y_{t-k}, definido para t=k..N-1
    Retorna vetor md de tamanho N-k, alinhado com t=k..N-1 (causal).
    """
    y = np.asarray(y).reshape(-1)
    return y[k:] - y[:-k]

def reconstruct_from_md(md_hat, y_tmk):
    """
    y_t = y_{t-k} + MD_k(t)
    """
    return y_tmk + md_hat

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

#%% ===================== Treino Keras helper =====================
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
y_raw = df[COL_TARGET].values.astype(float)

N = len(y_raw)
split = int((1 - TEST_RATIO) * N)

MAX_K = max(MD_LAGS)

# Treino: até split
y_train_raw = y_raw[:split]

# Teste: pega um pedaço que inclui histórico suficiente (WINDOW + MAX_K)
y_test_raw  = y_raw[split - (WINDOW + MAX_K):]

print(f"Dataset: {os.path.basename(CSV_PATH)} | Total={N} | Train={len(y_train_raw)} | Test(seg)={len(y_test_raw)}")

#%% ===================== Rodar um cenário MD(k) =====================
def run_md_k(k, make_plots=True, n_plot=30000):
    print("\n" + "="*80)
    print(f"RODANDO MD({k})")
    print("="*80)

    # 1) Transformação MD no treino e teste (separado, para não vazar)
    md_train = moving_difference(y_train_raw, k)  # tamanho len(train)-k
    md_test  = moving_difference(y_test_raw,  k)  # tamanho len(test)-k

    # 2) Escalonar MD (separado por k)
    scaler = MinMaxScaler()
    md_train_s = scaler.fit_transform(md_train.reshape(-1,1)).reshape(-1)
    md_test_s  = scaler.transform(md_test.reshape(-1,1)).reshape(-1)

    # 3) Janelas one-step no domínio MD
    Xtr, ytr = make_supervised(md_train_s, WINDOW)
    Xte, yte = make_supervised(md_test_s,  WINDOW)

    Xtr_rnn = Xtr.reshape(-1, WINDOW, 1)
    Xte_rnn = Xte.reshape(-1, WINDOW, 1)

    # 4) Treinar 6 modelos para prever MD
    preds_md = {}

    mlp = compile_and_fit(build_mlp(WINDOW), Xtr, ytr, Xte, yte)
    preds_md["MLP"] = mlp.predict(Xte, verbose=0).reshape(-1)

    rnn = compile_and_fit(build_rnn(WINDOW), Xtr_rnn, ytr, Xte_rnn, yte)
    preds_md["RNN"] = rnn.predict(Xte_rnn, verbose=0).reshape(-1)

    lstm = compile_and_fit(build_lstm(WINDOW), Xtr_rnn, ytr, Xte_rnn, yte)
    preds_md["LSTM"] = lstm.predict(Xte_rnn, verbose=0).reshape(-1)

    gru = compile_and_fit(build_gru(WINDOW), Xtr_rnn, ytr, Xte_rnn, yte)
    preds_md["GRU"] = gru.predict(Xte_rnn, verbose=0).reshape(-1)

    tcn = compile_and_fit(build_tcn(WINDOW), Xtr_rnn, ytr, Xte_rnn, yte)
    preds_md["TCN"] = tcn.predict(Xte_rnn, verbose=0).reshape(-1)

    elm = ELMRegressor(n_hidden=600, alpha=1e-2, random_state=SEED)
    elm.fit(Xtr, ytr)
    preds_md["ELM"] = elm.predict(Xte).reshape(-1)

    # 5) Desescalonar MD previsto -> MD em escala original
    for m in list(preds_md.keys()):
        preds_md[m] = scaler.inverse_transform(preds_md[m].reshape(-1,1)).reshape(-1)

    # 6) Reconstruir y original e montar y_true correspondente
    #
    # md_test foi construído assim: md_test[j] = y_test_raw[j+k] - y_test_raw[j]
    # md index j corresponde ao tempo t = j+k (no eixo do y_test_raw)
    #
    # make_supervised em md: alvo em md_index = i+WINDOW  (chame de idx)
    # então o tempo original previsto é: t = idx + k
    #
    # reconstrução: y_hat[t] = y_test_raw[idx] + md_hat[idx]
    # verdadeiro:   y_true   = y_test_raw[idx + k]
    n_samples = len(Xte)
    idx = np.arange(n_samples) + WINDOW  # idx no md_test
    y_base = y_test_raw[idx]             # y_{t-k} no original
    y_true = y_test_raw[idx + k]         # y_t no original

    preds_y = {m: reconstruct_from_md(preds_md[m], y_base) for m in preds_md}

    # 7) Métricas em y original
    results = {}
    for m, yhat in preds_y.items():
        results[m] = report_metrics(f"MD({k}) + {m}", y_true, yhat)

    # 8) Plots (opcional)
    if make_plots:
        n = min(n_plot, len(y_true))

        # Comparação rápida (zoom geral)
        plt.figure(figsize=(12,5))
        plt.plot(y_true[:n], label="True", linewidth=1.2)
        for m in ["MLP","RNN","LSTM","GRU","TCN","ELM"]:
            plt.plot(preds_y[m][:n], label=m, alpha=0.85)
        plt.title(f"MD({k}) — Predição reconstruída no sinal original (primeiros {n} pts do teste)")
        plt.xlabel("Índice (amostras do teste)")
        plt.ylabel(COL_TARGET)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Erro acumulado
        plt.figure(figsize=(12,5))
        for m in ["MLP","RNN","LSTM","GRU","TCN","ELM"]:
            err = np.abs(y_true - preds_y[m])
            plt.plot(np.cumsum(err), label=m, alpha=0.9)
        plt.title(f"MD({k}) — Erro acumulado (cumsum |erro|)")
        plt.xlabel("Índice (teste)")
        plt.ylabel("cumsum(|erro|)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return y_true, preds_y, results

#%% ===================== Rodar MD(1), MD(5), MD(15) =====================
all_results = {}

for k in MD_LAGS:
    y_true_k, preds_y_k, res_k = run_md_k(k, make_plots=True, n_plot=30000)
    all_results[k] = res_k

#%% ===================== Resumo final (tabela) =====================
print("\n" + "="*80)
print("RESUMO — MÉTRICAS (y original) por MD(k) e Modelo")
print("="*80)

for k in MD_LAGS:
    print(f"\n--- MD({k}) ---")
    # ordena por RMSE
    ranked = sorted(all_results[k].items(), key=lambda kv: kv[1]["RMSE"])
    for m, met in ranked:
        print(f"{m:>4} | MAE={met['MAE']:.3e} | RMSE={met['RMSE']:.3e} | R2={met['R2']:.4f} | MedAE={met['MedianAE']:.3e}")
