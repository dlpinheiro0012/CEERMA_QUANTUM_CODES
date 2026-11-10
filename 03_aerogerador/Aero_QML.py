#%% ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd

from scipy.stats import kurtosis

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pennylane as qml


#%% ============================================================
# CAMINHO LOCAL
# ============================================================

BASE_PATH = r"C:\Users\Arthur\Desktop\aerogerador"

EVENT_INFO_PATH = rf"{BASE_PATH}\event_info.csv"


#%% ============================================================
# CARREGAR EVENT INFO
# ============================================================

event_info = pd.read_csv(
    EVENT_INFO_PATH,
    sep=";"
)

print("\nEVENT INFO")
print(event_info.head())

print("\nDistribuição das classes:")
print(event_info["event_label"].value_counts())


#%% ============================================================
# PARÂMETROS
# ============================================================

WINDOW_SIZE = 72
STEP = 72

MAX_EVENTS_PER_CLASS = 2

BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", DEVICE)


#%% ============================================================
# SENSORES ESCOLHIDOS
# ============================================================

SENSOR_COLUMNS = [
    "power_29_avg"
]





# ============================================================
# EXTRAÇÃO DE FEATURES
# ============================================================

from scipy.stats import kurtosis, skew

def extract_features(window):

    features = []

    for col in window.columns:

        signal = window[col].values

        mean = np.mean(signal)

        std = np.std(signal)

        rms = np.sqrt(np.mean(signal**2))

        maximum = np.max(signal)

        minimum = np.min(signal)

        peak_to_peak = maximum - minimum

        # ========================================
        # SKEWNESS
        # ========================================

        if np.std(signal) < 1e-8:

            skewness = 0

        else:

            skewness = skew(signal)

        # ========================================
        # KURTOSIS
        # ========================================

        if np.std(signal) < 1e-8:

            kurt = 0

        else:

            kurt = kurtosis(signal)

        # ========================================
        # SANITIZAÇÃO
        # ========================================

        current_features = [

            mean,
            std,
            rms,
            maximum,
            minimum,
            peak_to_peak,
            skewness,
            kurt
        ]

        current_features = np.nan_to_num(

            current_features,

            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        features.extend(current_features)

    return features


#%% ============================================================
# PROCESSAR EVENTOS
# ============================================================

def process_events(events_df, label):

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    for _, row in events_df.iterrows():

        event_id = row["event_id"]

        try:

            # ============================================
            # CAMINHO CSV
            # ============================================

            csv_path = rf"{BASE_PATH}\{event_id}.csv"

            # ============================================
            # LEITURA
            # ============================================

            df = pd.read_csv(
                csv_path,
                sep=";"
            )

            # ============================================
            # MANTER APENAS SENSORES
            # ============================================

            df = df[SENSOR_COLUMNS + ["train_test"]]

            # ============================================
            # REMOVER NaNs
            # ============================================

            df = df.dropna()

            # ============================================
            # SPLIT TEMPORAL
            # ============================================

            df_train = df[
                df["train_test"] == "train"
            ].copy()

            df_test = df[
                df["train_test"] == "prediction"
            ].copy()

            # ============================================
            # REMOVER train_test
            # ============================================

            df_train = df_train[SENSOR_COLUMNS]

            df_test = df_test[SENSOR_COLUMNS]

            # ============================================
            # JANELAMENTO TREINO
            # ============================================

            for start in range(
                0,
                len(df_train) - WINDOW_SIZE,
                STEP
            ):

                end = start + WINDOW_SIZE

                window = df_train.iloc[start:end]

                feats = extract_features(window)

                X_train.append(feats)

                y_train.append(label)

            # ============================================
            # JANELAMENTO TESTE
            # ============================================

            for start in range(
                0,
                len(df_test) - WINDOW_SIZE,
                STEP
            ):

                end = start + WINDOW_SIZE

                window = df_test.iloc[start:end]

                feats = extract_features(window)

                X_test.append(feats)

                y_test.append(label)

        except Exception as e:

            print(f"Erro no evento {event_id}: {e}")

    return X_train, y_train, X_test, y_test


#%% ============================================================
# SELECIONAR EVENTOS
# ============================================================

anomaly_events = event_info[
    event_info["event_label"] == "anomaly"
].head(MAX_EVENTS_PER_CLASS)

normal_events = event_info[
    event_info["event_label"] == "normal"
].head(MAX_EVENTS_PER_CLASS)


# ============================================================
# PROCESSAR ANOMALIAS
# ============================================================

Xa_train, ya_train, Xa_test, ya_test = process_events(
    anomaly_events,
    label=1
)


# ============================================================
# PROCESSAR NORMAIS
# ============================================================

Xn_train, yn_train, Xn_test, yn_test = process_events(
    normal_events,
    label=0
)


# ============================================================
# CONCATENAR
# ============================================================

X_train = np.array(Xa_train + Xn_train)

y_train = np.array(ya_train + yn_train)

X_test = np.array(Xa_test + Xn_test)

y_test = np.array(ya_test + yn_test)


print("\nShapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


#%% ============================================================
# NORMALIZAÇÃO
# ============================================================

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = np.nan_to_num(X_train)

X_test = np.nan_to_num(X_test)


# ============================================================
# CONVERTER PARA TENSOR
# ============================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
).to(DEVICE)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
).unsqueeze(1).to(DEVICE)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
).to(DEVICE)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
).unsqueeze(1).to(DEVICE)


# ============================================================
# DATALOADER
# ============================================================

train_dataset = TensorDataset(
    X_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


#%% ============================================================
# DEVICE QUÂNTICO
# ============================================================
N_QUBITS = X_train.shape[1]
print("\nNúmero final de qubits:", N_QUBITS)

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# CIRCUITO QUÂNTICO
# ============================================================

@qml.qnode(
    dev,
    interface="torch"
)

def quantum_circuit(inputs, weights):

    # ============================================
    # ANGLE ENCODING
    # ============================================
    
    for i in range(N_QUBITS):
    
        qml.RY(inputs[..., i], wires=i)

    # ============================================
    # PQC
    # ============================================

    for layer in range(weights.shape[0]):

        for i in range(N_QUBITS):

            qml.RY(inputs[..., i], wires=i)

            qml.RZ(
                weights[layer, i, 1],
                wires=i
            )

            qml.RY(inputs[..., i], wires=i)

        # ========================================
        # ENTANGLEMENT
        # ========================================

        for i in range(N_QUBITS - 1):

            qml.CZ(
                wires=[i, i + 1]
            )

    # ============================================
    # MEASUREMENTS
    # ============================================

    return [
        qml.expval(qml.PauliZ(i))
        for i in range(N_QUBITS)
    ]


#%% ============================================================
# SHAPE DOS PESOS
# ============================================================

weight_shapes = {
    "weights": (
        2,
        N_QUBITS,
        3
    )
}


# ============================================================
# CAMADA QUÂNTICA
# ============================================================

qlayer = qml.qnn.TorchLayer(
    quantum_circuit,
    weight_shapes
)


# ============================================================
# MODELO HÍBRIDO
# ============================================================

class HybridModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.qlayer = qlayer

        self.fc1 = nn.Linear(
            N_QUBITS,
            16
        )

        self.fc2 = nn.Linear(
            16,
            8
        )

        self.fc3 = nn.Linear(
            8,
            1
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        # ========================================
        # CAMADA QUÂNTICA
        # ========================================

        x = self.qlayer(x)

        # ========================================
        # MLP CLÁSSICA
        # ========================================

        x = self.relu(
            self.fc1(x)
        )

        x = self.relu(
            self.fc2(x)
        )

        # ========================================
        # SAÍDA FINAL (SEM SIGMOID)
        # BCEWithLogitsLoss faz isso internamente
        # ========================================

        x = self.fc3(x)

        return x


# ============================================================
# INSTANCIAR MODELO
# ============================================================

model = HybridModel().to(DEVICE)


# ============================================================
# LOSS E OTIMIZADOR
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


#%% ============================================================
# TREINAMENTO
# ============================================================

print("\n====================================")
print("INICIANDO TREINAMENTO...")
print("====================================\n")

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        # ========================================
        # FORWARD
        # ========================================

        outputs = model(batch_x)

        # ========================================
        # LOSS
        # ========================================

        loss = criterion(
            outputs,
            batch_y
        )

        # ========================================
        # BACKPROP
        # ========================================

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"- Loss: {epoch_loss:.4f}"
    )


# ============================================================
# TESTE
# ============================================================

model.eval()

with torch.no_grad():

    # ============================================
    # LOGITS
    # ============================================

    y_pred_logits = model(X_test)

    # ============================================
    # CONVERTER PARA PROBABILIDADE
    # ============================================

    y_pred_prob = torch.sigmoid(
        y_pred_logits
    )

    # ============================================
    # BINARIZAÇÃO
    # ============================================

    y_pred = (
        y_pred_prob > 0.5
    ).float()


# ============================================================
# CONVERTER PARA NUMPY
# ============================================================

y_true = y_test.cpu().numpy()

y_pred = y_pred.cpu().numpy()


# ============================================================
# MÉTRICAS
# ============================================================

acc = accuracy_score(
    y_true,
    y_pred
)

balanced_acc = balanced_accuracy_score(y_true, y_pred)

prec = precision_score(
    y_true,
    y_pred
)

rec = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

cm = confusion_matrix(
    y_true,
    y_pred
)


#%% ============================================================
# RESULTADOS
# ============================================================

print("\n====================================")
print("RESULTADOS")
print("====================================")

print(f"\nAccuracy : {acc:.4f}")

print(f"\nBalanced Accuracy : {balanced_acc:.4f}")

print(f"Precision: {prec:.4f}")

print(f"Recall   : {rec:.4f}")

print(f"F1-Score : {f1:.4f}")


print("\n====================================")
print("MATRIZ DE CONFUSÃO")
print("====================================")

print(cm)


print("\n====================================")
print("CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        y_true,
        y_pred
    )
)