#%% ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd

from scipy.stats import kurtosis, skew

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


#%% ============================================================
# CAMINHO LOCAL
# ============================================================

BASE_PATH = r"C:\Users\Arthur\Desktop\aerogerador"

EVENT_INFO_PATH = rf"{BASE_PATH}\event_info.csv"


# ============================================================
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
# IDENTIFICAR CSVs EXISTENTES
# ============================================================

available_files = [

    f.replace(".csv", "")

    for f in os.listdir(BASE_PATH)

    if f.endswith(".csv")
]

available_ids = []

for f in available_files:

    try:
        available_ids.append(int(f))
    except:
        pass


# ============================================================
# FILTRAR EVENTOS EXISTENTES
# ============================================================

filtered_events = event_info[
    event_info["event_id"].isin(available_ids)
]


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
# SENSORES
# ============================================================

SENSOR_COLUMNS = [

    "power_29_avg"
]


# ============================================================
# EXTRAÇÃO DE FEATURES
# ============================================================

def extract_features(window):

    features = []

    for col in window.columns:

        signal = window[col].values

        mean = np.mean(signal)

        std = np.std(signal)

        rms = np.sqrt(np.mean(signal**2))

        # ========================================
        # SANITIZAÇÃO
        # ========================================

        current_features = [
            mean,
            std,
            rms
        ]

        current_features = np.nan_to_num(
            current_features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        features.extend(current_features)

    return features


# ============================================================
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

            csv_path = rf"{BASE_PATH}\{event_id}.csv"

            df = pd.read_csv(
                csv_path,
                sep=";"
            )

            df = df[SENSOR_COLUMNS + ["train_test"]]

            df = df.dropna()

            # ========================================
            # SPLIT TEMPORAL
            # ========================================

            df_train = df[
                df["train_test"] == "train"
            ].copy()

            df_test = df[
                df["train_test"] == "prediction"
            ].copy()

            df_train = df_train[SENSOR_COLUMNS]

            df_test = df_test[SENSOR_COLUMNS]

            # ========================================
            # JANELAMENTO TREINO
            # ========================================

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

            # ========================================
            # JANELAMENTO TESTE
            # ========================================

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


# ============================================================
# SEPARAR CLASSES
# ============================================================

anomaly_events = filtered_events[
    filtered_events["event_label"] == "anomaly"
].head(MAX_EVENTS_PER_CLASS)

normal_events = filtered_events[
    filtered_events["event_label"] == "normal"
].head(MAX_EVENTS_PER_CLASS)


# ============================================================
# PROCESSAR DADOS
# ============================================================

Xa_train, ya_train, Xa_test, ya_test = process_events(
    anomaly_events,
    label=1
)

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

scaler = MinMaxScaler()

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


# ============================================================
# PESOS DAS CLASSES
# ============================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train.cpu().numpy()),
    y=y_train.cpu().numpy().flatten()
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)


#%% ============================================================
# MODELO MLP
# ============================================================

INPUT_SIZE = X_train.shape[1]


class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(INPUT_SIZE, 32),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(32, 16),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(16, 8),

            nn.ReLU(),

            nn.Linear(8, 1)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# INSTANCIAR MODELO
# ============================================================

model = MLP().to(DEVICE)


# ============================================================
# LOSS E OTIMIZADOR
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=class_weights[1]
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


#%% ============================================================
# TREINAMENTO
# ============================================================

print("\n====================================")
print("INICIANDO TREINAMENTO MLP...")
print("====================================\n")

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(
            outputs,
            batch_y
        )

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

    y_pred_logits = model(X_test)

    y_pred_prob = torch.sigmoid(
        y_pred_logits
    )

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