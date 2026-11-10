#%% ============================================================
# IMPORTS
# ============================================================

import numpy as np

from sklearn.model_selection import train_test_split

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

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

import pennylane as qml


#%% ============================================================
# CAMINHO DO NPZ
# ============================================================

DATASET_PATH = (
    r"C:\Users\Arthur\Downloads"
    r"\wind_turbine_binary_smote_v2-balanceado.npz"
)


#%% ============================================================
# PARÂMETROS
# ============================================================

BATCH_SIZE = 8

EPOCHS = 50

LEARNING_RATE = 0.00001


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", DEVICE)


#%% ============================================================
# CARREGAR NPZ
# ============================================================

data = np.load(DATASET_PATH)

print("\nChaves do NPZ:")

print(data.files)


#%% ============================================================
# EXTRAIR DADOS
# ============================================================

X = data["X"]

y = data["y"]


print("\nShapes originais:")

print("X:", X.shape)

print("y:", y.shape)

print("\nShape da primeira amostra:")

print(X[0].shape)


#%% ============================================================
# EXTRAÇÃO DE FEATURES ESTATÍSTICAS
# ============================================================

def extract_features(signal):

    # ========================================================
    # GARANTIR ARRAY 1D
    # ========================================================

    signal = np.array(signal).flatten()

    # ========================================================
    # FEATURES
    # ========================================================

    mean = np.mean(signal)

    std = np.std(signal)

    rms = np.sqrt(np.mean(signal**2))

    maximum = np.max(signal)

    minimum = np.min(signal)

    peak_to_peak = maximum - minimum

    # ========================================================
    # ZERO PADDING
    # ========================================================

    features = [

        mean,
        std,
        rms,
        maximum,
        minimum,
        peak_to_peak,

        0.0,
        0.0
    ]

    # ========================================================
    # SANITIZAÇÃO
    # ========================================================

    features = np.nan_to_num(

        features,

        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return features


#%% ============================================================
# EXTRAIR FEATURES DE TODAS AS AMOSTRAS
# ============================================================

X_features = []

for sample in X:

    feats = extract_features(sample)

    X_features.append(feats)


X_features = np.array(X_features)


print("\nShape após feature extraction:")

print(X_features.shape)


#%% ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_features,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


print("\nShapes splitados:")

print("X_train:", X_train.shape)

print("X_test :", X_test.shape)

print("y_train:", y_train.shape)

print("y_test :", y_test.shape)


#%% ============================================================
# NORMALIZAÇÃO
# ============================================================

scaler = MinMaxScaler(
    feature_range=(0, 1)
)

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


#%% ============================================================
# SANITIZAÇÃO
# ============================================================

X_train = np.nan_to_num(
    X_train,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_test = np.nan_to_num(
    X_test,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


#%% ============================================================
# EVITAR VETORES NULOS
# ============================================================

epsilon = 1e-8

for i in range(len(X_train)):

    norm = np.linalg.norm(X_train[i])

    if norm < epsilon:

        X_train[i] += epsilon


for i in range(len(X_test)):

    norm = np.linalg.norm(X_test[i])

    if norm < epsilon:

        X_test[i] += epsilon


#%% ============================================================
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


#%% ============================================================
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
# NÚMERO DE QUBITS
# ============================================================

# 8 amplitudes -> 3 qubits

N_QUBITS = 3

print("\nNúmero final de qubits:", N_QUBITS)


#%% ============================================================
# DEVICE QUÂNTICO
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


#%% ============================================================
# CIRCUITO QUÂNTICO
# ============================================================

@qml.qnode(
    dev,
    interface="torch"
)

def quantum_circuit(inputs, weights):

    # ========================================================
    # AMPLITUDE ENCODING
    # ========================================================

    qml.AmplitudeEmbedding(

        features=inputs,

        wires=range(N_QUBITS),

        normalize=True,

        pad_with=0.0
    )

    # ========================================================
    # PQC
    # ========================================================

    for layer in range(weights.shape[0]):

        for i in range(N_QUBITS):

            qml.RY(
                weights[layer, i, 0],
                wires=i
            )

            qml.RZ(
                weights[layer, i, 1],
                wires=i
            )

            qml.RY(
                weights[layer, i, 2],
                wires=i
            )

        # ====================================================
        # ENTANGLEMENT
        # ====================================================

        for i in range(N_QUBITS - 1):

            qml.CZ(
                wires=[i, i + 1]
            )

    # ========================================================
    # MEDIDAS
    # ========================================================

    return [

        qml.expval(qml.PauliZ(i))

        for i in range(N_QUBITS)
    ]


#%% ============================================================
# SHAPE DOS PESOS
# ============================================================

weight_shapes = {

    "weights": (
        1,
        N_QUBITS,
        3
    )
}


#%% ============================================================
# CAMADA QUÂNTICA
# ============================================================

qlayer = qml.qnn.TorchLayer(
    quantum_circuit,
    weight_shapes
)


#%% ============================================================
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

        # ====================================================
        # CAMADA QUÂNTICA
        # ====================================================

        x = self.qlayer(x)

        # ====================================================
        # MLP
        # ====================================================

        x = self.relu(
            self.fc1(x)
        )

        x = self.relu(
            self.fc2(x)
        )

        x = self.fc3(x)

        return x


#%% ============================================================
# INSTANCIAR MODELO
# ============================================================

model = HybridModel().to(DEVICE)


#%% ============================================================
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

        # ====================================================
        # FORWARD
        # ====================================================

        outputs = model(batch_x)

        # ====================================================
        # EVITAR NAN
        # ====================================================

        outputs = torch.nan_to_num(
            outputs,
            nan=0.0
        )

        # ====================================================
        # LOSS
        # ====================================================

        loss = criterion(
            outputs,
            batch_y
        )

        # ====================================================
        # BACKPROP
        # ====================================================

        loss.backward()

        # ====================================================
        # CLIP GRADIENT
        # ====================================================

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        epoch_loss += loss.item()

    print(

        f"Epoch {epoch+1}/{EPOCHS} "
        f"- Loss: {epoch_loss:.4f}"
    )


#%% ============================================================
# TESTE
# ============================================================

model.eval()

with torch.no_grad():

    # ========================================================
    # LOGITS
    # ========================================================

    y_pred_logits = model(X_test)

    # ========================================================
    # SANITIZAÇÃO
    # ========================================================

    y_pred_logits = torch.nan_to_num(
        y_pred_logits,
        nan=0.0
    )

    # ========================================================
    # PROBABILIDADES
    # ========================================================

    y_pred_prob = torch.sigmoid(
        y_pred_logits
    )

    # ========================================================
    # BINARIZAÇÃO
    # ========================================================

    y_pred = (
        y_pred_prob > 0.5
    ).float()


#%% ============================================================
# CONVERTER PARA NUMPY
# ============================================================

y_true = y_test.cpu().numpy()

y_pred = y_pred.cpu().numpy()


#%% ============================================================
# MÉTRICAS
# ============================================================

acc = accuracy_score(
    y_true,
    y_pred
)

balanced_acc = balanced_accuracy_score(
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


print(f"\nAccuracy          : {acc:.4f}")

print(f"Balanced Accuracy : {balanced_acc:.4f}")

print(f"Precision         : {prec:.4f}")

print(f"Recall            : {rec:.4f}")

print(f"F1-Score          : {f1:.4f}")


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