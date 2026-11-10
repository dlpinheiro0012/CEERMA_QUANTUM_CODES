#%% ============================================================
# EnQode-like QNN (PennyLane) vs CNN (TensorFlow/Keras)
# Classificação MNIST (0–9) via One-vs-Rest (10 QNNs)
# ============================================================

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

import pennylane as qml
from pennylane import numpy as pnp

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

#%% ============================================================
# 1. Hiperparâmetros
# ============================================================
N_QUBITS = 4                  # 2^4 = 16 amplitudes
N_FEATURES = 2 ** N_QUBITS
N_LAYERS = 3                  # camadas EnQode-like (RZ + CY)
N_CLASSES = 10                # dígitos 0–9
SEED = 42

np.random.seed(SEED)
pnp.random.seed(SEED)
tf.random.set_seed(SEED)


#%% ============================================================
# 2. Pré-processamento PCA + normalização para amplitude embedding
# ============================================================
def preprocess_for_enqode(X, n_components=N_FEATURES):
    X = X.astype(float)
    pca = PCA(n_components=n_components, random_state=SEED)
    X_pca = pca.fit_transform(X)

    mins = X_pca.min(axis=0)
    X_shifted = X_pca - mins

    norms = np.linalg.norm(X_shifted, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    X_norm = X_shifted / norms

    return X_norm


#%% ============================================================
# 3. EnQode-like Variational Block (RZ + CY alternados)
# ============================================================
def enqode_layer(params, wires, layer_idx):
    # RZ em todos os qubits
    for i, w in enumerate(wires):
        qml.RZ(params[i], wires=w)

    # CY alternado (X-shape / arrow)
    if layer_idx % 2 == 0:
        pairs = [(wires[i], wires[i+1]) for i in range(0, len(wires)-1, 2)]
    else:
        pairs = [(wires[i], wires[i+1]) for i in range(1, len(wires)-1, 2)]

    for c, t in pairs:
        qml.CY(wires=[c, t])


#%% ============================================================
# 4. Definição do QNN (AmplitudeEmbedding + EnQode-like layers)
# ============================================================
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="autograd")
def qnode_enqode(x, weights):
    """
    x: vetor normalizado (16 componentes)
    weights: parâmetros (N_LAYERS x N_QUBITS)
    """
    qml.AmplitudeEmbedding(x, wires=range(N_QUBITS), normalize=False)

    # pré-rotação
    for w in range(N_QUBITS):
        qml.RX(-pnp.pi/2, wires=w)

    # camadas variacionais
    for l in range(N_LAYERS):
        enqode_layer(weights[l], wires=range(N_QUBITS), layer_idx=l)

    # pós-rotação
    for w in range(N_QUBITS):
        qml.RX(-pnp.pi/2, wires=w)
        qml.RY(-pnp.pi/2, wires=w)

    return qml.expval(qml.PauliZ(0))


def qnn_predict_batch(X, weights):
    return pnp.array([qnode_enqode(x, weights) for x in X])


#%% ============================================================
# 5. Loss + Treino do QNN (binário, one-vs-rest)
# ============================================================
def loss_fn(weights, X, y_binary):
    """
    y_binary: rótulos 0/1 para UMA classe específica.
    Saída do QNN é em [-1,1]; usamos y_scaled em {-1,+1}.
    """
    y_scaled = 2 * y_binary - 1
    preds = qnn_predict_batch(X, weights)
    return pnp.mean((preds - y_scaled) ** 2)


def train_qnn_binary(X_train, y_binary, epochs=15, lr=0.1, batch_size=16):
    weights = 0.01 * pnp.random.randn(N_LAYERS, N_QUBITS)
    opt = qml.AdamOptimizer(lr)
    n = len(X_train)

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n)
        Xb = X_train[perm]
        yb = y_binary[perm]

        for start in range(0, n, batch_size):
            end = start + batch_size
            Xmini = Xb[start:end]
            ymini = yb[start:end]

            weights, batch_loss = opt.step_and_cost(
                lambda w: loss_fn(w, Xmini, ymini), weights
            )

        if epoch % 5 == 0 or epoch == 1:
            full_loss = loss_fn(weights, X_train, y_binary)
            print(f"    [QNN binário] Epoch {epoch:03d} | Loss: {float(full_loss):.4f}")

    return weights


def train_qnn_one_vs_rest(X_train, y_train):
    """
    Treina 1 QNN binário para cada classe k = 0..9.
    Retorna lista de weights, um por classe.
    """
    weights_list = []
    for k in range(N_CLASSES):
        print(f"\n=== Treinando QNN para classe {k} (one-vs-rest) ===")
        y_bin = (y_train == k).astype(float)
        w_k = train_qnn_binary(X_train, pnp.array(y_bin, dtype=float))
        weights_list.append(w_k)
    return weights_list


def qnn_predict_multiclass(X, weights_list):
    """
    Para cada classe k, calcula score em [-1,1] via QNN_k.
    Retorna:
        y_pred: argmax dos scores (classe final 0–9)
        scores: matriz (n amostras, n_classes)
    """
    all_scores = []
    for k, w in enumerate(weights_list):
        preds_k = qnn_predict_batch(X, w)  # (n_samples,)
        all_scores.append(preds_k)

    scores = np.stack(all_scores, axis=1)  # (n_samples, N_CLASSES)
    y_pred = np.argmax(scores, axis=1)
    return np.array(y_pred, dtype=int), scores


#%% ============================================================
# 6. CNN clássica multiclasse
# ============================================================
def build_cnn(input_shape=(28, 28, 1), n_classes=N_CLASSES):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


#%% ============================================================
# 7. MAIN
# ============================================================
def main():
    print("Carregando MNIST (10 classes)…")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data
    y = mnist.target.astype(int)

    X = X / 255.0

    # reduz para não demorar demais
    N = 300   # pode aumentar depois (ex: 1000)
    X = X[:N]
    y = y[:N]

    print("Pré-processando (PCA → amplitude embedding)…")
    X_enq = preprocess_for_enqode(X)
    X_cnn = X.reshape(-1, 28, 28, 1)

    # mesmo split para QNN e CNN
    X_train_q, X_test_q, idx_train, idx_test, y_train, y_test = train_test_split(
        X_enq,
        np.arange(len(X)),
        y,
        test_size=0.3,
        random_state=SEED,
        stratify=y
    )
    X_train_c = X_cnn[idx_train]
    X_test_c = X_cnn[idx_test]

    # -----------------------------
    # QNN one-vs-rest
    # -----------------------------
    print("\n=== Treinando 10 QNNs (one-vs-rest) ===")
    weights_list = train_qnn_one_vs_rest(X_train_q, y_train)

    print("\n=== Avaliando QNN multiclasse ===")
    y_pred_qnn, _ = qnn_predict_multiclass(X_test_q, weights_list)
    print("\n--- Resultados QNN (one-vs-rest) ---")
    print(classification_report(y_test, y_pred_qnn))

    # -----------------------------
    # CNN multiclasse
    # -----------------------------
    print("\n=== Treinando CNN multiclasse ===")
    cnn = build_cnn()
    cnn.fit(
        X_train_c, y_train,
        epochs=4,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    probs_cnn = cnn.predict(X_test_c)
    y_pred_cnn = np.argmax(probs_cnn, axis=1)

    print("\n--- Resultados CNN ---")
    print(classification_report(y_test, y_pred_cnn))


if __name__ == "__main__":
    main()
