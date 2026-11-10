#%% ============================================================
# EnQode-like embedding + QSVM + QNN + QAE
# MNIST com PCA -> 2^n amplitudes
#
# - EnQode offline/online com multi-start
# - Fidelidades médias offline/online (treino e teste)
# - QSVM com kernel EnQode (10 classes)
# - QNN binário (0 vs 1) usando estados EnQode
# - QAE treinado só no dígito 0 (detecção de "normal x anômalo")
# ============================================================

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import pennylane as qml
from pennylane import numpy as pnp
from pennylane import math as qmlmath


#%% ============================================================
# 1. Hiperparâmetros
# ============================================================
SEED = 42

N_QUBITS = 4                   # 2^4 = 16 amplitudes
N_LAYERS_ENQ = 6               # EnQode ansatz depth
N_SAMPLES = 1000                # nº de imagens do MNIST usadas

FIDELITY_CUTOFF = 0.95         # alvo para escolher nº de clusters (inner product médio)

OFFLINE_ITERS = 800            # iterações para centróides EnQode
ONLINE_ITERS = 300             # iterações para amostras (warm start)
LR_OFFLINE = 0.02
LR_ONLINE = 0.02
N_RESTARTS_OFFLINE = 3         # multi-start para centróides

C_GRID = (0.1, 1, 10, 100)     # grid de C para QSVM

# QNN (binário 0 vs 1) – usamos estados EnQode como input
N_LAYERS_QNN = 3
LR_QNN = 0.1
EPOCHS_QNN = 30
BATCH_QNN = 32

# QAE (autoencoder variacional simples, sem compressão explícita)
DIGIT_NORMAL_QAE = 0           # treinar QAE só neste dígito
N_LAYERS_QAE_ENC = 3
N_LAYERS_QAE_DEC = 3
LR_QAE = 0.05
EPOCHS_QAE = 25
BATCH_QAE = 32

np.random.seed(SEED)
pnp.random.seed(SEED)


#%% ============================================================
# 2. Dados MNIST + Pré-processamento (PCA -> 2^n)
# ============================================================
def load_mnist_subset(n_samples=N_SAMPLES):
    print("Carregando MNIST via fetch_openml...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(float) / 255.0
    y = mnist.target.astype(int)

    # Embaralhar e pegar subconjunto
    idx = np.random.permutation(len(X))[:n_samples]
    X = X[idx]
    y = y[idx]
    print(f"X_raw shape = {X.shape}, y shape = {y.shape}")
    return X, y


def preprocess_for_enqode(X_raw, n_qubits=N_QUBITS):
    """
    PCA -> 2^n dimensões, shift para positividade e normalização ||x|| = 1.
    Isso gera vetores prontos para serem alvo do EnQode.
    """
    n_features = 2 ** n_qubits
    print(f"\nPré-processando para EnQode (PCA -> 2^{n_qubits} = {n_features} dimensões)...")

    pca = PCA(n_components=n_features, random_state=SEED)
    X_pca = pca.fit_transform(X_raw)

    # Shift para valores >= 0
    mins = X_pca.min(axis=0)
    X_shifted = X_pca - mins

    # Normalizar por norma L2
    norms = np.linalg.norm(X_shifted, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    X_norm = X_shifted / norms

    print("X_proc shape =", X_norm.shape)
    return X_norm


#%% ============================================================
# 3. EnQode ansatz: RZ + CY alternado
# ============================================================
def enqode_layer(params, wires, layer_idx):
    # rotações RZ
    for i, w in enumerate(wires):
        qml.RZ(params[i], wires=w)

    # entanglement CY alternado
    if layer_idx % 2 == 0:
        pairs = [(wires[i], wires[i + 1]) for i in range(0, len(wires) - 1, 2)]
    else:
        pairs = [(wires[i], wires[i + 1]) for i in range(1, len(wires) - 1, 2)]

    for control, target in pairs:
        qml.CY(wires=[control, target])


dev_enq = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev_enq, interface="autograd")
def enqode_state(theta):
    """
    theta: array com shape (N_LAYERS_ENQ, N_QUBITS)
    Retorna o estado quântico |psi(theta)> (statevector).
    """
    wires = range(N_QUBITS)

    # pré-rotação (como no paper EnQode)
    for w in wires:
        qml.RX(-pnp.pi / 2, wires=w)

    # EnQode layers
    for l in range(N_LAYERS_ENQ):
        enqode_layer(theta[l], wires=wires, layer_idx=l)

    # pós-rotação
    for w in wires:
        qml.RX(-pnp.pi / 2, wires=w)
        qml.RY(-pnp.pi / 2, wires=w)

    return qml.state()


def fidelity_pure(state, target):
    """
    Fidelidade para estados PUROS: |<state|target>|^2.
    Implementada "na mão" (overlap complexo) para funcionar bem com autograd.
    """
    overlap = qmlmath.sum(qmlmath.conj(state) * target)
    re = qmlmath.real(overlap)
    im = qmlmath.imag(overlap)
    return re**2 + im**2


def loss_fid(theta, target_vec):
    """
    Custo = 1 - fidelidade(|psi(theta)>, target_vec).
    target_vec deve ser normalizado (norma 1).
    """
    psi = enqode_state(theta)
    return 1.0 - fidelity_pure(psi, target_vec)


#%% ============================================================
# 4. Otimização OFFLINE / ONLINE (EnQode)
# ============================================================
def optimize_for_vector_single_start(target_vec, n_iters, init_theta=None, lr=0.05):
    """Otimização de fidelidade para UM chute inicial."""
    if init_theta is None:
        theta = 0.01 * pnp.random.randn(N_LAYERS_ENQ, N_QUBITS)
    else:
        theta = pnp.array(init_theta, dtype=float)

    opt = qml.AdamOptimizer(lr)
    best_theta = theta
    best_cost = 1.0

    for it in range(1, n_iters + 1):
        theta, cost_val = opt.step_and_cost(lambda th: loss_fid(th, target_vec), theta)

        if cost_val < best_cost:
            best_cost = cost_val
            best_theta = theta

        # early stopping se fidelidade já estiver altíssima
        if 1.0 - float(cost_val) >= 0.99:
            break

    return best_theta, 1.0 - best_cost


def optimize_for_vector(target_vec, n_iters=OFFLINE_ITERS, init_theta=None,
                        lr=LR_OFFLINE, n_restarts=1):
    """
    Multi-start: testa vários inícios e escolhe o melhor.
    """
    best_theta_global = None
    best_fid_global = -1.0

    for r in range(n_restarts):
        theta0 = init_theta if (init_theta is not None and r == 0) else None
        theta_r, fid_r = optimize_for_vector_single_start(
            target_vec, n_iters, init_theta=theta0, lr=lr
        )
        if fid_r > best_fid_global:
            best_fid_global = fid_r
            best_theta_global = theta_r

    return best_theta_global, best_fid_global


def determine_num_clusters(processed_vecs, distance_cutoff=FIDELITY_CUTOFF, random_state=SEED):
    """
    Imita lógica do artigo: escolhe nº de clusters tal que
    <x, centroid> médio >= sqrt(distance_cutoff).
    """
    print("\n=== Determinando número de clusters ===")
    X = processed_vecs
    candidate_ks = list(range(2, 6)) + list(range(5, 50, 5))

    for k in candidate_ks:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        total_inner = 0.0
        for i, x in enumerate(X):
            c = centroids[labels[i]]
            total_inner += np.dot(x, c)
        mean_inner = total_inner / len(X)
        print(f"[Clustering] k={k:3d} | média <x, centroid> = {mean_inner:.4f}")

        if mean_inner >= np.sqrt(distance_cutoff):
            print(f"=> atingiu cutoff; usando k = {k}")
            return k

    print(f"=> não atingiu cutoff; usando máximo k = {candidate_ks[-1]}")
    return candidate_ks[-1]


def offline_optimization_kmeans(processed_vecs):
    """
    Fase OFFLINE: K-means + otimização dos centróides (EnQode).
    """
    n_clusters = determine_num_clusters(processed_vecs)
    print("\n=== Rodando K-means (OFFLINE) ===")
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    kmeans.fit(processed_vecs)

    centroids = kmeans.cluster_centers_
    centroid_thetas = []
    centroid_fids = []

    print("\n=== Otimização offline por centróide ===")
    for idx_c, c in enumerate(centroids):
        print(f"\n[Centróide {idx_c}]")
        c_norm = c / np.linalg.norm(c)
        theta_c, fid_c = optimize_for_vector(
            pnp.array(c_norm, dtype=float),
            n_iters=OFFLINE_ITERS,
            lr=LR_OFFLINE,
            n_restarts=N_RESTARTS_OFFLINE
        )
        centroid_thetas.append(theta_c)
        centroid_fids.append(float(fid_c))
        print(f"  Fidelidade centróide {idx_c}: {fid_c:.4f}")

    return kmeans, centroid_thetas, np.array(centroid_fids)


def online_optimization(processed_vecs, kmeans, centroid_thetas):
    """
    Fase ONLINE: para cada amostra, pega centróide mais próximo
    e refina os parâmetros (warm start).
    """
    labels = kmeans.predict(processed_vecs)
    thetas_out = []
    fids_out = []

    print("\n=== Otimização ONLINE para amostras ===")
    for i, x in enumerate(processed_vecs):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Otimizando amostra {i+1}/{len(processed_vecs)}")

        x_norm = x / np.linalg.norm(x)
        c_idx = labels[i]
        theta_init = centroid_thetas[c_idx]

        theta_i, fid_i = optimize_for_vector(
            pnp.array(x_norm, dtype=float),
            n_iters=ONLINE_ITERS,
            init_theta=theta_init,
            lr=LR_ONLINE,
            n_restarts=1
        )
        thetas_out.append(theta_i)
        fids_out.append(float(fid_i))

    return thetas_out, np.array(fids_out)


#%% ============================================================
# 5. Estados EnQode + kernel QSVM
# ============================================================
def build_states_from_thetas(thetas_list):
    """
    Avalia o estado |psi(theta)> para cada theta da lista.
    """
    psi_list = []
    for th in thetas_list:
        psi = enqode_state(pnp.array(th, dtype=float))
        psi_list.append(np.array(psi))  # numpy puro
    return np.array(psi_list)


def qkernel(states_A, states_B):
    """
    Kernel K_ij = |<psi_i | psi_j>|^2
    """
    nA, nB = len(states_A), len(states_B)
    K = np.zeros((nA, nB))
    for i in range(nA):
        for j in range(nB):
            overlap = np.vdot(states_A[i], states_B[j])
            K[i, j] = np.abs(overlap) ** 2
    return K


def tune_svm_C(K_train_full, y_train_full, C_grid=C_GRID, val_size=0.2, seed=SEED):
    """
    Split interno train/val para escolher C pelo melhor acerto em validação.
    """
    n = K_train_full.shape[0]
    idx = np.arange(n)

    idx_tr, idx_val, y_tr, y_val = train_test_split(
        idx, y_train_full, test_size=val_size, random_state=seed, stratify=y_train_full
    )

    K_tr = K_train_full[np.ix_(idx_tr, idx_tr)]
    K_val = K_train_full[np.ix_(idx_val, idx_tr)]

    best_C = None
    best_acc = -1.0

    for C in C_grid:
        clf = SVC(kernel="precomputed", C=C)
        clf.fit(K_tr, y_tr)
        acc = clf.score(K_val, y_val)
        print(f"[Tuning QSVM] C={C} -> acc_val={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_C = C

    print(f"Melhor C encontrado: {best_C} (acc_val={best_acc:.4f})")
    return best_C


#%% ============================================================
# 6. QNN binário (0 vs 1) usando estados EnQode
# ============================================================
dev_qnn = qml.device("default.qubit", wires=N_QUBITS)

def qnn_ansatz(weights, wires):
    for l in range(N_LAYERS_QNN):
        # rotações RY por qubit
        for i, w in enumerate(wires):
            qml.RY(weights[l, i], wires=w)
        # entanglement simples em linha
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])

@qml.qnode(dev_qnn, interface="autograd")
def qnode_qnn(statevec, weights):
    """
    statevec: estado EnQode |psi_ENQ(x)>, tamanho 2^N_QUBITS
    weights: parâmetros do QNN
    """
    # pluga diretamente o estado EnQode na entrada do QNN
    qml.StatePrep(statevec, wires=range(N_QUBITS))
    qnn_ansatz(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))   # saída escalar em [-1,1]


def qnn_loss(weights, X_batch, y_batch):
    """
    y_batch em {0,1}. Mapeamos para {-1,+1} para casar com PauliZ.
    """
    y_scaled = 2 * y_batch - 1
    preds = [qnode_qnn(x, weights) for x in X_batch]
    preds = pnp.stack(preds)
    return pnp.mean((preds - y_scaled) ** 2)


def train_qnn(X_train_q, y_train_q, epochs=EPOCHS_QNN, batch_size=BATCH_QNN, lr=LR_QNN):
    weights = 0.01 * pnp.random.randn(N_LAYERS_QNN, N_QUBITS)
    opt = qml.AdamOptimizer(lr)
    n = len(X_train_q)

    for ep in range(1, epochs + 1):
        perm = np.random.permutation(n)
        Xb = X_train_q[perm]
        yb = y_train_q[perm]

        for start in range(0, n, batch_size):
            end = start + batch_size
            Xmini = Xb[start:end]
            ymini = yb[start:end]

            weights, loss_val = opt.step_and_cost(
                lambda w: qnn_loss(w, Xmini, ymini), weights
            )

        if ep == 1 or ep % 5 == 0:
            full_loss = float(qnn_loss(weights, X_train_q, y_train_q))
            print(f"[QNN] Epoch {ep:03d} | Loss = {full_loss:.4f}")

    return weights


def eval_qnn(weights, X_test_q, y_test_q):
    preds = [qnode_qnn(x, weights) for x in X_test_q]
    preds = pnp.array(preds)
    y_pred = (preds > 0).astype(int)
    print("\n--- Resultados QNN (0 vs 1, estados EnQode) ---")
    print(classification_report(y_test_q, np.array(y_pred, dtype=int)))


#%% ============================================================
# 7. QAE (autoencoder simples) treinado só no dígito 0
# ============================================================
dev_qae = qml.device("default.qubit", wires=N_QUBITS)

def qae_layer(params, wires):
    # Reuso de um bloco tipo EnQode (RZ + CY)
    for i, w in enumerate(wires):
        qml.RZ(params[i], wires=w)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])

def qae_encoder(theta_enc, wires):
    for l in range(N_LAYERS_QAE_ENC):
        qae_layer(theta_enc[l], wires=wires)

def qae_decoder(theta_dec, wires):
    for l in range(N_LAYERS_QAE_DEC):
        qae_layer(theta_dec[l], wires=wires)

@qml.qnode(dev_qae, interface="autograd")
def qae_state(x, theta_enc, theta_dec):
    """
    x: vetor (2^N_QUBITS) já normalizado (usamos X_proc).
    Queremos que o estado final seja o mais próximo possível de |x>.
    """
    wires = range(N_QUBITS)
    qml.AmplitudeEmbedding(x, wires=wires, normalize=False)
    qae_encoder(theta_enc, wires=wires)
    qae_decoder(theta_dec, wires=wires)
    return qml.state()


def qae_loss_thetas(params, X_batch):
    theta_enc, theta_dec = params
    losses = []
    for x in X_batch:
        psi_out = qae_state(x, theta_enc, theta_dec)
        # alvo = o próprio vetor x (estado desejado)
        losses.append(1.0 - fidelity_pure(psi_out, x))
    return pnp.mean(pnp.stack(losses))


def train_qae(X_train_qae, epochs=EPOCHS_QAE, batch_size=BATCH_QAE, lr=LR_QAE):
    theta_enc = 0.01 * pnp.random.randn(N_LAYERS_QAE_ENC, N_QUBITS)
    theta_dec = 0.01 * pnp.random.randn(N_LAYERS_QAE_DEC, N_QUBITS)
    params = (theta_enc, theta_dec)

    opt = qml.AdamOptimizer(lr)
    n = len(X_train_qae)

    for ep in range(1, epochs + 1):
        perm = np.random.permutation(n)
        Xb = X_train_qae[perm]

        for start in range(0, n, batch_size):
            end = start + batch_size
            Xmini = Xb[start:end]
            params, loss_val = opt.step_and_cost(
                lambda pr: qae_loss_thetas(pr, Xmini), params
            )

        if ep == 1 or ep % 5 == 0:
            full_loss = float(qae_loss_thetas(params, X_train_qae))
            print(f"[QAE] Epoch {ep:03d} | Loss = {full_loss:.6f}")

    theta_enc, theta_dec = params
    return theta_enc, theta_dec


def eval_qae(theta_enc, theta_dec, X_train_qae, X_test_proc, y_test):
    # Erro médio no treino (somente dígito normal)
    train_losses = []
    for x in X_train_qae:
        psi_out = qae_state(x, theta_enc, theta_dec)
        train_losses.append(1.0 - fidelity_pure(psi_out, x))
    train_loss_mean = float(pnp.mean(pnp.array(train_losses)))

    # Erro por amostra no teste (todas as classes)
    test_losses = []
    for x in X_test_proc:
        psi_out = qae_state(x, theta_enc, theta_dec)
        test_losses.append(1.0 - fidelity_pure(psi_out, x))
    test_losses = np.array(test_losses)

    print(f"\nErro médio de reconstrução (treino – dígito {DIGIT_NORMAL_QAE}): {train_loss_mean:.6f}")
    print(f"Erro médio de reconstrução (teste – todas classes): {test_losses.mean():.6f}\n")

    print("--- Erro médio de reconstrução por dígito (teste) ---")
    for d in range(10):
        mask_d = (y_test == d)
        if mask_d.sum() > 0:
            print(f"Dígito {d}: erro médio = {test_losses[mask_d].mean():.6f}")


#%% ============================================================
# 8. MAIN
# ============================================================
def main():
    # 1) Dados + pré-processamento para EnQode
    X_raw, y = load_mnist_subset(N_SAMPLES)
    X_proc = preprocess_for_enqode(X_raw, n_qubits=N_QUBITS)

    # 2) Split global train/test
    idx_all = np.arange(len(X_proc))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx_all, y, test_size=0.3, random_state=SEED, stratify=y
    )

    X_train_proc = X_proc[idx_train]
    X_test_proc = X_proc[idx_test]

    # ========================================================
    # EnQode: OFFLINE + ONLINE
    # ========================================================
    kmeans, centroid_thetas, fids_offline = offline_optimization_kmeans(X_train_proc)

    print("\nFidelidade média OFFLINE (centróides, treino):", fids_offline.mean())

    theta_train, fids_online_train = online_optimization(X_train_proc, kmeans, centroid_thetas)
    theta_test, fids_online_test = online_optimization(X_test_proc, kmeans, centroid_thetas)

    print("Fidelidade média ONLINE (treino):", fids_online_train.mean())
    print("Fidelidade média ONLINE (teste): ", fids_online_test.mean())

    print("\n================ RESULTADOS FINAIS ENQODE ================")
    print(f"N_QUBITS = {N_QUBITS}, N_LAYERS_ENQ = {N_LAYERS_ENQ}")
    print(f"N_SAMPLES = {N_SAMPLES}")
    print(f"Fidelidade média OFFLINE (centróides): {fids_offline.mean():.6f}")
    print(f"Fidelidade média ONLINE (treino):       {fids_online_train.mean():.6f}")
    print(f"Fidelidade média ONLINE (teste):        {fids_online_test.mean():.6f}")

    # ========================================================
    # QSVM com kernel EnQode (10 classes)
    # ========================================================
    print("\n=== Construindo estados quânticos EnQode (train/test) ===")
    psi_train = build_states_from_thetas(theta_train)
    psi_test = build_states_from_thetas(theta_test)

    print("\n=== Montando matrizes de kernel K_train e K_test ===")
    K_train = qkernel(psi_train, psi_train)
    K_test = qkernel(psi_test, psi_train)
    print("Shapes: K_train =", K_train.shape, "| K_test =", K_test.shape)

    print("\n=== Tuning de C para QSVM ===")
    best_C = tune_svm_C(K_train, y_train, C_grid=C_GRID)

    print("\n=== Treinando QSVM final com melhor C ===")
    svm = SVC(kernel="precomputed", C=best_C)
    svm.fit(K_train, y_train)
    y_pred = svm.predict(K_test)

    print("\n================ RESULTADOS QSVM (kernel EnQode) ================")
    print(classification_report(y_test, y_pred))

    # ========================================================
    # QNN binário 0 vs 1 usando estados EnQode
    # ========================================================
    print("\n=== QNN EnQode (binário 0 vs 1) ===")
    mask_train_01 = (y_train == 0) | (y_train == 1)
    mask_test_01 = (y_test == 0) | (y_test == 1)

    X_qnn_train = psi_train[mask_train_01]
    y_qnn_train = y_train[mask_train_01].astype(float)

    X_qnn_test = psi_test[mask_test_01]
    y_qnn_test = y_test[mask_test_01].astype(float)

    print(f"Treino QNN: {len(X_qnn_train)} amostras (0 vs 1)")
    print(f"Teste  QNN: {len(X_qnn_test)} amostras (0 vs 1)")

    if len(X_qnn_train) > 0 and len(X_qnn_test) > 0:
        theta_qnn = train_qnn(X_qnn_train, y_qnn_train)
        eval_qnn(theta_qnn, X_qnn_test, y_qnn_test)
    else:
        print("Poucas amostras de 0 vs 1 para QNN – aumente N_SAMPLES ou verifique o subset.")

    # ========================================================
    # QAE treinado só no dígito 0 (detecção de anomalia)
    # ========================================================
    print(f"\n=== Treinando QAE EnQode-like apenas no dígito {DIGIT_NORMAL_QAE} ===")
    mask_train_qae = (y_train == DIGIT_NORMAL_QAE)
    X_qae_train = X_train_proc[mask_train_qae]

    print(f"Amostras para QAE (treino, dígito {DIGIT_NORMAL_QAE}): {len(X_qae_train)}")

    if len(X_qae_train) > 0:
        theta_enc, theta_dec = train_qae(X_qae_train)
        eval_qae(theta_enc, theta_dec, X_qae_train, X_test_proc, y_test)
    else:
        print("Não há amostras suficientes do dígito normal para treinar o QAE.")


if __name__ == "__main__":
    main()


