#%% ============================================================
# EnQode-like embedding + QSVM (kernel pré-computado)
# MNIST (10 classes) com PCA -> 2^n amplitudes
#
# Inclui:
#  - EnQode offline/online com multi-start
#  - Fidelidades médias offline/online
#  - QSVM com tuning de C (SVC kernel="precomputed")
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

N_QUBITS = 4                 # 2^4 = 16 amplitudes
N_LAYERS = 6                 # mais camadas -> maior expressividade
N_SAMPLES = 300              # quantas imagens usar do MNIST
FIDELITY_CUTOFF = 0.95       # alvo para escolher nº de clusters

OFFLINE_ITERS = 800          # iterações para centróides
ONLINE_ITERS = 300           # iterações para amostras
LR_OFFLINE = 0.02
LR_ONLINE = 0.02
N_RESTARTS_OFFLINE = 3       # multi-start para centróides

C_GRID = (0.1, 1, 10, 100)   # grid de C para QSVM

np.random.seed(SEED)
pnp.random.seed(SEED)


#%% ============================================================
# 2. Carregar e pré-processar MNIST (PCA -> 2^n)
# ============================================================
def load_mnist_subset(n_samples=N_SAMPLES):
    print("Carregando MNIST via fetch_openml...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(float) / 255.0
    y = mnist.target.astype(int)

    # Embaralhar para pegar subconjunto mais diverso
    idx = np.random.permutation(len(X))[:n_samples]
    X = X[idx]
    y = y[idx]
    print(f"X_raw shape = {X.shape}, y shape = {y.shape}")
    return X, y


def preprocess_for_enqode(X_raw, n_qubits=N_QUBITS):
    """
    PCA -> 2^n dimensões, shift para positividade e normalização ||x|| = 1.
    """
    n_features = 2 ** n_qubits
    print(f"\nPré-processando para EnQode (PCA -> 2^{n_qubits} = {n_features} dimensões)...")

    pca = PCA(n_components=n_features, random_state=SEED)
    X_pca = pca.fit_transform(X_raw)

    # shift para valores >= 0
    mins = X_pca.min(axis=0)
    X_shifted = X_pca - mins

    # normalizar por norma L2
    norms = np.linalg.norm(X_shifted, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    X_norm = X_shifted / norms

    print("X_proc shape =", X_norm.shape)
    return X_norm


#%% ============================================================
# 3. EnQode ansatz: RZ + CY em padrão X-shape / arrow
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


dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="autograd")
def enqode_state(theta):
    """
    theta: array com shape (N_LAYERS, N_QUBITS)
    Retorna o estado quântico |psi(theta)> (statevector).
    """
    wires = range(N_QUBITS)

    # pré-rotação (como no artigo)
    for w in wires:
        qml.RX(-pnp.pi / 2, wires=w)

    # camadas variacionais
    for l in range(N_LAYERS):
        enqode_layer(theta[l], wires=wires, layer_idx=l)

    # pós-rotação
    for w in wires:
        qml.RX(-pnp.pi / 2, wires=w)
        qml.RY(-pnp.pi / 2, wires=w)

    return qml.state()


from pennylane import math as qmlmath

def fidelity(state, target):
    """
    Fidelidade |<state|target>|^2 para estados PUROS (vetores 1D).

    Implementada "na mão" usando apenas operações simples
    (conjugado, produto elemento a elemento e soma),
    para funcionar bem com o autograd.
    """
    # overlap = <state | target>
    overlap = qmlmath.sum(qmlmath.conj(state) * target)
    # |overlap|^2 = Re(overlap)^2 + Im(overlap)^2
    re = qmlmath.real(overlap)
    im = qmlmath.imag(overlap)
    return re**2 + im**2


def loss_fid(theta, target_vec):
    """
    Custo = 1 - fidelidade entre |psi(theta)> e target_vec.
    target_vec deve ser normalizado (norma 1).
    """
    psi = enqode_state(theta)
    return 1.0 - fidelity(psi, target_vec)



#%% ============================================================
# 4. Otimização offline/online (multi-start para centróides)
# ============================================================
def optimize_for_vector_single_start(target_vec, n_iters, init_theta=None, lr=0.05):
    """
    Otimização de fidelidade para UM chute inicial.
    """
    if init_theta is None:
        theta = 0.01 * pnp.random.randn(N_LAYERS, N_QUBITS)
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

        # early stopping se fidelidade já está muito alta
        if 1.0 - float(cost_val) >= 0.98:
            break

    return best_theta, 1.0 - best_cost


def optimize_for_vector(target_vec, n_iters=OFFLINE_ITERS, init_theta=None,
                        lr=LR_OFFLINE, n_restarts=1):
    """
    Multi-start: tenta vários inícios e escolhe o melhor.
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
    Fase OFFLINE: K-means + otimização de centróides
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
            n_restarts=1     # geralmente 1 basta na fase online
        )
        thetas_out.append(theta_i)
        fids_out.append(float(fid_i))

    return thetas_out, np.array(fids_out)


#%% ============================================================
# 5. Construir estados quânticos e kernel
# ============================================================
def build_states_from_thetas(thetas_list):
    """
    Avalia o estado |psi(theta)> para cada theta da lista.
    """
    psi_list = []
    for i, th in enumerate(thetas_list):
        psi = enqode_state(pnp.array(th, dtype=float))
        psi_list.append(np.array(psi))  # converte p/ numpy puro
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


#%% ============================================================
# 6. Tuning de C para QSVM
# ============================================================
def tune_svm_C(K_train_full, y_train_full, C_grid=C_GRID, val_size=0.2, seed=SEED):
    """
    Faz split train/val dentro do conjunto de treino e
    escolhe C pelo melhor acerto em validação.
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
# 7. MAIN
# ============================================================
def main():
    # 1) Dados
    X_raw, y = load_mnist_subset(N_SAMPLES)
    X_proc = preprocess_for_enqode(X_raw, n_qubits=N_QUBITS)

    # 2) Split train/test
    idx_all = np.arange(len(X_proc))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx_all, y, test_size=0.3, random_state=SEED, stratify=y
    )

    X_train_proc = X_proc[idx_train]
    X_test_proc = X_proc[idx_test]

    # 3) OFFLINE (Treino)
    kmeans, centroid_thetas, fids_offline = offline_optimization_kmeans(X_train_proc)

    print("\nFidelidade média OFFLINE (centróides, treino):", fids_offline.mean())

    # 4) ONLINE (Treino e Teste)
    theta_train, fids_online_train = online_optimization(X_train_proc, kmeans, centroid_thetas)
    theta_test, fids_online_test = online_optimization(X_test_proc, kmeans, centroid_thetas)

    print("Fidelidade média ONLINE (treino):", fids_online_train.mean())
    print("Fidelidade média ONLINE (teste): ", fids_online_test.mean())

    print("\n================ RESULTADOS FINAIS ENQODE ================")
    print(f"N_QUBITS = {N_QUBITS}, N_LAYERS = {N_LAYERS}")
    print(f"N_SAMPLES = {N_SAMPLES}")
    print(f"Fidelidade média OFFLINE (centróides): {fids_offline.mean():.6f}")
    print(f"Fidelidade média ONLINE (treino):       {fids_online_train.mean():.6f}")
    print(f"Fidelidade média ONLINE (teste):        {fids_online_test.mean():.6f}")

    # 5) Construir estados quânticos
    print("\n=== Construindo estados quânticos para QSVM ===")
    psi_train = build_states_from_thetas(theta_train)
    psi_test = build_states_from_thetas(theta_test)

    # 6) Kernel
    print("\n=== Montando matrizes de kernel K_train e K_test ===")
    K_train = qkernel(psi_train, psi_train)
    K_test = qkernel(psi_test, psi_train)
    print("Shapes: K_train =", K_train.shape, "| K_test =", K_test.shape)

    # 7) QSVM com tuning de C
    print("\n=== Tuning de C para QSVM ===")
    best_C = tune_svm_C(K_train, y_train, C_grid=C_GRID)

    print("\n=== Treinando QSVM final com melhor C ===")
    svm = SVC(kernel="precomputed", C=best_C)
    svm.fit(K_train, y_train)

    y_pred = svm.predict(K_test)

    print("\n================ RESULTADOS QSVM (kernel EnQode) ================")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
