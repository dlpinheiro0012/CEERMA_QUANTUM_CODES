#%% ============================================================
# Imports
# ============================================================
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import pennylane as qml
from pennylane import numpy as pnp

#%% ============================================================
# Hiperparâmetros globais
# ============================================================
N_QUBITS = 6                 # 2^6 = 64 amplitudes
N_FEATURES = 2 ** N_QUBITS   # tamanho do vetor de entrada
N_LAYERS = 3                 # camadas do ansatz EnQode-like
SEED = 42

# dados / clustering / otimização
N_SAMPLES = 300              # quantas imagens MNIST usar (pode aumentar)
FIDELITY_CUTOFF = 0.95       # alvo médio de fidelidade (como no artigo)
OFFLINE_ITERS = 400          # iterações para centróides
ONLINE_ITERS = 100            # iterações para refinamento por amostra

RESULTS_TAG = f"{N_QUBITS}q_{N_LAYERS}L"

np.random.seed(SEED)
pnp.random.seed(SEED)

#%% ============================================================
# 1. Carregar e pré-processar MNIST
# ============================================================
def load_mnist_subset(n_samples=N_SAMPLES):
    """Carrega MNIST completo e devolve apenas n_samples."""
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(float) / 255.0
    y = mnist.target.astype(int)
    X = X[:n_samples]
    y = y[:n_samples]
    return X, y


def preprocess_for_enqode(X, n_components=N_FEATURES):
    """
    PCA -> 2^n_qubits, shift para valores positivos,
    normalização L2 -> vetor válido para amplitude embedding.
    """
    pca = PCA(n_components=n_components, random_state=SEED)
    X_pca = pca.fit_transform(X)  # (n_samples, n_components)

    # tornar tudo >= 0
    mins = X_pca.min(axis=0)
    X_shifted = X_pca - mins

    # normalizar cada vetor para norma 1
    norms = np.linalg.norm(X_shifted, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    X_norm = X_shifted / norms

    return X_norm

#%% ============================================================
# 2. Determinar número de clusters (K-means + cutoff de fidelidade)
# ============================================================
def determine_num_clusters(processed_vecs, distance_cutoff=FIDELITY_CUTOFF, random_state=SEED):
    """
    Imita a lógica do artigo:
    escolhe n_clusters tal que o inner product médio entre ponto e centróide
    seja >= sqrt(distance_cutoff).
    """
    X = processed_vecs
    # tentar quantidade crescente de clusters
    cluster_counts = list(range(2, 5)) + list(range(5, 50, 5))

    for n_clusters in cluster_counts:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X)

        total_innerprod = 0.0
        for i, vec in enumerate(X):
            c = kmeans.cluster_centers_[labels[i]]
            total_innerprod += np.dot(vec, c)

        total_innerprod /= X.shape[0]

        print(f"[Clustering] k={n_clusters:3d} | média <x, centroid> = {total_innerprod:.4f}")

        if total_innerprod >= np.sqrt(distance_cutoff):
            print(f"=> escolhendo n_clusters = {n_clusters} (atingiu cutoff {distance_cutoff})")
            return n_clusters

    print(f"=> não atingiu cutoff; usando máximo k = {cluster_counts[-1]}")
    return cluster_counts[-1]

#%% ============================================================
# 3. Device + Ansatz EnQode-like (RZ + CY em X/arrow) em PennyLane
# ============================================================
dev = qml.device("default.qubit", wires=N_QUBITS)

def enqode_layer(params, wires, layer_idx):
    """Uma camada do ansatz: RZ em todos os qubits + padrão CY alternado."""
    # rotações RZ
    for i, w in enumerate(wires):
        qml.RZ(params[i], wires=w)

    # padrão de emaranhamento X / arrow
    if layer_idx % 2 == 0:
        pairs = [(wires[i], wires[i+1]) for i in range(0, len(wires) - 1, 2)]
    else:
        pairs = [(wires[i], wires[i+1]) for i in range(1, len(wires) - 1, 2)]

    for c, t in pairs:
        qml.CY(wires=[c, t])


@qml.qnode(dev, interface="autograd")
def enqode_state(theta):
    """
    Circuito que recebe apenas parâmetros theta e prepara um estado
    |psi(theta)> ~ vetor alvo que queremos aproximar.
    """
    # começamos em |0...0>

    # 1) pré-rotação RX(-pi/2)
    for w in range(N_QUBITS):
        qml.RX(-pnp.pi / 2, wires=w)

    # 2) camadas variacionais (RZ + CY)
    for l in range(N_LAYERS):
        enqode_layer(theta[l], wires=range(N_QUBITS), layer_idx=l)

    # 3) pós-rotações RX(-pi/2), RY(-pi/2)
    for w in range(N_QUBITS):
        qml.RX(-pnp.pi / 2, wires=w)
        qml.RY(-pnp.pi / 2, wires=w)

    return qml.state()  # retorna o vetor de estado completo (complexo)

#%% ============================================================
# 4. Fidelidade e função de custo
# ============================================================
# ===============================
# FIDELIDADE DIFERENCIÁVEL
# ===============================
def fidelity(psi, target):
    """Fidelidade diferenciável entre dois estados complexos."""
    re = pnp.real(psi)
    im = pnp.imag(psi)
    re_t = pnp.real(target)
    im_t = pnp.imag(target)

    # inner product real
    inner = pnp.sum(re * re_t + im * im_t)
    return inner**2   # |<psi|phi>|^2


def loss_fid(theta, target):
    psi = enqode_state(theta)
    return 1.0 - fidelity(psi, target)


#%% ============================================================
# 5. Otimização offline (centróides) e online (por amostra)
# ============================================================
def optimize_for_vector(target_vec, n_iters=200, init_theta=None, lr=0.1):
    """
    Minimiza 1 - fidelidade entre o estado do circuito e target_vec.
    Retorna parâmetros otimizados e fidelidade final.
    """
    if init_theta is None:
        theta = 0.01 * pnp.random.randn(N_LAYERS, N_QUBITS)
    else:
        theta = pnp.array(init_theta, requires_grad=True)

    opt = qml.AdamOptimizer(lr)

    for it in range(1, n_iters + 1):
        theta, cost_val = opt.step_and_cost(
            lambda th: loss_fid(th, target_vec), theta
        )

        if it % 50 == 0 or it == 1 or it == n_iters:
            fid_val = 1.0 - float(cost_val)
            print(f"    [Iter {it:03d}] loss={float(cost_val):.6f} | fidelidade={fid_val:.6f}")

    final_state = enqode_state(theta)
    final_fid = float(fidelity(final_state, target_vec))
    return theta, final_fid


def offline_optimization_kmeans(processed_vecs):
    """
    1) Determina n_clusters
    2) Roda K-means
    3) Para cada centróide, otimiza theta_cen para aproximar o centróide.
    """
    print("\n=== Determinando número de clusters ===")
    n_clusters = determine_num_clusters(processed_vecs, distance_cutoff=FIDELITY_CUTOFF)

    print("\n=== Rodando K-means ===")
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    labels = kmeans.fit_predict(processed_vecs)
    centroids = kmeans.cluster_centers_

    centroid_thetas = []
    centroid_fids = []

    print("\n=== Otimização offline por centróide ===")
    for idx, c in enumerate(centroids):
        print(f"\n[Centróide {idx}]")
        # garantir normalização numérica fina
        c_norm = c / np.linalg.norm(c)
        theta_c, fid_c = optimize_for_vector(
            pnp.array(c_norm, dtype=float),
            n_iters=OFFLINE_ITERS,
            lr=0.1
        )
        centroid_thetas.append(theta_c)
        centroid_fids.append(fid_c)
        print(f"  -> fidelidade centróide {idx}: {fid_c:.6f}")

    mean_offline_fid = np.mean(centroid_fids)
    print(f"\nFidelidade média OFFLINE nos centróides: {mean_offline_fid:.6f}")

    return kmeans, centroid_thetas, centroid_fids


def online_optimization_for_dataset(processed_vecs, kmeans, centroid_thetas):
    """
    Para cada ponto:
    - descobre o cluster
    - pega theta do centróide como warm start
    - faz algumas iterações online
    - calcula fidelidade final
    """
    labels = kmeans.predict(processed_vecs)
    fids_online = []

    print("\n=== Otimização ONLINE por amostra (refinamento) ===")
    for i, x in enumerate(processed_vecs):
        cluster_id = labels[i]
        theta_start = centroid_thetas[cluster_id]

        x_norm = x / np.linalg.norm(x)  # só por segurança numérica
        theta_opt, fid_end = optimize_for_vector(
            pnp.array(x_norm, dtype=float),
            n_iters=ONLINE_ITERS,
            init_theta=theta_start,
            lr=0.05
        )
        fids_online.append(fid_end)

        print(f"Amostra {i:03d} (cluster {cluster_id}) -> fidelidade ONLINE = {fid_end:.6f}")

    print(f"\nFidelidade média ONLINE (todas amostras): {np.mean(fids_online):.6f}")
    return fids_online

#%% ============================================================
# 6. Função main
# ============================================================
def main():
    print("Carregando MNIST...")
    X_raw, y = load_mnist_subset(N_SAMPLES)
    print(f"X_raw shape = {X_raw.shape}, y shape = {y.shape}")

    print("\nPré-processando para EnQode (PCA -> 2^n, normalização)...")
    X_proc = preprocess_for_enqode(X_raw, n_components=N_FEATURES)
    print(f"X_proc shape = {X_proc.shape}")

    # OFFLINE: centróides
    kmeans, centroid_thetas, centroid_fids = offline_optimization_kmeans(X_proc)

    # ONLINE: refinamento por amostra
    fids_online = online_optimization_for_dataset(X_proc, kmeans, centroid_thetas)

    print("\n================ RESULTADOS FINAIS ================")
    print(f"N_QUBITS = {N_QUBITS}, N_LAYERS = {N_LAYERS}")
    print(f"N_SAMPLES = {N_SAMPLES}")
    print(f"Fidelidade média OFFLINE (centróides): {np.mean(centroid_fids):.6f}")
    print(f"Fidelidade média ONLINE (amostras):    {np.mean(fids_online):.6f}")
    
    # ---------------------------------------------------
    # Salvar fidelidades para comparação posterior
    # ---------------------------------------------------
    np.save(f"fids_offline_{RESULTS_TAG}.npy", np.array(centroid_fids))
    np.save(f"fids_online_{RESULTS_TAG}.npy",  np.array(fids_online))
    print(f"\nArquivos salvos: fids_offline_{RESULTS_TAG}.npy, fids_online_{RESULTS_TAG}.npy")

    
    # ---------------------------------------------------
    # Gráficos de fidelidade (OFFLINE x ONLINE)
    # ---------------------------------------------------
    
    # Histograma fidelidades OFFLINE (centróides)
    plt.figure(figsize=(6, 4))
    plt.hist(centroid_fids, bins=10, alpha=0.7, edgecolor='black')
    plt.title(f"Fidelidade OFFLINE (centróides) - {N_QUBITS} qubits")
    plt.xlabel("Fidelidade")
    plt.ylabel("Contagem")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Histograma fidelidades ONLINE (todas as amostras)
    plt.figure(figsize=(6, 4))
    plt.hist(fids_online, bins=10, alpha=0.7, edgecolor='black')
    plt.title(f"Fidelidade ONLINE (amostras) - {N_QUBITS} qubits")
    plt.xlabel("Fidelidade")
    plt.ylabel("Contagem")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Boxplot comparando OFFLINE x ONLINE
    plt.figure(figsize=(5, 4))
    plt.boxplot(
        [centroid_fids, fids_online],
        labels=["Offline\n(centróides)", "Online\n(amostras)"]
    )
    plt.ylabel("Fidelidade")
    plt.title(f"Comparação fidelidade - {N_QUBITS} qubits")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


#%% ============================================================
# 7. Executar main
# ============================================================
if __name__ == "__main__":
    main()


#%% ============================================================
# Comparar 4 qubits vs 6 qubits (OFFLINE e ONLINE)
# ============================================================
import os

def load_fids(tag):
    off = np.load(f"fids_offline_{tag}.npy")
    on  = np.load(f"fids_online_{tag}.npy")
    return off, on

# Ajuste estes nomes conforme os RESULTS_TAG usados em cada run:
tag_4q = "4q_3L"
tag_6q = "6q_3L"

if not (os.path.exists(f"fids_offline_{tag_4q}.npy") and
        os.path.exists(f"fids_online_{tag_4q}.npy") and
        os.path.exists(f"fids_offline_{tag_6q}.npy") and
        os.path.exists(f"fids_online_{tag_6q}.npy")):
    print("Arquivos .npy de 4q e 6q ainda não existem. Rode os dois experimentos primeiro.")
else:
    off_4, on_4 = load_fids(tag_4q)
    off_6, on_6 = load_fids(tag_6q)

    # Médias
    m_off_4, m_on_4 = off_4.mean(), on_4.mean()
    m_off_6, m_on_6 = off_6.mean(), on_6.mean()

    print("=== Médias de fidelidade ===")
    print(f"4q OFFLINE: {m_off_4:.4f} | 4q ONLINE: {m_on_4:.4f}")
    print(f"6q OFFLINE: {m_off_6:.4f} | 6q ONLINE: {m_on_6:.4f}")

    # ---------------- Barplot de médias ----------------
    labels = ["Offline", "Online"]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, [m_off_4, m_on_4], width, label="4 qubits")
    plt.bar(x + width/2, [m_off_6, m_on_6], width, label="6 qubits")

    plt.xticks(x, labels)
    plt.ylabel("Fidelidade média")
    plt.title("Fidelidade média - 4q vs 6q")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------- Boxplot detalhado ----------------
    plt.figure(figsize=(8,4))
    data = [off_4, on_4, off_6, on_6]
    labels_box = ["4q OFF", "4q ON", "6q OFF", "6q ON"]
    plt.boxplot(data, labels=labels_box)
    plt.ylabel("Fidelidade")
    plt.title("Distribuição de fidelidade - 4q vs 6q")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
