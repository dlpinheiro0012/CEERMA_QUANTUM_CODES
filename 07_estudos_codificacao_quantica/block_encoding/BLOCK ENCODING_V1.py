#%%
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) MNIST -> binariza -> downsample (16x16)
# =========================
def load_mnist_one_sample(index=0, train=True):
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if train:
        return x_train[index], int(y_train[index])
    return x_test[index], int(y_test[index])

def binarize(img_28x28, threshold=128):
    img = img_28x28.astype(np.float32)
    return (img >= threshold).astype(np.float32)  # 0/1

def downsample_to_pow2(img, target=16):
    if target not in (8, 16):
        raise ValueError("Use target=8 ou target=16.")
    H, W = img.shape
    ys = np.linspace(0, H, target + 1, dtype=int)
    xs = np.linspace(0, W, target + 1, dtype=int)
    out = np.zeros((target, target), dtype=np.float32)
    for i in range(target):
        for j in range(target):
            block = img[ys[i]:ys[i+1], xs[j]:xs[j+1]]
            out[i, j] = block.mean() if block.size else 0.0
    return out

# =========================
# 2) Hermitização + diagonalização
# =========================
def hermitize(A):
    return 0.5 * (A + A.T)

def eigh_decompose(Ah):
    # Ah precisa ser simétrica (Hermitiana real)
    w, V = np.linalg.eigh(Ah)   # Ah = V diag(w) V^T
    return w, V

def reconstruct_from_eigh(w, V):
    return V @ np.diag(w) @ V.T

# =========================
# 3) Transformações no espectro (exemplos simples)
# =========================
def spectral_transform(w, mode="threshold", tau=0.05, power=2, keep_top_k=None):
    """
    mode:
      - "identity": não muda nada (sanity check)
      - "threshold": zera autovalores pequenos |λ| < tau  (tipo denoise)
      - "power": aplica f(λ)=sign(λ)*|λ|^power (realça diferença de magnitudes)
      - "topk": mantém só os K maiores em |λ|  (aproximação low-rank)
    """
    if mode == "identity":
        return w.copy()

    if mode == "threshold":
        return np.where(np.abs(w) >= tau, w, 0.0)

    if mode == "power":
        return np.sign(w) * (np.abs(w) ** power)

    if mode == "topk":
        if keep_top_k is None:
            raise ValueError("Forneça keep_top_k no modo topk.")
        idx = np.argsort(np.abs(w))[::-1]
        keep = set(idx[:keep_top_k])
        w2 = np.zeros_like(w)
        for i in range(len(w)):
            if i in keep:
                w2[i] = w[i]
        return w2

    raise ValueError("mode inválido.")

# =========================
# 4) Rodar pipeline completo e plotar
# =========================
def run_mnist_diag_demo(index=7, threshold=128, target=16, train=True,
                        mode="threshold", tau=0.05, power=2, keep_top_k=20):
    img28, label = load_mnist_one_sample(index=index, train=train)
    bw28 = binarize(img28, threshold=threshold)
    A = downsample_to_pow2(bw28, target=target).astype(np.float64)

    Ah = hermitize(A)
    w, V = eigh_decompose(Ah)

    # Reconstrução "perfeita" (sanity check)
    Ah_rec = reconstruct_from_eigh(w, V)
    err_rec = np.linalg.norm(Ah - Ah_rec)

    # Transformação espectral + reconstrução
    w2 = spectral_transform(w, mode=mode, tau=tau, power=power, keep_top_k=keep_top_k)
    A2 = reconstruct_from_eigh(w2, V)

    # Métricas
    err_change = np.linalg.norm(Ah - A2)
    energy_original = np.linalg.norm(w)
    energy_kept = np.linalg.norm(w2)

    print("=== INFO ===")
    print(f"Label MNIST: {label}")
    print(f"Tamanho: {target}x{target}")
    print(f"Sanity (||Ah - V diag(w) V^T||_F): {err_rec:.3e}")
    print(f"Modo transformação: {mode}")
    if mode == "threshold":
        print(f"tau = {tau}")
    if mode == "power":
        print(f"power = {power}")
    if mode == "topk":
        print(f"keep_top_k = {keep_top_k}")
    print(f"||Ah - A2||_F (mudança após transformação): {err_change:.3e}")
    print(f"||w|| (energia espectral): {energy_original:.3e}")
    print(f"||w2|| (energia após transformação): {energy_kept:.3e}")

    # Plots
    plt.figure(); plt.title(f"MNIST original 28x28 (label={label})")
    plt.imshow(img28, cmap="gray"); plt.axis("off")

    plt.figure(); plt.title(f"A (PB + downsample {target}x{target})")
    plt.imshow(A, cmap="gray"); plt.axis("off")

    plt.figure(); plt.title("Ah = (A + A^T)/2 (Hermitizada)")
    plt.imshow(Ah, cmap="gray"); plt.axis("off")

    plt.figure(); plt.title("Espectro (autovalores) de Ah")
    plt.plot(w, marker="o"); plt.grid(True)

    plt.figure(); plt.title("A2 = V f(Λ) V^T (após transformação espectral)")
    plt.imshow(A2, cmap="gray"); plt.axis("off")

    plt.figure(); plt.title("|A2 - Ah| (diferença)")
    plt.imshow(np.abs(A2 - Ah), cmap="gray"); plt.axis("off")

    plt.show()

    return {
        "label": label,
        "A": A,
        "Ah": Ah,
        "eigvals": w,
        "eigvecs": V,
        "Ah_rec": Ah_rec,
        "A2": A2,
        "w2": w2
    }

#%% =========================
# Executar exemplo
# =========================
out = run_mnist_diag_demo(
    index=7,
    threshold=128,
    target=16,
    train=True,
    mode="threshold",   # troque para "identity", "topk", "power"
    tau=0.05,
    keep_top_k=25
)
