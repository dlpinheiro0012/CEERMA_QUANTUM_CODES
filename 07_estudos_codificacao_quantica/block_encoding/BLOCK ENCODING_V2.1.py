#%% ============================================================
# V2.1 — Comparação direta:
#   V1: A1 = V f(Λ) V^T  (diagonalização explícita)
#   V2: A2 = f(Ah)        (polinômio em Ah, sem diagonalizar)
#
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1) MNIST -> binariza -> downsample (16x16)
# ---------------------------
def load_mnist_one_sample(index=0, train=True):
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if train:
        return x_train[index], int(y_train[index])
    return x_test[index], int(y_test[index])

def binarize(img_28x28, threshold=128):
    img = img_28x28.astype(np.float32)
    return (img >= threshold).astype(np.float32)

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

# ---------------------------
# 2) Hermitização
# ---------------------------
def hermitize(A):
    return 0.5 * (A + A.T)

# ---------------------------
# 3) f(λ) polinomial (no espectro) e f(A) polinomial (na matriz)
# ---------------------------
def f_on_eigvals(w, coeffs):
    """
    coeffs = [c0, c1, c2, ..., cd]
    f(λ) = c0 + c1 λ + c2 λ^2 + ... + cd λ^d
    """
    w2 = np.zeros_like(w, dtype=np.float64)
    for k, ck in enumerate(coeffs):
        w2 += ck * (w ** k)
    return w2

def poly_on_matrix(Ah, coeffs):
    """
    A2 = sum_k c_k * (Ah^k), com Ah^0 = I
    """
    n = Ah.shape[0]
    A2 = coeffs[0] * np.eye(n, dtype=np.float64)
    Ak = np.eye(n, dtype=np.float64)
    for k in range(1, len(coeffs)):
        Ak = Ak @ Ah
        A2 += coeffs[k] * Ak
    return A2

# ---------------------------
# 4) Métricas auxiliares
# ---------------------------
def fro_norm(M):
    return np.linalg.norm(M, ord="fro")

def max_abs(M):
    return np.max(np.abs(M))

# ---------------------------
# 5) Demo V2.1
# ---------------------------
def run_v2_1_demo(index=7, train=True, threshold=128, target=16, coeffs=None, show=True):
    if coeffs is None:
        # f(λ) = 0 + 1*λ + 0.5*λ^2
        coeffs = [0.0, 1.0, 0.5]

    img28, label = load_mnist_one_sample(index=index, train=train)
    bw28 = binarize(img28, threshold=threshold)
    A = downsample_to_pow2(bw28, target=target).astype(np.float64)
    Ah = hermitize(A)

    # ---------- V1: diagonaliza e reconstrói ----------
    w, V = np.linalg.eigh(Ah)          # Ah = V diag(w) V^T
    w2 = f_on_eigvals(w, coeffs)       # aplica f no espectro
    A_v1 = V @ np.diag(w2) @ V.T       # reconstrói

    # ---------- V2: aplica polinômio direto ----------
    A_v2 = poly_on_matrix(Ah, coeffs)

    # ---------- Comparação ----------
    diff = A_v1 - A_v2
    print("=== V2.1 RESULTADOS ===")
    print(f"Label MNIST: {label}")
    print(f"Tamanho: {target}x{target}")
    print(f"coeffs (f(λ)=Σ c_k λ^k): {coeffs}")
    print(f"||A_v1 - A_v2||_F: {fro_norm(diff):.3e}")
    print(f"max|A_v1 - A_v2|: {max_abs(diff):.3e}")
    print(f"||A_v1||_F: {fro_norm(A_v1):.3e} | ||A_v2||_F: {fro_norm(A_v2):.3e}")

    if show:
        plt.figure(); plt.title(f"MNIST original 28x28 (label={label})")
        plt.imshow(img28, cmap="gray"); plt.axis("off")

        plt.figure(); plt.title(f"A (PB + downsample {target}x{target})")
        plt.imshow(A, cmap="gray"); plt.axis("off")

        plt.figure(); plt.title("Ah = (A + A^T)/2 (Hermitizada)")
        plt.imshow(Ah, cmap="gray"); plt.axis("off")

        plt.figure(); plt.title("V1: A_v1 = V f(Λ) V^T (via diagonalização)")
        plt.imshow(A_v1, cmap="gray"); plt.axis("off")

        plt.figure(); plt.title("V2: A_v2 = f(Ah) (polinômio em Ah)")
        plt.imshow(A_v2, cmap="gray"); plt.axis("off")

        plt.figure(); plt.title("|A_v1 - A_v2| (diferença)")
        plt.imshow(np.abs(diff), cmap="gray"); plt.axis("off")

        plt.show()

    return {
        "label": label,
        "A": A,
        "Ah": Ah,
        "eigvals": w,
        "A_v1": A_v1,
        "A_v2": A_v2,
        "diff": diff,
        "coeffs": coeffs
    }

# ---------------------------
# Execute
# ---------------------------
out = run_v2_1_demo(
    index=7,
    train=True,
    threshold=128,
    target=16,
    coeffs=[0.0, 1.0, 0.5],
    show=True
)

