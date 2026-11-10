#%% ============================================================
# V2 — MNIST -> PB -> 16x16 -> Ah (hermitizada) -> "f(Ah)" via POLINÔMIO
# (sem diagonalizar explicitamente)
#
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1) Carregar MNIST (28x28, grayscale 0..255)
# ---------------------------
def load_mnist_one_sample(index=0, train=True):
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if train:
        return x_train[index], int(y_train[index])
    return x_test[index], int(y_test[index])

# ---------------------------
# 2) Binarizar (preto e branco) por limiar
# ---------------------------
def binarize(img_28x28, threshold=128):
    img = img_28x28.astype(np.float32)
    return (img >= threshold).astype(np.float32)  # 0/1

# ---------------------------
# 3) Downsample para potência de 2 (16x16 recomendado)
#    (média por blocos; tons de cinza aparecem por mistura nos blocos)
# ---------------------------
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
# 4) Hermitização (para ter matriz simétrica real)
# ---------------------------
def hermitize(A):
    return 0.5 * (A + A.T)

# ---------------------------
# 5) Construir A2 = p(Ah) sem diagonalização explícita
#    p(A) = c0 I + c1 A + c2 A^2 + ... + cd A^d
#
# ---------------------------
def poly_on_matrix(Ah, coeffs):
    """
    coeffs: lista [c0, c1, c2, ..., cd]
    retorna: A2 = sum_k c_k * (Ah^k), com Ah^0 = I
    """
    n = Ah.shape[0]
    A2 = coeffs[0] * np.eye(n, dtype=np.float64)

    Ak = np.eye(n, dtype=np.float64)  # Ah^0
    for k in range(1, len(coeffs)):
        Ak = Ak @ Ah                   # Ah^k
        A2 += coeffs[k] * Ak

    return A2

# ---------------------------
# 6) Ferramentas de diagnóstico
# ---------------------------
def fro_norm(M):
    return np.linalg.norm(M, ord="fro")

def is_symmetric(M, tol=1e-12):
    return np.max(np.abs(M - M.T)) < tol

# ---------------------------
# 7) Demo V2 completo + plots
# ---------------------------
def run_v2_demo(
    index=7,
    train=True,
    threshold=128,
    target=16,
    coeffs=None,
    show=True
):
    if coeffs is None:
        # Polinômio:
        # A2 = 0.0*I + 1.0*Ah + 0.5*Ah^2
        coeffs = [0.0, 1.0, 0.5]

    img28, label = load_mnist_one_sample(index=index, train=train)
    bw28 = binarize(img28, threshold=threshold)
    A = downsample_to_pow2(bw28, target=target).astype(np.float64)

    Ah = hermitize(A)

    # V2: transforma via polinômio
    A2 = poly_on_matrix(Ah, coeffs)

    # Métricas simples (mudança introduzida pelo polinômio)
    delta = A2 - Ah
    err_change = fro_norm(delta)

    # Diagnósticos
    print("=== V2 RESULTADOS ===")
    print(f"Label MNIST: {label}")
    print(f"Tamanho: {target}x{target}")
    print(f"coeffs (p(A)=c0 I + c1 A + ...): {coeffs}")
    print(f"Ah simétrica? {is_symmetric(Ah)} | A2 simétrica? {is_symmetric(A2)}")
    print(f"||A2 - Ah||_F (mudança vs Ah): {err_change:.3e}")
    print(f"||Ah||_F: {fro_norm(Ah):.3e} | ||A2||_F: {fro_norm(A2):.3e}")

    if show:
        plt.figure(); plt.title(f"MNIST original 28x28 (label={label})")
        plt.imshow(img28, cmap="gray"); plt.axis("off")

        plt.figure(); plt.title(f"A (PB + downsample {target}x{target})")
        plt.imshow(A, cmap="gray"); plt.axis("off")

        plt.figure(); plt.title("Ah = (A + A^T)/2 (Hermitizada)")
        plt.imshow(Ah, cmap="gray"); plt.axis("off")

        plt.figure(); plt.title(f"A2 = p(Ah) com coeffs={coeffs}")
        plt.imshow(A2, cmap="gray"); plt.axis("off")

        plt.figure(); plt.title("|A2 - Ah| (diferença)")
        plt.imshow(np.abs(delta), cmap="gray"); plt.axis("off")

        plt.show()

    return {
        "label": label,
        "img28": img28,
        "A": A,
        "Ah": Ah,
        "A2": A2,
        "delta": delta,
        "coeffs": coeffs
    }

#---------------------------
# 8) Rode o V2
# ---------------------------
out = run_v2_demo(
    index=7,
    train=True,
    threshold=128,
    target=16,
    coeffs=[0.0, 1.0, 0.5],
    show=True
)

# ============================================================
# ============================================================
