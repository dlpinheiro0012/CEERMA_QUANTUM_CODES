import os, numpy as np, pandas as pd
from scipy.signal import stft
BASE = os.path.expanduser("~/mnt/Dados da Bancada Geral")
PATHS = {"AB": os.path.join(BASE, "Bancada Alinhada e Balanceada Ponto I com a ponteira 1"),
         "DD": os.path.join(BASE, "Bancada Desalinhada e Desbalanceada Ponto I")}
W, STEP, FS = 2048, 1024, 25600
CLS = {"healthy": 0, "incipient": 1, "partial": 2}

def feats(s):
    return [s.mean(), s.std(), s.max(), s.min(), np.sqrt(np.mean(s**2)), s.var(), np.mean(np.abs(s)), np.max(np.abs(s))]

def spec8(s):
    _, _, Z = stft(s, fs=FS, nperseg=128, noverlap=64, boundary=None, padded=False)
    M = np.log1p(np.abs(Z))[:64]                    # 64 freq x T
    M = M.reshape(8, 8, -1).mean(1)                 # 8 bandas
    T = M.shape[1]
    cols = np.array_split(np.arange(T), 8)
    return np.stack([M[:, c].mean(1) for c in cols], 1)   # 8x8

for cond, p in PATHS.items():
    F, R, S, Y, G = [], [], [], [], []
    fid = 0
    for root, _, files in sorted(os.walk(p)):
        low = root.lower()
        if "total" in low or "interno" in low: continue
        if "sem_defeito" in low or "saudavel" in low: lab = "healthy"
        elif "parcial" in low and "nova" in low: lab = "incipient"
        elif "parcial" in low: lab = "partial"
        else: continue
        for f in sorted(files):
            try:
                df = pd.read_csv(os.path.join(root, f), skiprows=2)
                s = df.iloc[:, 1].values.astype(float); s = s[~np.isnan(s)]
            except Exception: continue
            if len(s) < 1000: continue
            for i in range(0, len(s) - W + 1, STEP):
                w = s[i:i+W]
                F.append(feats(w)); R.append(w); S.append(spec8(w)); Y.append(CLS[lab]); G.append(fid)
            fid += 1
    np.savez_compressed(os.path.join(os.path.dirname(__file__), f"dados_{cond}.npz"),
        feats=np.array(F), raw=np.array(R, dtype=np.float32), spec=np.array(S), y=np.array(Y), g=np.array(G))
    print(cond, "janelas", len(Y), "arquivos", fid, "classes", np.bincount(Y), flush=True)
