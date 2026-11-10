import argparse, time, json, numpy as np, torch, pandas as pd, math
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from qsim import *
torch.set_num_threads(1)

def split_by_file(y, g, r, n_test=2):
    rng = np.random.RandomState(42 + r); te = []
    for c in np.unique(y):
        files = np.unique(g[y == c]); te += list(rng.choice(files, n_test, replace=False))
    m = np.isin(g, te); return np.where(~m)[0], np.where(m)[0]

class Quanv(torch.nn.Module):
    """quanvolucao treinavel: patch 2x2 -> 4 qubits (angulo) -> 2 camadas (RY,RZ,CNOT em anel) -> <Z>x4, passo 2 -> 4x4x4=64 -> linear 64->3"""
    def __init__(self, seed=0, L=2):
        super().__init__(); g = torch.Generator().manual_seed(seed); self.L = L
        self.w = torch.nn.Parameter(torch.randn(L, 4, 2, generator=g)*0.5)
        self.head = torch.nn.Linear(64, 3)
    def forward(self, img):                       # img (B,8,8) em [0,1]
        B = img.shape[0]
        p = img.unfold(1, 2, 2).unfold(2, 2, 2).reshape(B*16, 4) * math.pi
        s = angle_state(p, 4)
        for l in range(self.L):
            for q in range(4):
                s = ap1(s, ry(self.w[l, q, 0]), q+1); s = ap1(s, rz(self.w[l, q, 1]), q+1)
            for q in range(4): s = cu(s, None, q, (q+1) % 4)
        pr = (s.abs()**2)
        z = torch.stack([pr.sum(dim=[a for a in range(1, 5) if a != q+1]).matmul(torch.tensor([1., -1.])) for q in range(4)], 1)
        return self.head(z.reshape(B, 64))
    def n_params(self): return self.w.numel(), sum(p.numel() for p in self.head.parameters())

def prep(variant, d, tr, te):
    if variant == "A":   # 8 atributos, MinMax ajustado so no treino
        sc = MinMaxScaler().fit(d["feats"][tr]); f = lambda i: np.clip(sc.transform(d["feats"][i]), 0, 1)
        return f(tr), f(te)
    if variant == "B":   # janela crua, tira media da janela (offset), amplitude
        f = lambda i: d["raw"][i] - d["raw"][i].mean(1, keepdims=True)
        return f(tr), f(te)
    if variant in ("C", "C2"):
        s = d["spec"]; lo, hi = s[tr].min(), s[tr].max()
        if variant == "C": f = lambda i: np.clip((s[i]-lo)/(hi-lo), 0, 1)
        else:
            mu = s[tr].mean(0); f = lambda i: (s[i]-mu).reshape(len(i), 64)
        return f(tr), f(te)

def to_state(variant, X):
    X = torch.tensor(X, dtype=torch.float32)
    if variant == "A": return angle_state(X * math.pi, 8)
    if variant == "B": return amp_state(X, 11)
    if variant == "C2": return amp_state(X, 6)

def train_q(variant, Xtr, ytr, Xte, epochs, lr, bs, seed, sub):
    torch.manual_seed(seed)
    if variant == "C": model = Quanv(seed); enc = lambda X: torch.tensor(X, dtype=torch.float32); fw = model
    else:
        model = QCNN({"A": 8, "B": 11, "C2": 6}[variant], seed); enc = lambda X: to_state(variant, X); fw = model.logits
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    yt = torch.tensor(ytr); idx = np.arange(0, len(ytr), sub)
    Etr = enc(Xtr[idx]); yt = yt[idx]; rng = np.random.RandomState(seed); hist = []
    for ep in range(epochs):
        perm = rng.permutation(len(idx)); tot = 0
        for b in range(0, len(perm), bs):
            j = perm[b:b+bs]; opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(fw(Etr[j]), yt[j]); loss.backward(); opt.step(); tot += loss.item()*len(j)
        hist.append(tot/len(perm))
    with torch.no_grad():
        out = []; Ete = enc(Xte)
        for b in range(0, len(Xte), 512): out.append(fw(Ete[b:b+512]).argmax(1))
    return torch.cat(out).numpy(), model, hist

def mlp(variant, d, tr, te, seed):
    if variant == "A": X = d["feats"]; sc = MinMaxScaler().fit(X[tr])
    elif variant == "B":
        X = d["raw"] - d["raw"].mean(1, keepdims=True); X = X/np.linalg.norm(X, axis=1, keepdims=True); sc = StandardScaler().fit(X[tr])
    else: X = d["spec"].reshape(len(d["y"]), 64); sc = MinMaxScaler().fit(X[tr])
    m = MLPClassifier((64, 32), max_iter=500, random_state=seed).fit(sc.transform(X[tr]), d["y"][tr])
    return m.predict(sc.transform(X[te])), sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond"); ap.add_argument("--variant"); ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--out", default="res.csv")
    a = ap.parse_args(); d = dict(np.load(f"dados_{a.cond}.npz")); rows = []; P = {}
    for r in range(a.reps):
        tr, te = split_by_file(d["y"], d["g"], r); t0 = time.time()
        Xtr, Xte = prep(a.variant, d, tr, te)
        pred, model, hist = train_q(a.variant, Xtr, d["y"][tr], Xte, a.epochs, a.lr, a.bs, r, a.sub)
        pm, npm = mlp("C" if a.variant == "C2" else a.variant, d, tr, te, r)
        yte = d["y"][te]; yt = d["y"][tr]
        P[f"y_{r}"] = yte; P[f"q_{r}"] = pred; P[f"mlp_{r}"] = pm; P[f"idx_{r}"] = te
        if a.variant in ("C", "C2"):
            from sklearn.linear_model import LogisticRegression
            Xs = d["spec"].reshape(len(d["y"]), 64); sc2 = MinMaxScaler().fit(Xs[tr])
            P[f"lr_{r}"] = LogisticRegression(max_iter=2000).fit(sc2.transform(Xs[tr]), yt).predict(sc2.transform(Xs[te]))
        # acuracia no proprio treino (para ver overfitting) so para o quantico
        rows.append(dict(cond=a.cond, variant=a.variant, rep=r, n_train=len(tr), n_test=len(te),
            q_acc=accuracy_score(yte, pred), q_bacc=balanced_accuracy_score(yte, pred),
            mlp_acc=accuracy_score(yte, pm), mlp_bacc=balanced_accuracy_score(yte, pm),
            q_final_loss=hist[-1], n_params_q=str(model.n_params()), n_params_mlp=npm, secs=time.time()-t0))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False); np.savez_compressed(a.out.replace('.csv', '_preds.npz'), **P)
