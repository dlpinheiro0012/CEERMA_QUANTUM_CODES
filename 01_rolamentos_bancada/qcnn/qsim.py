"""Simulador de statevector em torch (lote) + QCNN. Equivalencia com PennyLane testada em test_equiv.py"""
import math, torch
CD = torch.complex64

def ap1(s, U, ax):
    return torch.movedim(torch.tensordot(s, U, dims=([ax], [1])), -1, ax)

def ry(t):
    c, s = torch.cos(t/2), torch.sin(t/2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(CD)

def rz(t):
    e = torch.exp(-0.5j*t.to(CD)); z = torch.zeros_like(e)
    return torch.stack([torch.stack([e, z]), torch.stack([z, e.conj()])])

def cu(s, U, c, t):          # controle c, alvo t (indices de qubit); eixo 0 = lote
    ac = c+1; s0, s1 = s.select(ac, 0), s.select(ac, 1)
    at = (t+1) - (1 if t > c else 0)
    s1 = ap1(s1, U, at) if U is not None else torch.flip(s1, [at])
    return torch.stack([s0, s1], ac)

def angle_state(x, n):       # x: (B,n) angulos -> (B,2,...,2) produto de RY|0>
    B = x.shape[0]; s = torch.ones(B, dtype=CD)
    for q in range(n):
        v = torch.stack([torch.cos(x[:, q]/2), torch.sin(x[:, q]/2)], 1).to(CD)
        s = s.unsqueeze(-1) * v.view(B, *([1]*q), 2) if q else v
    return s.reshape(B, *([2]*n))

def amp_state(x, n):         # x: (B,2^n) -> normalizado
    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    return x.to(CD).reshape(x.shape[0], *([2]*n))

class QCNN(torch.nn.Module):
    """conv (bloco de 8 parametros compartilhado, em tijolos) + pooling (CRY compartilhado, descarta o controle)
    ate sobrar 2 qubits; leitura = probabilidades dos 2 qubits (estados 00,01,10 = 3 classes)."""
    def __init__(self, n, seed=0, init=0.5):
        super().__init__()
        g = torch.Generator().manual_seed(seed); self.n = n
        self.layers = []; k = n
        while k > 2: self.layers.append(k); k = k//2 + k % 2
        self.conv = torch.nn.Parameter(torch.randn(len(self.layers)+1, 8, generator=g)*init)
        self.pool = torch.nn.Parameter(torch.randn(len(self.layers), generator=g)*init)
    def block(self, s, p, a, b):
        s = ap1(s, ry(p[0]), a+1); s = ap1(s, ry(p[1]), b+1)
        s = ap1(s, rz(p[2]), a+1); s = ap1(s, rz(p[3]), b+1)
        s = cu(s, None, a, b)
        s = ap1(s, ry(p[4]), a+1); s = ap1(s, ry(p[5]), b+1)
        s = cu(s, None, b, a)
        s = ap1(s, ry(p[6]), a+1); s = ap1(s, ry(p[7]), b+1)
        return s
    def conv_layer(self, s, p, act):
        for par in (0, 1):
            for i in range(par, len(act)-1, 2):
                s = self.block(s, p, act[i], act[i+1])
        return s
    def forward(self, s):
        act = list(range(self.n))
        for li in range(len(self.layers)):
            s = self.conv_layer(s, self.conv[li], act)
            new = []
            for i in range(0, len(act)-1, 2):
                s = cu(s, ry(self.pool[li]), act[i], act[i+1]); new.append(act[i+1])
            if len(act) % 2: new.append(act[-1])
            act = new
        s = self.conv_layer(s, self.conv[-1], act)
        B = s.shape[0]
        p = (s.abs()**2)
        drop = [q+1 for q in range(self.n) if q not in act]
        p = p.sum(dim=drop) if drop else p
        p = p.reshape(B, 4)
        return p
    def logits(self, s):
        p = self.forward(s)[:, :3]
        return torch.log(p / p.sum(1, keepdim=True).clamp_min(1e-9) + 1e-6)
    def n_params(self): return self.conv.numel() + self.pool.numel()
