import numpy as np, torch, pennylane as qml
from qsim import *
n=8; torch.manual_seed(0)
m = QCNN(n, seed=1, init=1.0)
x = torch.rand(4, n)*math.pi
p_t = m(angle_state(x, n)).detach().numpy()
dev = qml.device("default.qubit", wires=n)
conv, pool = m.conv.detach().numpy(), m.pool.detach().numpy()
def block(p,a,b):
    qml.RY(p[0],a); qml.RY(p[1],b); qml.RZ(p[2],a); qml.RZ(p[3],b); qml.CNOT([a,b])
    qml.RY(p[4],a); qml.RY(p[5],b); qml.CNOT([b,a]); qml.RY(p[6],a); qml.RY(p[7],b)
def conv_layer(p, act):
    for par in (0,1):
        for i in range(par, len(act)-1, 2): block(p, act[i], act[i+1])
@qml.qnode(dev)
def circ(xs):
    for q in range(n): qml.RY(xs[q], q)
    act=list(range(n))
    for li in range(len(m.layers)):
        conv_layer(conv[li], act); new=[]
        for i in range(0,len(act)-1,2): qml.CRY(pool[li], [act[i],act[i+1]]); new.append(act[i+1])
        if len(act)%2: new.append(act[-1])
        act=new
    conv_layer(conv[-1], act)
    return qml.probs(wires=act)
p_p = np.array([circ(xi.numpy()) for xi in x])
print("max |diff| prob:", np.abs(p_t-p_p).max(), "| n_params:", m.n_params(), "| camadas:", m.layers)
# 11 qubits: so checa normalizacao e forma
m11=QCNN(11); s=amp_state(torch.randn(3,2048),11); p=m11(s); print("11q", p.shape, p.sum(1).detach().numpy(), m11.n_params(), m11.layers)
