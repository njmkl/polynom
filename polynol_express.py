import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ---- мини-генерация данных: корни -> коэффициенты (мон. квинтики) ----
def sample_quintic_roots(n, seed=0, rmin=0.3, rmax=2.0, pair_prob=0.6):
    rng = np.random.default_rng(seed)
    all_roots = []
    for _ in range(n):
        roots, left = [], 5
        while left > 0:
            if left >= 2 and rng.random() < pair_prob:
                r = rng.uniform(rmin, rmax)
                th = rng.uniform(0, np.pi)
                z = r*(np.cos(th)+1j*np.sin(th))
                roots += [z, np.conj(z)]; left -= 2
            else:
                sign = rng.choice([-1,1])
                r = rng.uniform(rmin, rmax)*sign
                roots += [r+0j]; left -= 1
        rng.shuffle(roots)
        all_roots.append(np.array(roots, np.complex128))
    return all_roots

def roots_to_row(roots):
    coeffs = np.poly(roots)                        # [1, a4, a3, a2, a1, a0]
    coeffs = np.real_if_close(coeffs, tol=1e-10)
    a = coeffs[1:].astype(float)                   # a4..a0
    r_sorted = sorted(roots, key=lambda z: (np.real(z), np.imag(z)))
    y = []
    for z in r_sorted:
        y += [float(np.real(z)), float(np.imag(z))]  # re, im x5
    return a, np.array(y, float)

def make_df(n, seed):
    rows=[]
    for r in sample_quintic_roots(n, seed=seed):
        a,y = roots_to_row(r)
        row = dict(a4=a[0], a3=a[1], a2=a[2], a1=a[3], a0=a[4])
        for i in range(5):
            row[f"r{i+1}_re"] = y[2*i]; row[f"r{i+1}_im"] = y[2*i+1]
        rows.append(row)
    return pd.DataFrame(rows)

# ---- маленькие датасеты (быстро) ----
df_tr = make_df(200, seed=1)
df_va = make_df(50,  seed=2)

Xtr = df_tr[["a4","a3","a2","a1","a0"]].values.astype(np.float32)
Ytr = df_tr[[f"r{i}_re" for i in range(1,6)] + [f"r{i}_im" for i in range(1,6)]].values.astype(np.float32)
Xva = df_va[["a4","a3","a2","a1","a0"]].values.astype(np.float32)
Yva = df_va[[f"r{i}_re" for i in range(1,6)] + [f"r{i}_im" for i in range(1,6)]].values.astype(np.float32)

scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr).astype(np.float32)
Xva_s = scaler.transform(Xva).astype(np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print(torch.version.cuda)

# ---- простой ResMLP (минимум, без Hungarian и DK — чтобы было мгновенно) ----
class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.act = nn.GELU()
        self.ln2 = nn.LayerNorm(d)
        self.fc2 = nn.Linear(d, d)
    def forward(self, x):
        h = self.fc1(self.ln1(x))
        h = self.act(h)
        h = self.fc2(self.ln2(h))
        return x + h

class ResMLP(nn.Module):
    def __init__(self, d_in=5, d=64, depth=3, d_out=10):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.blocks = nn.ModuleList([ResBlock(d) for _ in range(depth)])
        self.out = nn.Linear(d, d_out)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out(h)

torch.manual_seed(0)
model = ResMLP(d_in=5, d=64, depth=3, d_out=10).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

Xtr_t = torch.tensor(Xtr_s, device=device)
Ytr_t = torch.tensor(Ytr,   device=device)
Xva_t = torch.tensor(Xva_s, device=device)
Yva_t = torch.tensor(Yva,   device=device)

def mse_t(a,b): return torch.mean((a-b)**2)

print("5 эпох, batch=64")
val_curve = []
for ep in range(5):  
    model.train()
    
    idx = torch.randperm(Xtr_t.shape[0], device=device)[:64]
    xb, yb = Xtr_t[idx], Ytr_t[idx]
    opt.zero_grad(set_to_none=True)
    pred = model(xb)
    loss = mse_t(pred, yb)
    loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        val = mse_t(model(Xva_t), Yva_t).item()
    val_curve.append(val)
    print(f"epoch {ep+1}/5  |  train_mse={loss.item():.4f}  val_mse={val:.4f}")

plt.figure()
plt.plot(val_curve, marker="o")
plt.title("Validation MSE (smoke test)")
plt.xlabel("epoch"); plt.ylabel("val MSE"); plt.show()
print("done")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
