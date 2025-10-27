import argparse, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Hungarian assignment for permutation-invariant loss
from scipy.optimize import linear_sum_assignment

# -----------------------------
#  Data generation
# -----------------------------
def sample_quintic_roots(n, rmin=0.3, rmax=2.0, pair_prob=0.6, seed=1):
    rng = np.random.default_rng(seed)
    roots_all = []
    for _ in range(n):
        roots = []
        remaining = 5
        while remaining > 0:
            if remaining >= 2 and rng.random() < pair_prob:
                r = rng.uniform(rmin, rmax)
                theta = rng.uniform(0, np.pi)  # upper half
                z = r * (np.cos(theta) + 1j*np.sin(theta))
                roots += [z, np.conj(z)]
                remaining -= 2
            else:
                sign = rng.choice([-1, 1])
                r = rng.uniform(rmin, rmax) * sign
                roots.append(r + 0j)
                remaining -= 1
        rng.shuffle(roots)
        roots_all.append(np.array(roots, np.complex128))
    return roots_all

def roots_to_row(roots):
    coeffs = np.poly(roots)
    coeffs = np.real_if_close(coeffs, tol=1e-10)
    a = coeffs[1:].astype(float)  # a4..a0
    r_sorted = sorted(roots, key=lambda z: (np.real(z), np.imag(z)))
    y = []
    for z in r_sorted:
        y += [float(np.real(z)), float(np.imag(z))]
    return a, np.array(y, dtype=float)

def make_dataframe(n, seed):
    roots_list = sample_quintic_roots(n, seed=seed)
    rows = []
    for r in roots_list:
        a, y = roots_to_row(r)
        row = dict(a4=a[0], a3=a[1], a2=a[2], a1=a[3], a0=a[4])
        for i in range(5):
            row[f"r{i+1}_re"] = y[2*i]
            row[f"r{i+1}_im"] = y[2*i+1]
        rows.append(row)
    return pd.DataFrame(rows)

# -----------------------------
#  Model: ResMLP
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, d, ln=True, act="gelu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d) if ln else nn.Identity()
        self.fc1 = nn.Linear(d, d)
        self.act = nn.GELU() if act=="gelu" else nn.SiLU()
        self.ln2 = nn.LayerNorm(d) if ln else nn.Identity()
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        h = self.fc1(self.ln1(x))
        h = self.act(h)
        h = self.fc2(self.ln2(h))
        return x + h

class ResMLP(nn.Module):
    def __init__(self, d_in=5, d_hidden=128, depth=6, d_out=10, ln=True, act="gelu"):
        super().__init__()
        self.input = nn.Linear(d_in, d_hidden)
        self.blocks = nn.ModuleList([ResBlock(d_hidden, ln=ln, act=act) for _ in range(depth)])
        self.out = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        h = self.input(x)
        for b in self.blocks:
            h = b(h)
        y = self.out(h)
        return y

# -----------------------------
#  Losses
# -----------------------------
def loss_sorted_mse(pred, target):
    return torch.mean((pred - target)**2)

def loss_hungarian_mse(pred, target):
    # pred/target: [B,10] => 5 complex points each
    # build cost matrix per item, solve assignment on CPU
    B = pred.shape[0]
    pred_c = pred.view(B, 5, 2).detach().cpu().numpy()
    targ_c = target.view(B, 5, 2).detach().cpu().numpy()
    total = 0.0
    for b in range(B):
        P = pred_c[b]  # [5,2]
        T = targ_c[b]  # [5,2]
        # squared euclidean distances
        D = ((P[:,None,:]-T[None,:,:])**2).sum(axis=2)
        r_idx, c_idx = linear_sum_assignment(D)
        total += D[r_idx, c_idx].mean()
    return torch.tensor(total/B, dtype=pred.dtype, device=pred.device)

# -----------------------------
#  Durand–Kerner refinement (vectorized) — optional
# -----------------------------
def durand_kerner_step(coeffs, roots):
    """
    coeffs: [B,6] full coeffs [1, a4, a3, a2, a1, a0] (real64)
    roots : [B,5] complex64/128
    returns updated roots after one DK iteration
    """
    B = coeffs.shape[0]
    # evaluate polynomial at roots (Horner)
    z = roots
    p = torch.ones(B, 1, dtype=roots.dtype, device=roots.device) * coeffs[:,0:1]  # leading 1
    for k in range(1,6):
        p = p*z + coeffs[:,k:k+1]
    p = p.squeeze(1)  # [B,5]

    # product over (z_j - z_k)
    denom = torch.ones_like(roots)
    for k in range(5):
        diff = z - z[:,k:k+1]
        diff[:,k] = torch.ones_like(diff[:,k])  # avoid zero on diag
        denom[:,k] = torch.prod(diff, dim=1)

    return z - p/denom

def apply_refinement(coeffs, roots, steps=2):
    z = roots
    for _ in range(steps):
        z = durand_kerner_step(coeffs, z)
    return z

def pack_complex_to_vec(z):
    # z: [B,5] complex -> [B,10] (re,im)
    return torch.stack([z.real, z.imag], dim=-1).reshape(z.shape[0], 10)

# -----------------------------
#  Utilities
# -----------------------------
def to_numpy_roots(a):
    # a: [B,5] -> roots sorted by (re,im)
    out = []
    for row in a:
        coeffs = np.concatenate([[1.0], row.astype(np.float64)])
        r = np.roots(coeffs)
        r_sorted = sorted(r, key=lambda z: (np.real(z), np.imag(z)))
        v = []
        for z in r_sorted:
            v += [float(np.real(z)), float(np.imag(z))]
        out.append(v)
    return np.array(out, dtype=np.float64)

def rmse(a, b):  # numpy
    return float(np.sqrt(np.mean((a - b)**2)))

# -----------------------------
#  Train/eval
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--depths", nargs="+", type=int, default=[3,6,12,16])
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["sorted","hungarian"], default="hungarian")
    ap.add_argument("--refine", choices=["none","dk1","dk2"], default="dk2")
    ap.add_argument("--dtype", choices=["float32","float64"], default="float64")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # data
    df_tr = make_dataframe(args.n_train, seed=args.seed+1)
    df_va = make_dataframe(args.n_val,   seed=args.seed+2)
    df_te = make_dataframe(args.n_test,  seed=args.seed+3)

    Xtr = df_tr[["a4","a3","a2","a1","a0"]].values
    Ytr = df_tr[[f"r{i}_re" for i in range(1,6)] + [f"r{i}_im" for i in range(1,6)]].values
    Xva = df_va[["a4","a3","a2","a1","a0"]].values
    Yva = df_va[[f"r{i}_re" for i in range(1,6)] + [f"r{i}_im" for i in range(1,6)]].values
    Xte = df_te[["a4","a3","a2","a1","a0"]].values
    Yte = df_te[[f"r{i}_re" for i in range(1,6)] + [f"r{i}_im" for i in range(1,6)]].values

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64 if args.dtype=="float64" else torch.float32

    Xtr_t = torch.tensor(Xtr_s, dtype=dtype, device=device)
    Ytr_t = torch.tensor(Ytr,   dtype=dtype, device=device)
    Xva_t = torch.tensor(Xva_s, dtype=dtype, device=device)
    Yva_t = torch.tensor(Yva,   dtype=dtype, device=device)
    Xte_t = torch.tensor(Xte_s, dtype=dtype, device=device)
    Yte_t = torch.tensor(Yte,   dtype=dtype, device=device)

    def criterion_fn(pred, target):
        if args.loss == "sorted":
            return loss_sorted_mse(pred, target)
        else:
            return loss_hungarian_mse(pred, target)

    def maybe_refine(full_coeffs, pred_vec):
        if args.refine == "none":
            return pred_vec
        steps = 1 if args.refine=="dk1" else 2
        # unpack NN prediction to complex roots
        z = pred_vec.view(-1,5,2)
        zc = torch.complex(z[:,:,0], z[:,:,1])
        zc = apply_refinement(full_coeffs, zc, steps=steps)
        return pack_complex_to_vec(zc)

    results = {}

    for depth in args.depths:
        model = ResMLP(d_in=5, d_hidden=args.width, depth=depth, d_out=10, ln=True, act="gelu").to(device=device, dtype=dtype)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.1)

        # full coeffs for refinement: [1,a4,a3,a2,a1,a0]
        full_tr = torch.cat([torch.ones(Xtr_t.size(0),1, dtype=dtype, device=device), Xtr_t @ torch.diag(torch.tensor(scaler.scale_, dtype=dtype, device=device)) + torch.tensor(scaler.mean_, dtype=dtype, device=device)], dim=1)
        full_va = torch.cat([torch.ones(Xva_t.size(0),1, dtype=dtype, device=device), Xva_t @ torch.diag(torch.tensor(scaler.scale_, dtype=dtype, device=device)) + torch.tensor(scaler.mean_, dtype=dtype, device=device)], dim=1)

        val_curve = []
        B = args.batch
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(Xtr_t.size(0), device=device)
            idx = perm[:min(B, Xtr_t.size(0))]
            xb, yb = Xtr_t[idx], Ytr_t[idx]
            full_b = full_tr[idx]  # for refinement

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            pred_ref = maybe_refine(full_b, pred)
            loss = criterion_fn(pred_ref, yb)
            loss.backward()
            opt.step()
            sched.step()

            # val
            model.eval()
            with torch.no_grad():
                predv = model(Xva_t)
                predv = maybe_refine(full_va, predv)
                val_loss = float(criterion_fn(predv, Yva_t).detach().cpu().item())
            val_curve.append(val_loss)

        # Final val and test
        model.eval()
        with torch.no_grad():
            predv = model(Xva_t)
            predv = maybe_refine(full_va, predv)
            val_final = float(criterion_fn(predv, Yva_t).detach().cpu().item())

            # test RMSE vs NN; numpy.roots baseline (float64)
            predt = model(Xte_t)
            full_te = torch.cat([torch.ones(Xte_t.size(0),1, dtype=dtype, device=device), Xte_t @ torch.diag(torch.tensor(scaler.scale_, dtype=dtype, device=device)) + torch.tensor(scaler.mean_, dtype=dtype, device=device)], dim=1)
            predt = maybe_refine(full_te, predt).detach().cpu().numpy()
            rmse_nn = rmse(predt, Yte)

            roots_np = to_numpy_roots((Xte))  # use unstandardized coeffs
            rmse_np = rmse(roots_np, Yte)

        results[depth] = dict(val_curve=val_curve, val_final=val_final, rmse_nn=rmse_nn, rmse_np=rmse_np)
        print(f"[depth={depth}]  Val={val_final:.4f} | RMSE(NN)={rmse_nn:.4f} | RMSE(np.roots)={rmse_np:.4f}")

    # plots
    depths = list(results.keys())
    vals = [results[d]["val_final"] for d in depths]
    plt.figure()
    plt.bar(range(len(depths)), vals, tick_label=depths)
    plt.title(f"Val loss vs depth (width={args.width}, loss={args.loss}, refine={args.refine})")
    plt.xlabel("hidden (Res) layers"); plt.ylabel("val loss"); plt.show()

    plt.figure()
    for d in depths:
        plt.plot(results[d]["val_curve"], label=f"{d} hidden")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("val loss")
    plt.title("Validation loss curves"); plt.show()

    print("Best depth:", depths[int(np.argmin(vals))], " | results:", results)

if __name__ == "__main__":
    main()
