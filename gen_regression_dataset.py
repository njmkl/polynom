import numpy as np
import pandas as pd
from pathlib import Path

def sample_quintic_roots(rng, rmin=0.3, rmax=2.0, pair_prob=0.6):
    roots = []
    remaining = 5
    while remaining > 0:
        if remaining >= 2 and rng.random() < pair_prob:
            r = rng.uniform(rmin, rmax)
            theta = rng.uniform(0, np.pi)  # верхняя полуплоскость
            z = r * (np.cos(theta) + 1j*np.sin(theta))
            roots += [z, np.conj(z)]
            remaining -= 2
        else:
            sign = rng.choice([-1, 1])
            r = rng.uniform(rmin, rmax) * sign
            roots.append(r + 0j)
            remaining -= 1
    roots = np.array(roots, dtype=np.complex128)
    rng.shuffle(roots)
    return roots

def roots_to_coeffs_targets(roots):
    # np.poly -> [1, a4, a3, a2, a1, a0] для монческого квинтика
    coeffs = np.poly(roots)
    coeffs = np.real_if_close(coeffs, tol=1e-10)
    a = coeffs[1:].astype(float)  # a4..a0
    r_sorted = sorted(roots, key=lambda z: (np.real(z), np.imag(z)))
    y = []
    for z in r_sorted:
        y += [float(np.real(z)), float(np.imag(z))]
    return a, y  # (5,), (10,)

def make_dataset(n, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        roots = sample_quintic_roots(rng)
        a, y = roots_to_coeffs_targets(roots)
        row = {
            "a4": a[0], "a3": a[1], "a2": a[2], "a1": a[3], "a0": a[4],
        }
        for i in range(5):
            row[f"r{i+1}_re"] = y[2*i]
            row[f"r{i+1}_im"] = y[2*i+1]
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out_dir = Path("data/regression"); out_dir.mkdir(parents=True, exist_ok=True)
    
    df_train = make_dataset(1000, seed=1)
    df_val   = make_dataset(300,  seed=2)
    df_test  = make_dataset(300,  seed=3)

    df_train.to_csv(out_dir/"train.csv", index=False)
    df_val.to_csv(out_dir/"val.csv", index=False)
    df_test.to_csv(out_dir/"test.csv", index=False)
    print("Saved:", out_dir/"train.csv", out_dir/"val.csv", out_dir/"test.csv")
