"""Curvature test without leverage: P(jump) = Phi(c0 + c1 b + c2 b^2), Bitcoin and S&P 500."""
import numpy as np, pandas as pd
from common import load_prices, predicted_stress, ewma_sigma, probit, block_boot_se

def run(fname, start, K, B=300):
    rng = np.random.default_rng(7)
    px = load_prices(fname); y = px["r"].values
    px["b"], (mu, sd, P) = predicted_stress(y)
    px["jump"] = (px["r"] < -K*ewma_sigma(y)).astype(int)
    d = px[px["Date"] >= start].reset_index(drop=True); b = d["b"].values; J = d["jump"].values
    X = np.column_stack([np.ones(len(d)), b, b**2])
    w, ll = probit(X, J); w0, ll0 = probit(X[:, :2], J)
    se, wb = block_boot_se(X, J, rng, B=B)
    print(f"{fname:26s} -{K}s jumps={J.sum():3d}  b^2={w[2]:5.2f} (z {w[2]/se[2]:5.2f})  "
          f"LR={2*(ll-ll0):5.2f}  boot share b^2>0={np.mean(wb[:, 2] > 0):.3f}  "
          f"[p00={P[0,0]:.3f} p11={P[1,1]:.3f}]")

if __name__ == "__main__":
    for K in (2.5, 3, 4): run("BTC-USD_daily.csv", "2014-10-01", K)
    for K in (2.5, 3, 4): run("GSPC_daily_1990_2026.csv", "1991-01-01", K)
