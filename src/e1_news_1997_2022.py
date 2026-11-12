"""News-clarity split with leverage, S&P 500 1997-2022 (uses panels from e1_leverage_curvature.py)."""
import numpy as np, pandas as pd
from common import probit, block_boot_se, case_boot_se, load_bbds

def run(panel_file):
    rng = np.random.default_rng(11)
    p = pd.read_csv(panel_file, parse_dates=["Date"])
    p = p[p["Date"] <= "2022-11-30"].merge(load_bbds(), on="Date", how="left")
    med = p["clarity"].median(); p["down"] = p["clarity"].notna().astype(int)
    p["J_low"] = (p["clarity"] < med).astype(int); p["J_high"] = (p["clarity"] >= med).astype(int)
    p["J_unk"] = (p["unknown"] >= 0.5).astype(int)
    b, L = p["b"].values, p["L"].values
    X = np.column_stack([np.ones(len(p)), b, b**2, L, b**2*L])
    print(f"{panel_file}: {p['Date'].min().date()}..{p['Date'].max().date()}, down jumps={p['down'].sum()}")
    for lab in ("J_low", "J_high", "J_unk"):
        yv = p[lab].values; w, ll = probit(X, yv); w0, ll0 = probit(X[:, :4], yv); se, _ = block_boot_se(X, yv, rng)
        if lab != "J_unk":
            print(f"  A. {lab:6s} b^2xL = {w[4]:6.2f} (z {w[4]/se[4]:5.2f})  LR = {2*(ll-ll0):.2f}")
    q = p[p["down"] == 1].reset_index(drop=True)
    Z = np.column_stack([np.ones(len(q)), q["b"], q["L"], q["b"]*q["L"]])
    w, ll = probit(Z, q["J_low"].values); se, _ = case_boot_se(Z, q["J_low"].values, rng)
    print(f"  B. P(low clarity | jump): b x L = {w[3]:6.3f} (z {w[3]/se[3]:5.2f})")

if __name__ == "__main__":
    run("../results/panel_sp500_excess.csv")
    run("../results/panel_sp500_level.csv")
