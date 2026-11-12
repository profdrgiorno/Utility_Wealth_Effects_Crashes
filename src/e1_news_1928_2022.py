"""MAIN RESULT (Figure 1, Table 1): crash news clarity and predicted stress, S&P 500 1928-2022."""
import numpy as np, pandas as pd
from common import load_prices, predicted_stress, ewma_sigma, probit, block_boot_se, case_boot_se, load_bbds

rng = np.random.default_rng(21)
px = load_prices("GSPC_daily_1928.csv", drop_zero_returns=True); y = px["r"].values
px["b"], (mu, sd, P) = predicted_stress(y, iters=150)
p = px[px["Date"] <= "2022-11-30"].merge(load_bbds()[["Date", "clarity"]], on="Date", how="left")
med = p["clarity"].median(); p["down"] = p["clarity"].notna().astype(int)
p["J_low"] = (p["clarity"] < med).astype(int); p["J_high"] = (p["clarity"] >= med).astype(int)
print(f"Sample {p['Date'].min().date()}..{p['Date'].max().date()}: days={len(p)}, down jumps={p['down'].sum()}, clarity median={med:.2f}")
print(f"HMM: calm sd={sd[0]:.2f}, stress sd={sd[1]:.2f}, p00={P[0,0]:.4f}, p11={P[1,1]:.4f}")

b = p["b"].values; X = np.column_stack([np.ones(len(p)), b, b**2])
print("\nA. Daily probit on b, b^2")
for lab in ("J_low", "J_high"):
    yv = p[lab].values; w, ll = probit(X, yv); w0, ll0 = probit(X[:, :2], yv); se, _ = block_boot_se(X, yv, rng)
    print(f"  {lab:6s} b={w[1]:5.2f} (z {w[1]/se[1]:5.2f})  b^2={w[2]:5.2f} (z {w[2]/se[2]:5.2f})  LR(b^2)={2*(ll-ll0):.2f}")

q = p[p["down"] == 1].reset_index(drop=True); yv = q["J_low"].values
Z = np.column_stack([np.ones(len(q)), q["b"]]); w, ll = probit(Z, yv); se, _ = case_boot_se(Z, yv, rng)
print(f"\nB. P(low clarity | crash) on b: {w[1]:.3f} (z {w[1]/se[1]:.2f})")
q["bin"] = pd.cut(q["b"], [0, .2, .5, .8, 1], include_lowest=True)
fig = q.groupby("bin", observed=False)["J_low"].agg(["mean", "count"])
fig["mean"] = (100*fig["mean"]).round(1)
print("Figure 1 data (% low clarity, crash days):"); print(fig.to_string())
fig.to_csv("../results/figure1_clarity_by_stress.csv")
for lo, hi in (("1928", "1945"), ("1946", "1996"), ("1997", "2022")):
    s = q[(q["Date"] >= lo) & (q["Date"] <= hi + "-12-31")]
    print(f"  {lo}-{hi}: low stress {100*s[s.b <= .5].J_low.mean():.1f}% (n={int((s.b <= .5).sum())}), "
          f"high stress {100*s[s.b > .5].J_low.mean():.1f}% (n={int((s.b > .5).sum())})")

print("\nC. Table 1 robustness")
p["zsize"] = np.abs(p["r"]) / ewma_sigma(p["r"].values)
q = p[p["down"] == 1].reset_index(drop=True); yv = q["J_low"].values
era = np.column_stack([(q.Date.dt.year >= 1946) & (q.Date.dt.year <= 1996), q.Date.dt.year >= 1997]).astype(float)
for lab, Z in (("+ size", np.column_stack([np.ones(len(q)), q.b, q.zsize])),
               ("+ size + era", np.column_stack([np.ones(len(q)), q.b, q.zsize, era])),
               ("+ size + era + trend", np.column_stack([np.ones(len(q)), q.b, q.zsize, era, (q.Date.dt.year-1975)/50]))):
    w, ll = probit(Z, yv); se, _ = case_boot_se(Z, yv, rng)
    print(f"  {lab:22s} b={w[1]:.3f} (z {w[1]/se[1]:.2f})")
q["size_tercile"] = pd.qcut(q["zsize"], 3, labels=["small", "mid", "large"])
print((100*q.pivot_table(index="size_tercile", columns=q.b > 0.5, values="J_low", aggfunc="mean", observed=False)).round(1).to_string())

p["prev5"] = p["down"].rolling(5).sum().shift(1).fillna(0).gt(0).astype(float)
q = p[p["down"] == 1].reset_index(drop=True); yv = q["J_low"].values
era = np.column_stack([(q.Date.dt.year >= 1946) & (q.Date.dt.year <= 1996), q.Date.dt.year >= 1997]).astype(float)
Z = np.column_stack([np.ones(len(q)), q.b, q.zsize, era, q.prev5]); w, ll = probit(Z, yv); se, _ = case_boot_se(Z, yv, rng)
print(f"  + crash in previous 5 days: b={w[1]:.3f} (z {w[1]/se[1]:.2f}); prev5={w[-1]:.3f} (z {w[-1]/se[-1]:.2f})")
s = q[q.prev5 == 0]; m = q.prev5.values == 0
Z = np.column_stack([np.ones(len(s)), s.b, s.zsize, era[m]]); w, ll = probit(Z, s.J_low.values); se, _ = case_boot_se(Z, s.J_low.values, rng)
print(f"  first crashes only (n={len(s)}): b={w[1]:.3f} (z {w[1]/se[1]:.2f}); "
      f"low-clarity {100*s[s.b <= .5].J_low.mean():.1f}% (n={int((s.b <= .5).sum())}) vs {100*s[s.b > .5].J_low.mean():.1f}% (n={int((s.b > .5).sum())})")
