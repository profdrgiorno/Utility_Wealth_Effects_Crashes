"""Table (leverage channel): probit of jump days on b, b^2, L, b^2*L, S&P 500 1997-2026.
Pre-set grid: thresholds -2.5/-3/-4 sigma x leverage measure (level, excess)."""
import numpy as np, pandas as pd
from common import load_prices, predicted_stress, ewma_sigma, probit, block_boot_se, finra_leverage

def run(K, measure, B=200, verbose=True):
    rng = np.random.default_rng(12345)
    px = load_prices("GSPC_daily_1990_2026.csv"); y = px["r"].values
    px["b"], (mu, sd, P) = predicted_stress(y)
    px["jump"] = (px["r"] < -K*ewma_sigma(y)).astype(int)
    px["L"] = finra_leverage(px, measure)
    d = px[px["Date"] >= "1997-03-01"].dropna(subset=["L"]).reset_index(drop=True)
    b, L, J = d["b"].values, d["L"].values, d["jump"].values
    X = np.column_stack([np.ones(len(d)), b, b**2, L, b**2*L])
    w, ll = probit(X, J); w0, ll0 = probit(X[:, :4], J)
    se, _ = block_boot_se(X, J, rng, B=B)
    return d, w, se, 2*(ll-ll0)

if __name__ == "__main__":
    print(f"{'leverage':8s} {'threshold':>9s} {'jumps':>6s} {'b^2 x L':>8s} {'z':>6s} {'LR':>6s}")
    for measure in ("level", "excess"):
        for K in (2.5, 3.0, 4.0):
            d, w, se, LR = run(K, measure)
            print(f"{measure:8s} {-K:>8.1f}s {d['jump'].sum():6d} {w[4]:8.2f} {w[4]/se[4]:6.2f} {LR:6.2f}")
            if measure == "level" and K == 3.0:
                d[["Date", "r", "b", "L"]].to_csv("../results/panel_sp500_level.csv", index=False)
            if measure == "excess" and K == 4.0:
                d[["Date", "r", "b", "L"]].to_csv("../results/panel_sp500_excess.csv", index=False)
