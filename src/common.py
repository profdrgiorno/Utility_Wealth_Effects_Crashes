"""Shared tools for 'Wealth effects, perceived stress and crashes without news'.

Contents
- load_prices: reads Yahoo/yfinance CSVs (simple or multi-header format)
- hmm_fit / predicted_stress: two-state Gaussian HMM (Baum-Welch) and the
  forward-filter predicted probability of the high-volatility state, b-_t,
  which uses information up to t-1 only
- ewma_sigma: lagged EWMA volatility (lambda = 0.94)
- probit, block_boot_se, case_boot_se: probit by maximum likelihood and
  bootstrap standard errors
- finra_leverage: FINRA margin debt relative to the index level, lagged
  two months (publication delay)
- load_bbds: Baker-Bloom-Davis-Sammon WSJ jump codings (market dates)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

DATA = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------- prices
def load_prices(filename, drop_zero_returns=False):
    """Return DataFrame with Date, Adjusted_Close, r (100 x log return)."""
    path = DATA / filename
    head = pd.read_csv(path, nrows=2, header=None)
    if str(head.iloc[1, 0]).startswith("Ticker"):          # yfinance multi-header
        df = pd.read_csv(path, skiprows=[1, 2]).rename(
            columns={"Price": "Date", "Adj Close": "Adjusted_Close"})
    else:
        df = pd.read_csv(path)
    df = df[["Date", "Adjusted_Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["r"] = 100 * np.log(df["Adjusted_Close"]).diff()
    df = df.dropna(subset=["r"])
    if drop_zero_returns:                                  # stale early prices
        df = df[df["r"] != 0]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- HMM
def hmm_fit(y, iters=300):
    """Two-state Gaussian HMM by Baum-Welch. State 1 = stressed (high vol).
    Returns means, sds, transition matrix P and filtered probabilities."""
    mu = np.array([np.mean(y) + 0.02, np.mean(y) - 0.05])
    sd = np.array([0.7 * np.std(y), 1.8 * np.std(y)])
    P = np.array([[0.99, 0.01], [0.03, 0.97]])
    pi0 = np.array([0.8, 0.2])
    T = len(y)
    for _ in range(iters):
        dens = norm.pdf(y[:, None], mu, sd) + 1e-300
        a = np.zeros((T, 2)); c = np.zeros(T)
        a[0] = pi0 * dens[0]; c[0] = a[0].sum(); a[0] /= c[0]
        for t in range(1, T):
            a[t] = (a[t - 1] @ P) * dens[t]; c[t] = a[t].sum(); a[t] /= c[t]
        b = np.ones((T, 2))
        for t in range(T - 2, -1, -1):
            b[t] = P @ (dens[t + 1] * b[t + 1]); b[t] /= c[t + 1]
        g = a * b; g /= g.sum(1, keepdims=True)
        xi = np.zeros((2, 2))
        for t in range(T - 1):
            m = np.outer(a[t], dens[t + 1] * b[t + 1]) * P; xi += m / m.sum()
        P = xi / xi.sum(1, keepdims=True); pi0 = g[0]
        mu = (g * y[:, None]).sum(0) / g.sum(0)
        sd = np.sqrt((g * (y[:, None] - mu) ** 2).sum(0) / g.sum(0))
    if sd[0] > sd[1]:
        mu, sd, P, a = mu[::-1], sd[::-1], P[::-1, ::-1], a[:, ::-1]
    return mu, sd, P, a


def predicted_stress(y, iters=300):
    """b-_t = Pr(stressed state at t | returns up to t-1)."""
    mu, sd, P, filt = hmm_fit(y, iters)
    pred = np.vstack([[0.5, 0.5], filt[:-1] @ P])
    return pred[:, 1], (mu, sd, P)


def ewma_sigma(y, lam=0.94):
    """sigma_t uses returns up to t-1 only."""
    v = np.var(y[:250]); s = np.empty(len(y))
    for t in range(len(y)):
        s[t] = np.sqrt(v); v = lam * v + (1 - lam) * y[t] ** 2
    return s


# ---------------------------------------------------------------- probit
def probit(X, y):
    nll = lambda w: -np.sum(y * norm.logcdf(X @ w) + (1 - y) * norm.logcdf(-(X @ w)))
    w0 = np.zeros(X.shape[1]); w0[0] = norm.ppf(min(max(y.mean(), 1e-3), 0.999))
    r = minimize(nll, w0, method="BFGS")
    return r.x, -r.fun


def block_boot_se(X, y, rng, B=300, blk=63):
    n = len(y); out = []
    for _ in range(B):
        idx = np.concatenate([np.arange(a, a + blk)
                              for a in rng.integers(0, n - blk, int(np.ceil(n / blk)))])[:n]
        out.append(probit(X[idx], y[idx])[0])
    return np.array(out).std(0), np.array(out)


def case_boot_se(X, y, rng, B=300):
    n = len(y); out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        out.append(probit(X[idx], y[idx])[0])
    return np.array(out).std(0), np.array(out)


# ---------------------------------------------------------------- leverage
def finra_leverage(prices, measure="level", filename="margin-statistics.xlsx"):
    """Monthly leverage, standardised, lagged two months, mapped to days.
    measure='level' : log(debit balances / index), linearly detrended
    measure='excess': 12-month growth of debit balances minus index growth"""
    f = pd.read_excel(DATA / filename)
    f.columns = ["ym", "debit", "fcc", "fcm"]
    f["ym"] = pd.PeriodIndex(f["ym"], freq="M")
    f = f.sort_values("ym")
    px = prices.copy(); px["ym"] = px["Date"].dt.to_period("M")
    me = px.groupby("ym")["Adjusted_Close"].last()
    f = f.set_index("ym").join(me.rename("idx"), how="inner")
    if measure == "level":
        f["x"] = np.log(f["debit"] / f["idx"])
        tt = np.arange(len(f)); f["x"] = f["x"] - np.polyval(np.polyfit(tt, f["x"], 1), tt)
    else:
        f["x"] = np.log(f["debit"]).diff(12) - np.log(f["idx"]).diff(12)
        f = f.dropna(subset=["x"])
    f["L"] = (f["x"] - f["x"].mean()) / f["x"].std()
    lagged = f["L"].copy(); lagged.index = lagged.index + 2
    return px["ym"].map(lagged).values


# ---------------------------------------------------------------- news
def load_bbds(filename="WSJ_Stock_Jumps_1900-present.xlsx"):
    """Downward US jumps with clarity index, dated by market day."""
    j = pd.read_excel(DATA / filename, sheet_name="jumps by day (wsj)", header=0)
    j = j.loc[:, ~j.columns.duplicated()]
    j["Date"] = pd.to_datetime(dict(year=j["year"], month=j["month"], day=j["day"]))
    return j[j["return"] < 0][["Date", "clarity", "unknown"]]
