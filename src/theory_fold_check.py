"""Propositions 1-3: closed-form fold, leverage threshold q*, and the sign of dm*/dq.
Reproduces the calibration numbers and the random check reported in the paper."""
import numpy as np

def fold(q, g=2.0, v=1.0, c=-0.3, x0=1.0, A=0.15, mA=0.6, S=1.0):
    """Closed-form fold point m*, fold price P* and fold wealth W* (None if no fold)."""
    al = 1/(q*g) + (q-1)*c/q; be = (q-1)*x0/q
    inside = v*(be*(S-A*mA) - al*A)
    if be <= 0 or inside <= 0: return None
    m = (A*v - al + 2*np.sqrt(inside))/be
    P = (be*m - al - A*v)/(2*be)
    return m, P, c + x0*P

def qstar(g=2.0, c=-0.3, x0=1.0, A=0.15, mA=0.6, S=1.0):
    return 1 + (A/g)/(x0*(S-A*mA) - A*c)

def equilibria(m, q, g=2.0, v=1.0, c=-0.3, x0=1.0, A=0.15, mA=0.6, S=1.0):
    """Grid search of market-clearing prices (independent check of the closed form)."""
    a, b = 1/(q*g), (q-1)/q
    P = np.linspace(0.001, 3, 60001); T = a + b*(c + x0*P)
    Z = np.where(T > 0, T*(m-P)/v, 0) + A*(mA-P) - S
    s = np.sign(Z); return P[np.where(s[:-1] != s[1:])[0]]

if __name__ == "__main__":
    print(f"Calibration: q* = {qstar():.4f}")
    for q in (1.5, 2.5):
        m, P, W = fold(q); print(f"  q={q}: closed-form m* = {m:.3f}, P* = {P:.3f}, W* = {W:.3f}")
    print("Largest one-step price drop as m falls from 6 to 1 (grid search):")
    for q in (1.0, 1.5):
        prev = None; jump = 0; at = None
        for m in np.linspace(6, 1, 1001):
            r = equilibria(m, q); p = r.max() if len(r) else 0.0
            if prev is not None and prev - p > jump: jump, at = prev - p, m
            prev = p
        print(f"  q={q}: drop = {jump:.3f} at m = {at:.3f}")
    print("Leverage lowers q* (Corollary 1):", {c: round(qstar(c=c), 4) for c in (0.0, -0.3, -0.9)})

    rng = np.random.default_rng(0); agree = tot = 0
    for _ in range(20000):
        g = rng.uniform(0.3, 4); c = rng.uniform(-1.5, 0.5); A = rng.uniform(0.02, 0.5)
        mA = rng.uniform(0, 1); q = rng.uniform(1.05, 4)
        r1 = fold(q, g, 1, c, 1, A, mA, 1); r2 = fold(q + 1e-5, g, 1, c, 1, A, mA, 1)
        if r1 is None or r2 is None: continue
        m, P, W = r1
        if P <= 0 or 1/(q*g) + (q-1)/q*W <= 0: continue
        tot += 1; agree += np.sign(r2[0]-m) == np.sign(1/g - W)
    print(f"Proposition 3 random check: {agree} of {tot} relevant folds agree ({agree/tot:.4f})")

    rng = np.random.default_rng(1); up = down = 0
    for _ in range(40000):
        g = rng.uniform(0.3, 4); c = rng.uniform(-1.5, 0.5); A = rng.uniform(0.02, 0.5)
        mA = rng.uniform(0, 1); q = rng.uniform(1.05, 4)
        r = fold(q, g, 1, c, 1, A, mA, 1)
        if r is None: continue
        m, P, W = r
        if P <= 0 or 1/(q*g) + (q-1)/q*W <= 0: continue
        if W < 1/g: up += 1
        else: down += 1
    print(f"Fragile folds (W*<1/gamma): {up}; robust folds (W*>1/gamma): {down}")
