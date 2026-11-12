"""Proposition 6: Gamma stationary law of the Feller fear state q-1 and the
probability that the market is capable of a fold, Pr(q > q*)."""
import math, numpy as np

def gamma_tail(a, x):
    """Upper regularised incomplete gamma Q(a, x)."""
    term = 1/a; tot = term; j = 1
    while term > 1e-14*tot:
        term *= x/(a+j); tot += term; j += 1
    return 1 - math.exp(-x + a*math.log(x) - math.lgamma(a))*tot

if __name__ == "__main__":
    k, th, s = 2.0, 0.5, 0.8           # kappa_q, q0-1, sigma_q
    dt = 1/250; n = 400000; rng = np.random.default_rng(3); Y = th; out = np.empty(n)
    for i in range(n):
        Y = Y + k*(th-Y)*dt + s*math.sqrt(max(Y, 0))*math.sqrt(dt)*rng.standard_normal()
        Y = max(Y, 0); out[i] = Y
    out = out[20000:]
    shape = 2*k*th/s**2; rate = 2*k/s**2; qs = 1.6
    print(f"Gamma shape a_q = {shape:.3f} (Feller condition holds: {shape >= 1})")
    print(f"Pr(q > {qs}): simulated {np.mean(out > qs-1):.4f}  vs  Gamma tail {gamma_tail(shape, rate*(qs-1)):.4f}")
