"""Propagation: co-jumps between the S&P 500, the Ibovespa and Bitcoin (jumps: r < -3 EWMA sigma)."""
import numpy as np, pandas as pd
from common import load_prices, ewma_sigma

def jumps(f):
    px = load_prices(f); px["jump"] = px["r"] < -3*ewma_sigma(px["r"].values); return px.set_index("Date")
g, b, c = jumps("GSPC_daily_1990_2026.csv"), jumps("BVSP_daily_1995_2026.csv"), jumps("BTC-USD_daily.csv")
common = g.index.intersection(b.index); common = common[common >= "1995-03-01"]
G, B = g.loc[common, "jump"], b.loc[common, "jump"]; Gn = G.shift(1, fill_value=False)
print(f"S&P/Ibovespa: Ibov jump rate {100*B.mean():.2f}% overall, {100*B[G].mean():.1f}% on S&P jump days; "
      f"share of Ibov jumps with S&P jump same/previous day {((G | Gn) & B).sum()/B.sum():.2f}")
cc = g.index.intersection(c.index); G2, C = g.loc[cc, "jump"], c.loc[cc, "jump"]
print(f"S&P/Bitcoin: BTC jump rate {100*C.mean():.2f}% overall, {100*C[G2].mean():.1f}% on S&P jump days; "
      f"co-jump dates {[d.date().isoformat() for d in cc[(G2 & C).values]]}")
