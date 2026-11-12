"""Ten largest Bitcoin liquidation days (2021-2026): predicted stress and returns around each event.
Event dates and sizes as reported by CoinGlass; illustrative, not a test."""
import numpy as np, pandas as pd
from common import load_prices, predicted_stress, ewma_sigma

EVENTS = [("2025-10-10", 19.16, "US tariff hike on China"), ("2021-04-17", 9.94, "AML rumour + mining halt"),
          ("2021-05-18", 9.01, "Tesla reversal + regulation"), ("2021-02-21", 4.10, "Overheated rally correction"),
          ("2021-09-06", 3.65, "El Salvador BTC law"), ("2025-09-21", 3.62, "Over-leveraged longs flushed"),
          ("2021-02-22", 3.15, "Yellen remarks"), ("2026-08-18", 2.99, "Treasury buybacks + SEC"),
          ("2021-04-22", 2.92, "Capital-gains tax plan"), ("2021-04-15", 2.77, "Turkey payment ban")]
px = load_prices("BTC-USD_daily.csv"); y = px["r"].values
px["b"], _ = predicted_stress(y); px["z"] = px["r"]/ewma_sigma(y)
px["ret30"] = 100*np.log(px["Adjusted_Close"]).diff(30)
rows = []
for d, usd, lab in EVENTS:
    t = px.index[px["Date"] == d][0]; w = px.loc[t:t+1]; worst = w.loc[w["r"].idxmin()]
    rows.append((d, usd, lab, px.b[t], px.ret30[t-1], worst.r, int(worst.z < -3)))
print(pd.DataFrame(rows, columns=["date", "liq $bn", "event", "b-", "prior 30d %", "worst ret %", "jump@3s"]).round(2).to_string(index=False))
