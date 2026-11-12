# Input data

Data files are **not** included in this repository. Their sources' terms of use generally do not allow redistribution. Download them as described below and save them in this folder with exactly these names.

| File name | Content | Source |
|---|---|---|
| `GSPC_daily_1928.csv` | S&P 500 daily prices, 1928–present | Yahoo Finance, ticker `^GSPC` |
| `GSPC_daily_1990_2026.csv` | S&P 500 daily prices, 1990–present | Yahoo Finance, ticker `^GSPC` |
| `BVSP_daily_1995_2026.csv` | Ibovespa daily prices, 1995–present | Yahoo Finance, ticker `^BVSP` |
| `BTC-USD_daily.csv` | Bitcoin daily prices, 2014–present | Yahoo Finance, ticker `BTC-USD` |
| `margin-statistics.xlsx` | Monthly margin debt, 1997–present | FINRA, https://www.finra.org/investors/margin-statistics ("Download the Data") |
| `WSJ_Stock_Jumps_1900-present.xlsx` | US daily jumps with cause and clarity codings | Baker, Bloom, Davis and Sammon, https://www.stockmarketjumps.com |

## Price file format

Either format works with `src/common.py`:

1. **Simple:** columns `Date,Adjusted_Close` (other columns ignored).
2. **yfinance export:** the default output of

```python
import yfinance as yf
yf.download("^GSPC", start="1928-01-01", interval="1d", auto_adjust=False).to_csv("GSPC_daily_1928.csv")
yf.download("^GSPC", start="1990-01-01", interval="1d", auto_adjust=False).to_csv("GSPC_daily_1990_2026.csv")
yf.download("^BVSP", start="1995-01-01", interval="1d", auto_adjust=False).to_csv("BVSP_daily_1995_2026.csv")
yf.download("BTC-USD", start="2014-01-01", interval="1d", auto_adjust=False).to_csv("BTC-USD_daily.csv")
```

The results in the paper were computed with files downloaded in November 2026.
