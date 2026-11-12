# Utility, Wealth Effects and Crashes

Code for **"Wealth effects, perceived stress and crashes without news"** (J. Giorno, submitted to *Economics Letters*) and its longer companion working paper, *Belief-Driven Crash Risk: Wealth Effects, Filtered Regime Beliefs and Feller–Tsallis Fear*.

**Author:** J. Giorno, PhD — Graduate Program in Economics, Maringá State University (UEM), Brazil; PhDRisk Ltda. Contact: jgiorno@uem.br

## The idea in one paragraph

Levered investors with wealth effects (a HARA utility written in Tsallis *q*-exponential form) make excess demand a concave quadratic in price, so the equilibrium price can **fold**. A closed-form threshold *q\** governs when a fold exists, and leverage lowers it. Higher perceived stress moves the market towards the fold, so a smaller or less clear shock is enough to trigger a crash. The prediction is that **crashes without a clear news cause are more common when perceived stress is high**. In US data from 1928 to 2022, the share of crashes without a clear cause rises from 26% at low predicted stress to 61% at high stress.

## Repository layout

```
Utility_Wealth_Effects_Crashes/
├── README.md
├── run_all.py                      # runs everything, writes results/*.txt
├── requirements.txt
├── data/README.md                  # where to download each input file
├── src/
│   ├── common.py                   # loaders, HMM filter, probit, bootstraps, leverage, news codings
│   ├── theory_fold_check.py        # Propositions 1–3: fold, q*, sign of the fear effect
│   ├── theory_feller_gamma_check.py# Feller fear state: Gamma tail Pr(q > q*)
│   ├── e1_news_1928_2022.py        # MAIN RESULT: Figure 1 and Table 1
│   ├── e1_leverage_curvature.py    # leverage–curvature grid (not supported)
│   ├── e1_news_1997_2022.py        # news split with leverage, 1997–2022
│   ├── e1_curvature_no_leverage.py # curvature tests, Bitcoin and S&P 500
│   ├── btc_liquidation_events.py   # ten largest Bitcoin liquidation days
│   └── cojumps.py                  # propagation: S&P 500, Ibovespa, Bitcoin
├── results/                        # outputs of run_all.py
└── paper/                          # LaTeX sources (single self-contained files)
```

## Map from paper to code

| Paper item | Script | Output |
|---|---|---|
| Propositions 1–3, calibration, random sign check | `theory_fold_check.py` | `results/theory_fold_check.txt` |
| Feller fear state, Gamma tail | `theory_feller_gamma_check.py` | `results/theory_feller_gamma_check.txt` |
| Figure 1, Table 1 (main result) | `e1_news_1928_2022.py` | `results/e1_news_1928_2022.txt`, `results/figure1_clarity_by_stress.csv` |
| Leverage channel (not supported) | `e1_leverage_curvature.py`, `e1_news_1997_2022.py` | `results/e1_*.txt` |
| Curvature, Bitcoin liquidations, propagation (companion paper) | remaining scripts | `results/*.txt` |

## How to reproduce

1. Install Python 3.10+ and the requirements: `pip install -r requirements.txt`
2. Download the six input files listed in [`data/README.md`](data/README.md) into `data/`.
3. Run `python run_all.py`. A full run takes roughly 20–40 minutes, mostly the hidden-Markov-model fits.

Random seeds are fixed, so bootstrap standard errors reproduce exactly given the same data files. Market data vendors occasionally revise historical prices, which can change results slightly.

## Methods in brief

- **Perceived stress** `b-_t`: predicted probability of the high-volatility state from a two-state Gaussian hidden Markov model (Baum–Welch), computed with the forward filter, so it uses only information up to day *t−1*.
- **Crash news clarity**: newspaper codings of US daily moves larger than 2.5% by Baker, Bloom, Davis and Sammon (2021); a crash is *low-clarity* if its clarity index is below the sample median.
- **Leverage**: FINRA margin debt relative to the index level (detrended) or its 12-month excess growth, lagged two months for FINRA's publication delay.
- **Inference**: probit by maximum likelihood; 63-day block bootstrap for daily panels, case bootstrap for crash-day samples.

## What the results do and do not show

- **Supported:** crashes without clear news are more frequent when perceived stress is high (1928–2022), robust to crash size, era and trend.
- **Weaker:** after controlling for a crash in the previous five days the effect is marginal (z ≈ 1.7).
- **Not supported with these data:** the leverage channel (FINRA retail margin debt) and convexity of crash probability in perceived stress.

All designs tried are reported, including the negative ones.

## Data caveats

- B3 spreadsheets exported to Excel swap day and month when the day is 12 or less (not used in these scripts, but relevant for extensions to Brazil).
- FINRA reports free-credit balances as one combined figure before February 2010.

## Citation

> Giorno, J. (2026). Wealth effects, perceived stress and crashes without news. Working paper, Maringá State University.

## Licence

Code released under the MIT Licence (see `LICENSE`). Data are not redistributed here; please obtain them from the original sources and respect their terms of use.
