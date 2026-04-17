# Market Regime Detection with Network and Volatility Structure

A beginner-friendly Python project that studies **how market relationships change between calm and stressed periods**.

## What this project does

This project downloads daily prices for a basket of sector ETFs, then:

- computes daily returns
- builds rolling volatility and rolling correlation estimates
- turns correlations into network graphs
- tracks when the market becomes **more tightly connected**
- compares calm vs stressed periods
- creates plots and a short write-up around the theme **"When diversification weakens"**

## Why this project is strong

It connects:

- **complexity science**: dynamic networks, clustering, contagion structure
- **investment analytics**: sector rotation, diversification breakdown, stress monitoring
- **research workflow**: reproducible notebook, clear metrics, interpretable visuals

## Project structure

```text
market_regime_detection_project/
├── market_regime_detection.ipynb
├── requirements.txt
├── README.md
└── src/
    └── regime_analysis.py
```

## Quick start

### 1) Create a virtual environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Open the notebook

```bash
jupyter lab
```

Then open:

```text
market_regime_detection.ipynb
```

## Default universe

The notebook starts with 20 U.S. sector / industry / style ETFs:

`SPY, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLC, XLRE, IWM, QQQ, DIA, TLT, HYG, XBI, SMH, KRE`

This is a good starting point because ETFs are easier to explain than individual stocks and reduce missing-data headaches.

## Main ideas used

### Rolling returns
We compute daily percentage returns from adjusted close prices.

### Rolling volatility
We estimate realized volatility over a moving window. A common choice is 21 trading days, which is roughly one month.

### Rolling correlation matrix
For each date, we estimate how strongly assets have been moving together over the recent window.

### Network graph
We treat each asset as a **node**. If two assets have a strong enough correlation, we connect them with an **edge**.

### Regime detection
A stressed regime often shows:

- higher average volatility
- higher average pairwise correlation
- denser networks
- stronger clustering or tighter connectivity

## Suggested GitHub pitch

> This project explores how diversification weakens during market stress by modeling rolling correlations as dynamic networks. It combines volatility analysis, graph structure, and regime comparison to show when markets become unusually synchronized.

## Good next upgrades

- compare ETFs vs single stocks
- add community detection
- test different edge thresholds
- use minimum spanning trees
- add hidden Markov models for regime classification
- export an HTML dashboard

## Notes

This project uses **yfinance**, which is an open-source library for research and educational use with Yahoo Finance data access. Review the package's documentation and Yahoo-related disclaimer before using it in production or commercial settings.
