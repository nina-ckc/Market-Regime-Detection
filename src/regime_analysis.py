"""
Core functions for market regime detection with network and volatility structure.

This module is intentionally written in a clear, beginner-friendly style.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import yfinance as yf


DEFAULT_TICKERS = [
    "SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    "XLC", "XLRE", "IWM", "QQQ", "DIA", "TLT", "HYG", "XBI", "SMH", "KRE"
]


def download_prices(
    tickers: list[str] | None = None,
    start: str = "2018-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance using yfinance.

    Parameters
    ----------
    tickers : list[str] | None
        Asset symbols to download. If None, uses DEFAULT_TICKERS.
    start : str
        Start date in YYYY-MM-DD format.
    end : str | None
        Optional end date.

    Returns
    -------
    pd.DataFrame
        DataFrame where rows are dates and columns are tickers.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError("No data was downloaded. Check tickers or date range.")

    if "Close" in data.columns:
        prices = data["Close"].copy()
    else:
        # Fallback for different yfinance return shapes
        prices = data.copy()

    prices = prices.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple daily returns.
    """
    returns = prices.pct_change().dropna(how="all")
    return returns


def rolling_annualized_volatility(
    returns: pd.DataFrame,
    window: int = 21,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Compute rolling annualized volatility from daily returns.
    """
    return returns.rolling(window).std() * np.sqrt(trading_days)


def average_pairwise_correlation(corr_matrix: pd.DataFrame) -> float:
    """
    Average the off-diagonal values in a correlation matrix.
    """
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    values = corr_matrix.where(mask).stack()
    return float(values.mean())


def correlation_density(corr_matrix: pd.DataFrame, threshold: float = 0.6) -> float:
    """
    Fraction of asset pairs whose absolute correlation exceeds the threshold.
    """
    n = corr_matrix.shape[0]
    if n < 2:
        return np.nan

    upper = corr_matrix.where(np.triu(np.ones((n, n), dtype=bool), k=1))
    strong_links = (upper.abs() >= threshold).stack()
    total_pairs = n * (n - 1) / 2
    return float(strong_links.sum() / total_pairs)


def build_network_from_correlation(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.6,
    use_absolute_value: bool = True,
) -> nx.Graph:
    """
    Create a NetworkX graph from a correlation matrix.

    Each asset becomes a node.
    Each sufficiently strong correlation becomes an edge.
    """
    G = nx.Graph()

    assets = list(corr_matrix.columns)
    G.add_nodes_from(assets)

    for i, asset_i in enumerate(assets):
        for j, asset_j in enumerate(assets):
            if j <= i:
                continue

            corr = corr_matrix.loc[asset_i, asset_j]
            strength = abs(corr) if use_absolute_value else corr

            if pd.notna(corr) and strength >= threshold:
                G.add_edge(asset_i, asset_j, weight=float(corr), strength=float(strength))

    return G


def compute_rolling_market_metrics(
    returns: pd.DataFrame,
    window: int = 63,
    corr_threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Compute date-by-date rolling metrics that summarize market structure.

    Metrics:
    - mean_volatility
    - avg_pairwise_corr
    - network_density
    - clustering
    - connected_components
    """
    results = []

    for end_idx in range(window, len(returns) + 1):
        window_returns = returns.iloc[end_idx - window:end_idx]
        date = window_returns.index[-1]

        corr = window_returns.corr()
        vol = window_returns.std() * np.sqrt(252)
        G = build_network_from_correlation(corr, threshold=corr_threshold)

        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
            clustering = nx.average_clustering(G)
            connected_components = nx.number_connected_components(G)
            density = nx.density(G)
        else:
            clustering = np.nan
            connected_components = np.nan
            density = 0.0

        results.append(
            {
                "date": date,
                "mean_volatility": float(vol.mean()),
                "avg_pairwise_corr": average_pairwise_correlation(corr),
                "strong_corr_share": correlation_density(corr, threshold=corr_threshold),
                "network_density": float(density),
                "clustering": float(clustering) if pd.notna(clustering) else np.nan,
                "connected_components": connected_components,
            }
        )

    metrics = pd.DataFrame(results).set_index("date")
    return metrics


def label_regimes(
    metrics: pd.DataFrame,
    stress_percentile: float = 0.8,
) -> pd.DataFrame:
    """
    Label dates as calm or stressed using a simple stress score.

    Stress score = average of normalized mean volatility and normalized avg correlation.
    """
    labeled = metrics.copy()

    z_vol = (labeled["mean_volatility"] - labeled["mean_volatility"].mean()) / labeled["mean_volatility"].std()
    z_corr = (labeled["avg_pairwise_corr"] - labeled["avg_pairwise_corr"].mean()) / labeled["avg_pairwise_corr"].std()

    labeled["stress_score"] = (z_vol + z_corr) / 2.0
    cutoff = labeled["stress_score"].quantile(stress_percentile)

    labeled["regime"] = np.where(labeled["stress_score"] >= cutoff, "stressed", "calm")
    return labeled


def get_regime_snapshot_dates(
    labeled_metrics: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Pick one representative calm date and one representative stressed date.
    """
    calm_date = labeled_metrics.loc[labeled_metrics["stress_score"].idxmin()].name
    stressed_date = labeled_metrics.loc[labeled_metrics["stress_score"].idxmax()].name
    return calm_date, stressed_date


def correlation_snapshot(
    returns: pd.DataFrame,
    end_date: pd.Timestamp,
    window: int = 63,
) -> pd.DataFrame:
    """
    Get the rolling correlation matrix ending at a chosen date.
    """
    end_loc = returns.index.get_indexer([end_date], method="nearest")[0]
    start_loc = max(0, end_loc - window + 1)
    window_returns = returns.iloc[start_loc:end_loc + 1]
    return window_returns.corr()


def plot_network(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.6,
    title: str = "Correlation Network",
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Plot a correlation network using a spring layout.
    """
    G = build_network_from_correlation(corr_matrix, threshold=threshold)

    plt.figure(figsize=figsize)

    if G.number_of_edges() == 0:
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(G, pos=pos, with_labels=True, node_size=1000, font_size=9)
        plt.title(title + " (No edges above threshold)")
        plt.axis("off")
        plt.tight_layout()
        return

    pos = nx.spring_layout(G, seed=42, weight="strength")
    edge_widths = [1 + 4 * G[u][v]["strength"] for u, v in G.edges()]
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_size=1000,
        font_size=9,
        width=edge_widths,
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()


def compare_regime_summary(labeled_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Compare average metrics between calm and stressed dates.
    """
    summary = labeled_metrics.groupby("regime")[
        ["mean_volatility", "avg_pairwise_corr", "strong_corr_share", "network_density", "clustering"]
    ].mean().round(3)

    return summary


def diversification_writeup(summary: pd.DataFrame) -> str:
    """
    Generate a short automatic write-up from the regime comparison table.
    """
    calm = summary.loc["calm"]
    stressed = summary.loc["stressed"]

    text = f"""
When diversification weakens:

In calm periods, the average market volatility is about {calm['mean_volatility']:.2f},
while the average pairwise correlation is about {calm['avg_pairwise_corr']:.2f}.
In stressed periods, volatility rises to about {stressed['mean_volatility']:.2f}
and average correlation rises to about {stressed['avg_pairwise_corr']:.2f}.

The share of strong correlation links increases from {calm['strong_corr_share']:.2f}
to {stressed['strong_corr_share']:.2f}, which means more assets begin moving together.
Network density also rises from {calm['network_density']:.2f} to {stressed['network_density']:.2f}.

This is the key diversification message: when markets are stressed, relationships between assets
become tighter and more synchronized. Assets that seemed separate during calm periods can start
behaving like one connected system.
""".strip()

    return text
