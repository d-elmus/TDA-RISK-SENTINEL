"""
data_pipeline.py — ETL & données pour TDA-Risk-Sentinel.

Ce module gère la récupération et le prétraitement des données de marché.
En cas d'indisponibilité de yfinance (réseau absent, rate-limit, etc.),
des données simulées réalistes sont générées pour la démonstration.

Les actifs couverts représentent différentes classes :
  - SPY  : Actions US (S&P 500)
  - BTC-USD : Cryptomonnaies (Bitcoin)
  - GLD  : Métal précieux (Or)
  - GSG  : Matières premières (Goldman Sachs Commodity)
  - VNQ  : Immobilier (REITs US)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta


# Tickers par défaut du projet
DEFAULT_TICKERS = ["SPY", "BTC-USD", "GLD", "GSG", "VNQ"]

# Périodes de crises historiques pour annotation
CRISIS_PERIODS = {
    "COVID Crash (2020)": ("2020-02-20", "2020-03-23"),
    "Crypto Winter (2022)": ("2022-01-01", "2022-06-30"),
    "SVB Crisis (2023)": ("2023-03-08", "2023-03-17"),
    "Inflation Shock (2022)": ("2022-06-01", "2022-10-15"),
}


def fetch_market_data(
    tickers: list = DEFAULT_TICKERS,
    start: str = "2019-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Récupère les prix de clôture ajustés depuis Yahoo Finance.

    Utilise yfinance avec gestion des erreurs. Si la récupération échoue
    (pas de connexion, ticker invalide, etc.), bascule automatiquement
    sur des données simulées réalistes.

    Args:
        tickers: Liste de tickers Yahoo Finance.
        start: Date de début au format 'YYYY-MM-DD'.
        end: Date de fin (défaut = aujourd'hui).

    Returns:
        DataFrame (T × N) de prix, index=DatetimeIndex, colonnes=tickers.
        Aucune colonne entièrement NaN n'est retournée.

    Raises:
        ValueError: Si aucun ticker valide n'est disponible et que la
                    simulation échoue.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    prices = None

    try:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )

        # Extraire colonne 'Close' ou le niveau 1 si multi-index
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]] if "Close" in raw.columns else raw

        # Supprimer colonnes entièrement vides
        prices = prices.dropna(axis=1, how="all")

        # Valider qu'on a des données
        if prices.empty or len(prices) < 100:
            raise ValueError("Données insuffisantes depuis yfinance")

        # Compléter les colonnes manquantes avec la simulation
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            sim = _simulate_prices(missing, prices.index)
            prices = pd.concat([prices, sim], axis=1)

        # Interpoler les NaN internes (jours fériés, etc.)
        prices = prices.interpolate(method="time").dropna()

    except Exception as e:
        warnings.warn(
            f"yfinance indisponible ({e}). Utilisation de données simulées.",
            UserWarning,
            stacklevel=2,
        )
        prices = _simulate_all(tickers, start, end)

    return prices


def _simulate_prices(
    tickers: list,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Génère des séries de prix simulées réalistes pour des tickers donnés.

    Utilise un modèle GBM (Geometric Brownian Motion) avec des paramètres
    calibrés sur les volatilités historiques typiques de chaque classe d'actif,
    et injecte des régimes de stress pour reproduire des dynamiques de crise.

    Args:
        tickers: Tickers à simuler.
        index: Index DatetimeIndex cible.

    Returns:
        DataFrame de prix simulés, même index que `index`.
    """
    # Paramètres GBM par classe d'actif (mu annuel, sigma annuel)
    params: Dict[str, Tuple[float, float]] = {
        "SPY":     (0.10, 0.16),
        "BTC-USD": (0.50, 0.80),
        "GLD":     (0.05, 0.12),
        "GSG":     (0.03, 0.22),
        "VNQ":     (0.08, 0.18),
    }
    default_params = (0.07, 0.20)

    T = len(index)
    dt = 1 / 252  # Pas journalier

    frames = {}
    np.random.seed(42)

    for ticker in tickers:
        mu, sigma = params.get(ticker, default_params)
        prices_arr = np.zeros(T)
        prices_arr[0] = 100.0

        # Injecter des régimes de stress (volatilité × 3 pendant ~20 jours)
        stress_mask = np.zeros(T, dtype=bool)
        n_crises = max(1, T // 300)
        for _ in range(n_crises):
            crisis_start = np.random.randint(100, T - 50)
            crisis_len = np.random.randint(15, 40)
            stress_mask[crisis_start:crisis_start + crisis_len] = True

        for t in range(1, T):
            vol = sigma * 3 if stress_mask[t] else sigma
            drift = (mu - 0.5 * vol**2) * dt
            shock = vol * np.sqrt(dt) * np.random.randn()
            prices_arr[t] = prices_arr[t - 1] * np.exp(drift + shock)

        frames[ticker] = prices_arr

    return pd.DataFrame(frames, index=index)


def _simulate_all(
    tickers: list,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Génère des données simulées sur une plage de dates complète.

    Crée un index de jours de trading (lundi-vendredi) entre start et end,
    puis appelle _simulate_prices.

    Args:
        tickers: Tickers à simuler.
        start: Date de début 'YYYY-MM-DD'.
        end: Date de fin 'YYYY-MM-DD'.

    Returns:
        DataFrame complet de prix simulés.
    """
    index = pd.bdate_range(start=start, end=end)
    return _simulate_prices(tickers, index)


def get_data_summary(prices: pd.DataFrame) -> Dict:
    """Calcule des statistiques descriptives sur les prix récupérés.

    Args:
        prices: DataFrame de prix.

    Returns:
        Dict avec clés : n_days, n_assets, start, end, missing_pct.
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    missing_pct = prices.isna().mean().mean() * 100

    return {
        "n_days": len(prices),
        "n_assets": len(prices.columns),
        "start": prices.index[0].strftime("%Y-%m-%d"),
        "end": prices.index[-1].strftime("%Y-%m-%d"),
        "missing_pct": round(missing_pct, 2),
        "tickers": list(prices.columns),
        "annual_vol": (returns.std() * np.sqrt(252)).round(4).to_dict(),
    }
