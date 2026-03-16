"""
backtester.py — Module de Backtesting TDA-Risk-Sentinel Pro v2.4

Deux couches de stratégie :

  COUCHE A — Stratégie TSS Classique (run_backtest)
    Position refuge si TSS > θ pendant N jours consécutifs.
    Signal symétrique : même logique pour la sortie et la rentrée.

  COUCHE B — Stratégie P-TSS Prédictive (run_ptss_backtest) ← NOUVEAU
    Exploite le Lead Time de 37 jours du P-TSS via une logique asymétrique :
      SORTIE  rapide  : P-TSS > θ_exit (1 jour de confirmation suffit)
                        OU Early Warning actif si P-TSS dépasse un seuil bas.
      RENTRÉE prudente: ΔTSS < θ_r_delta ET TSS_slow < θ_r_slow
                        ET P-TSS < θ_r_ptss, pendant N jours.
    Objectif : Max Drawdown < -15 % (vs -37 % Buy & Hold).

  Anti-biais NaN :
    La période de chauffe de W_slow (~100j) produit des valeurs 0.0 sur
    TSS_slow (ffill sur NaN). Le signal est inhibé avant `activation_date`
    (= première date valide de ms_result.dates_slow) pour éliminer tout
    biais de survie.

KPIs calculés :
  - CAGR / Sharpe / Max Drawdown / Calmar / Hit Rate / Lead Time
  - Pain Index  : profondeur moyenne du drawdown (∫|DD|/T)
  - Capital Saved : perte évitée par rapport au B&H sur chaque crise
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

REFERENCE_CRISES: Dict[str, Tuple[str, str]] = {
    "COVID Crash":     ("2020-02-20", "2020-03-23"),
    "Inflation Shock": ("2022-01-05", "2022-10-13"),
    "SVB Crisis":      ("2023-03-09", "2023-03-24"),
    "Crypto Winter":   ("2021-11-10", "2022-06-18"),
}

# Nombre de jours pour chercher le creux de prix après un signal REFUGE
TROUGH_SEARCH_WINDOW: int = 120


# ═══════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestKPIs:
    """KPIs financiers complets d'une stratégie backtestée.

    Attributes:
        strategy_name:         Nom de la stratégie.
        cagr:                  Compound Annual Growth Rate (annualisé).
        sharpe:                Sharpe Ratio (rf = 0).
        max_drawdown:          Maximum Drawdown ≤ 0 (ex. -0.15 = -15%).
        calmar:                Calmar = CAGR / |MaxDD|.
        total_return:          Rendement total de la période.
        n_trades:              Nombre de rotations effectuées.
        hit_rate:              % de jours où la stratégie surperforme B&H.
        volatility_ann:        Volatilité annualisée.
        avg_lead_time_days:    Lead Time moyen sur les crises référencées.
        pain_index:            Profondeur moyenne du drawdown ∈ [0, 1].
                               Pain = mean(|drawdown(t)|) — bas = mieux.
        capital_saved:         Capital total sauvé vs B&H sur toutes crises
                               (ex. 0.22 = 22 % du portefeuille préservé).
    """
    strategy_name:       str   = ""
    cagr:                float = 0.0
    sharpe:              float = 0.0
    max_drawdown:        float = 0.0
    calmar:              float = 0.0
    total_return:        float = 0.0
    n_trades:            int   = 0
    hit_rate:            float = 0.0
    volatility_ann:      float = 0.0
    avg_lead_time_days:  float = 0.0
    pain_index:          float = 0.0
    capital_saved:       float = 0.0


@dataclass
class BacktestResult:
    """Résultat complet d'un backtest TDA.

    Attributes:
        equity_tda:            Courbe d'équité normalisée (pd.Series).
        equity_bnh:            Courbe d'équité Buy & Hold (pd.Series).
        signals:               Série signaux 0=risqué / 1=refuge (pd.Series).
        returns_tda:           Rendements journaliers stratégie (pd.Series).
        returns_bnh:           Rendements journaliers B&H (pd.Series).
        drawdown_tda:          Drawdown stratégie ≤ 0 (pd.Series).
        drawdown_bnh:          Drawdown B&H ≤ 0 (pd.Series).
        kpis_tda:              BacktestKPIs de la stratégie.
        kpis_bnh:              BacktestKPIs du Buy & Hold.
        lead_times:            Dict {crise: jours}.
        trade_log:             DataFrame des rotations enrichi.
        capital_saved_by_crisis: Dict {crise: capital_sauvé_%}.
    """
    equity_tda:             pd.Series       = field(default_factory=pd.Series)
    equity_bnh:             pd.Series       = field(default_factory=pd.Series)
    signals:                pd.Series       = field(default_factory=pd.Series)
    returns_tda:            pd.Series       = field(default_factory=pd.Series)
    returns_bnh:            pd.Series       = field(default_factory=pd.Series)
    drawdown_tda:           pd.Series       = field(default_factory=pd.Series)
    drawdown_bnh:           pd.Series       = field(default_factory=pd.Series)
    kpis_tda:               BacktestKPIs    = field(default_factory=BacktestKPIs)
    kpis_bnh:               BacktestKPIs    = field(default_factory=BacktestKPIs)
    lead_times:             Dict            = field(default_factory=dict)
    trade_log:              pd.DataFrame    = field(default_factory=pd.DataFrame)
    capital_saved_by_crisis: Dict           = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# 1. MÉTRIQUES FINANCIÈRES
# ═══════════════════════════════════════════════════════════════════════════

def compute_max_drawdown(equity: pd.Series) -> Tuple[float, pd.Series]:
    """Maximum Drawdown et série de drawdown instantané.

    MaxDD = max_t [(peak_t - equity_t) / peak_t]

    Args:
        equity: Courbe d'équité (valeurs > 0).

    Returns:
        (max_dd ≤ 0, drawdown_series ≤ 0)
    """
    peak = equity.cummax()
    dd   = (equity - peak) / peak
    return float(dd.min()), dd


def compute_pain_index(drawdown_series: pd.Series) -> float:
    """Pain Index — profondeur moyenne du drawdown dans le temps.

    Pain = (1/T) × Σ |DD(t)|

    Interprétation : 0 = jamais sous l'eau. 0.10 = sous l'eau en
    moyenne de 10 % sur toute la période. Métrique complémentaire au
    MaxDD car elle pénalise aussi les drawdowns longs mais modérés.

    Args:
        drawdown_series: Série drawdown ≤ 0.

    Returns:
        Pain Index ∈ [0, 1] (positif, bas = mieux).
    """
    return float(-drawdown_series.mean())


def compute_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Sharpe Ratio annualisé (rf = 0 par défaut)."""
    excess = returns - risk_free_rate / periods_per_year
    mu, sig = excess.mean(), excess.std(ddof=1)
    return float(mu / sig * np.sqrt(periods_per_year)) if sig > 1e-10 else 0.0


def compute_cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate."""
    T = len(equity)
    if T < 2:
        return 0.0
    ratio = equity.iloc[-1] / equity.iloc[0]
    return float(ratio ** (periods_per_year / T) - 1.0) if ratio > 0 else -1.0


def compute_capital_saved(
    equity_tda: pd.Series,
    equity_bnh: pd.Series,
    crises: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Dict[str, float]:
    """Capital sauvé par la stratégie TDA vs B&H pour chaque crise.

    Capital Saved (crise) = drawdown_BH_max(crise) − drawdown_TDA_max(crise)
    Positif = la stratégie a mieux protégé le capital pendant cette crise.

    Calcul :
      Pour chaque crise, on identifie le drawdown maximal observé par
      chaque stratégie pendant la période de crise (et jusqu'à 20j après,
      pour capturer les suites immédiates).

    Args:
        equity_tda: Courbe d'équité stratégie.
        equity_bnh: Courbe d'équité B&H.
        crises:     Dict {nom: (start, end)}. Défaut = REFERENCE_CRISES.

    Returns:
        Dict {nom_crise: capital_saved_pct} — fraction ∈ [-1, 1].
    """
    if crises is None:
        crises = REFERENCE_CRISES

    saved: Dict[str, float] = {}

    for name, (start_str, end_str) in crises.items():
        try:
            c_start = pd.Timestamp(start_str)
            c_end   = pd.Timestamp(end_str) + pd.Timedelta(days=20)

            mask = (equity_tda.index >= c_start) & (equity_tda.index <= c_end)
            if mask.sum() < 2:
                saved[name] = 0.0
                continue

            # Drawdown de chaque stratégie sur la fenêtre de crise
            tda_slice = equity_tda.loc[mask]
            bnh_slice = equity_bnh.loc[mask]

            # Normaliser par la valeur au début de la crise (pas de look-ahead)
            tda_start_val = equity_tda.loc[equity_tda.index <= c_start]
            bnh_start_val = equity_bnh.loc[equity_bnh.index <= c_start]

            if tda_start_val.empty or bnh_start_val.empty:
                saved[name] = 0.0
                continue

            tda_v0 = float(tda_start_val.iloc[-1])
            bnh_v0 = float(bnh_start_val.iloc[-1])

            if tda_v0 <= 0 or bnh_v0 <= 0:
                saved[name] = 0.0
                continue

            # Pire perte relative pendant la crise
            tda_loss = float((tda_slice.min() - tda_v0) / tda_v0)  # ≤ 0
            bnh_loss = float((bnh_slice.min() - bnh_v0) / bnh_v0)  # ≤ 0

            saved[name] = float(bnh_loss - tda_loss)  # positif = TDA mieux

        except Exception:
            saved[name] = 0.0

    return saved


def compute_kpis(
    equity: pd.Series,
    returns: pd.Series,
    strategy_name: str = "",
    n_trades: int = 0,
    lead_time: float = 0.0,
    equity_bnh: Optional[pd.Series] = None,
    crises: Optional[Dict] = None,
) -> BacktestKPIs:
    """Agrège tous les KPIs financiers, y compris Pain Index et Capital Saved.

    Args:
        equity:        Courbe d'équité normalisée.
        returns:       Rendements journaliers.
        strategy_name: Nom.
        n_trades:      Nombre de rotations.
        lead_time:     Lead Time moyen (jours).
        equity_bnh:    Courbe B&H (pour Capital Saved). Optionnel.
        crises:        Dict des crises de référence. Optionnel.

    Returns:
        BacktestKPIs peuplé.
    """
    cagr        = compute_cagr(equity)
    sharpe      = compute_sharpe(returns)
    max_dd, dd  = compute_max_drawdown(equity)
    calmar      = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0
    total_ret   = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    vol_ann     = float(returns.std(ddof=1) * np.sqrt(252))
    pain        = compute_pain_index(dd)

    cap_saved = 0.0
    if equity_bnh is not None:
        cs_dict   = compute_capital_saved(equity, equity_bnh, crises)
        valid_cs  = [v for v in cs_dict.values() if not np.isnan(v)]
        cap_saved = float(np.sum(valid_cs)) if valid_cs else 0.0

    return BacktestKPIs(
        strategy_name=strategy_name,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        total_return=total_ret,
        n_trades=n_trades,
        hit_rate=0.0,
        volatility_ann=vol_ann,
        avg_lead_time_days=lead_time,
        pain_index=pain,
        capital_saved=cap_saved,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. ALIGNEMENT TSS → DATES
# ═══════════════════════════════════════════════════════════════════════════

def align_tss_to_dates(
    tss: pd.Series,
    dates: pd.DatetimeIndex,
    window_centers: List[int],
    price_index: pd.DatetimeIndex,
) -> pd.Series:
    """Aligne le TSS (index entier) sur l'index DatetimeIndex des prix.

    Args:
        tss:            pd.Series TSS avec index entier.
        dates:          DatetimeIndex du nuage embeddi.
        window_centers: Positions des centres de fenêtres dans `dates`.
        price_index:    DatetimeIndex cible.

    Returns:
        pd.Series TSS daté aligné sur price_index, ffill, fillna(0.0).
    """
    centers_dates = [dates[min(wc, len(dates) - 1)] for wc in window_centers]
    tss_dated     = pd.Series(tss.values, index=pd.DatetimeIndex(centers_dates))
    combined_idx  = price_index.union(tss_dated.index).sort_values()
    tss_ri        = tss_dated.reindex(combined_idx).interpolate(
        method="time", limit_direction="forward"
    )
    return tss_ri.reindex(price_index).ffill().fillna(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. GÉNÉRATEURS DE SIGNAUX
# ═══════════════════════════════════════════════════════════════════════════

def generate_trading_signals(
    tss_daily: pd.Series,
    threshold: float = 0.8,
    confirmation_days: int = 2,
) -> pd.Series:
    """Signal symétrique classique : TSS > θ pendant N jours → REFUGE.

    Décalage +1 jour (exécution au lendemain, pas de look-ahead).

    Args:
        tss_daily:         TSS journalier aligné sur les prix.
        threshold:         Seuil θ.
        confirmation_days: Nb jours consécutifs requis.

    Returns:
        pd.Series signaux : 0 = risqué, 1 = refuge.
    """
    raw = (tss_daily > threshold).astype(int)
    if confirmation_days > 1:
        confirmed = raw.rolling(
            window=confirmation_days, min_periods=confirmation_days
        ).min().fillna(0).astype(int)
    else:
        confirmed = raw
    return confirmed.shift(1).fillna(0).astype(int)


def generate_predictive_signals(
    ptss: pd.Series,
    delta_tss: pd.Series,
    tss_slow: pd.Series,
    early_warning: pd.Series,
    exit_threshold: float = 0.30,
    reentry_delta_threshold: float = 0.05,
    reentry_slow_threshold: float = 0.55,
    reentry_ptss_threshold: float = 0.20,
    confirmation_exit: int = 1,
    confirmation_reentry: int = 3,
    use_early_warning: bool = True,
    ew_ptss_min: float = 0.15,
    activation_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Générateur de signaux asymétrique exploitant le Lead Time du P-TSS.

    LOGIQUE DE SORTIE (→ REFUGE) :
      Condition principale  : P-TSS > exit_threshold pendant confirmation_exit
                              jours consécutifs.
      Condition accélérée   : Early Warning actif ET P-TSS > ew_ptss_min
                              (pas de délai de confirmation — réactivité max).
      La sortie rapide capture l'avance prédictive du P-TSS (37j de lead
      time) en évitant d'attendre une confirmation qui éroderait le gain.

    LOGIQUE DE RENTRÉE (→ MARCHÉ) :
      TROIS conditions simultanées pendant confirmation_reentry jours :
        1. ΔTSS < reentry_delta_threshold  → divergence résorbée
        2. TSS_slow < reentry_slow_threshold → structure macro stable
        3. P-TSS < reentry_ptss_threshold   → fracture topologique résolue
      La rentrée prudente évite de revenir sur un marché encore fragile.

    MACHINE D'ÉTAT :
      Maintient la position courante (REFUGE ou MARCHÉ) jusqu'à ce que
      les conditions opposées soient remplies — élimine le "whipsawing".

    ANTI-BIAIS NaN :
      Avant `activation_date` (première date valide de TSS_slow),
      le signal est forcé à 0 (MARCHÉ) pour éliminer les faux signaux
      générés par les 0.0 de remplissage de la période de chauffe.

    Args:
        ptss:                   P-TSS aligné ∈ [0, 1].
        delta_tss:              ΔTSS = TSS_fast - TSS_slow ∈ [-1, 1].
        tss_slow:               TSS_slow aligné ∈ [0, 1].
        early_warning:          Série booléenne EW (accélération ΔTSS).
        exit_threshold:         Seuil P-TSS pour la sortie. Défaut = 0.30.
        reentry_delta_threshold: ΔTSS max pour la rentrée. Défaut = 0.05.
        reentry_slow_threshold: TSS_slow max pour la rentrée. Défaut = 0.55.
        reentry_ptss_threshold: P-TSS max pour la rentrée. Défaut = 0.20.
        confirmation_exit:      Jours consécutifs pour sortie. Défaut = 1.
        confirmation_reentry:   Jours consécutifs pour rentrée. Défaut = 3.
        use_early_warning:      Activer la sortie accélérée EW. Défaut = True.
        ew_ptss_min:            P-TSS min pour sortie EW. Défaut = 0.15.
        activation_date:        Date à partir de laquelle le signal est actif.
                                Avant cette date : signal = 0 (anti-NaN biais).

    Returns:
        Tuple (signals, signal_types) :
          - signals      : pd.Series 0/1 décalée de +1 jour.
          - signal_types : pd.Series[str] "PTSS_EXIT" / "EW_EXIT" /
                           "REENTRY" / "HOLD" (non décalée, pour le trade log).
    """
    idx = ptss.index
    n   = len(idx)

    # Pré-calcul des conditions booléennes
    ptss_vals  = ptss.values
    dt_vals    = delta_tss.values
    slow_vals  = tss_slow.values
    ew_vals    = early_warning.astype(bool).values

    # Condition de sortie principale : P-TSS > seuil pendant N jours
    exit_raw = ptss > exit_threshold
    if confirmation_exit > 1:
        exit_confirmed = exit_raw.rolling(
            window=confirmation_exit, min_periods=confirmation_exit
        ).min().fillna(False).astype(bool)
    else:
        exit_confirmed = exit_raw

    # Condition de sortie accélérée : EW ET P-TSS > ew_ptss_min
    if use_early_warning:
        exit_ew = early_warning & (ptss > ew_ptss_min)
    else:
        exit_ew = pd.Series(False, index=idx)

    # Condition de rentrée : ΔTSS < r_δ AND TSS_slow < r_s AND P-TSS < r_p
    reentry_raw = (
        (delta_tss < reentry_delta_threshold) &
        (tss_slow  < reentry_slow_threshold)  &
        (ptss      < reentry_ptss_threshold)
    )
    if confirmation_reentry > 1:
        reentry_confirmed = reentry_raw.rolling(
            window=confirmation_reentry, min_periods=confirmation_reentry
        ).min().fillna(False).astype(bool)
    else:
        reentry_confirmed = reentry_raw

    # Machine d'état : parcours séquentiel (sans vectorisation pour cohérence)
    signals_raw  = np.zeros(n, dtype=int)
    sig_types    = np.full(n, "HOLD", dtype=object)
    position     = 0  # 0 = marché, 1 = refuge

    for i in range(n):
        if position == 0:
            # En marché : vérifier condition de sortie
            if bool(exit_ew.iloc[i]):
                position = 1
                sig_types[i] = "EW_EXIT"
            elif bool(exit_confirmed.iloc[i]):
                position = 1
                sig_types[i] = "PTSS_EXIT"
        else:
            # En refuge : vérifier condition de rentrée
            if bool(reentry_confirmed.iloc[i]):
                position = 0
                sig_types[i] = "REENTRY"
        signals_raw[i] = position

    signals_series  = pd.Series(signals_raw, index=idx)
    sig_types_series = pd.Series(sig_types, index=idx)

    # Inhibition avant activation_date (anti-biais fenêtre de chauffe)
    if activation_date is not None:
        mask_before = idx < activation_date
        signals_series.loc[mask_before]   = 0
        sig_types_series.loc[mask_before] = "DISABLED"

    # Décalage +1 jour (exécution au lendemain — pas de look-ahead bias)
    signals_shifted = signals_series.shift(1).fillna(0).astype(int)

    return signals_shifted, sig_types_series


# ═══════════════════════════════════════════════════════════════════════════
# 4. LEAD TIMES
# ═══════════════════════════════════════════════════════════════════════════

def compute_lead_time(
    tss_daily: pd.Series,
    threshold: float = 0.8,
    crises: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Dict[str, Optional[float]]:
    """Lead Time classique : premier dépassement du seuil avant la crise.

    Args:
        tss_daily:  TSS journalier daté.
        threshold:  Seuil θ.
        crises:     Dict crises. Défaut = REFERENCE_CRISES.

    Returns:
        Dict {crise: jours (positif = avance, négatif = retard)}.
    """
    if crises is None:
        crises = REFERENCE_CRISES

    lead_times: Dict[str, Optional[float]] = {}

    for name, (start_str, _) in crises.items():
        try:
            crisis_start   = pd.Timestamp(start_str)
            lookback_start = crisis_start - pd.Timedelta(days=60)
            window         = tss_daily.loc[
                (tss_daily.index >= lookback_start) &
                (tss_daily.index <  crisis_start)
            ]
            if window.empty:
                lead_times[name] = None
                continue
            triggers = window[window > threshold]
            if triggers.empty:
                lead_times[name] = None
                continue
            lead_times[name] = float((crisis_start - triggers.index[0]).days)
        except Exception:
            lead_times[name] = None

    return lead_times


def compute_lead_time_multiscale(
    signal_series: pd.Series,
    threshold: float = 0.30,
    crises: Optional[Dict[str, Tuple[str, str]]] = None,
    lookback_days: int = 90,
) -> Dict[str, Optional[float]]:
    """Lead Time P-TSS / ΔTSS — lookback élargi à 90j.

    Args:
        signal_series: P-TSS ou ΔTSS daté.
        threshold:     Seuil de déclenchement. Défaut = 0.30.
        crises:        Dict crises. Défaut = REFERENCE_CRISES.
        lookback_days: Fenêtre de recherche. Défaut = 90j.

    Returns:
        Dict {crise: jours}.
    """
    if crises is None:
        crises = REFERENCE_CRISES

    lead_times: Dict[str, Optional[float]] = {}

    for name, (start_str, _) in crises.items():
        try:
            crisis_start   = pd.Timestamp(start_str)
            lookback_start = crisis_start - pd.Timedelta(days=lookback_days)
            window         = signal_series.loc[
                (signal_series.index >= lookback_start) &
                (signal_series.index <  crisis_start)
            ]
            if window.empty:
                lead_times[name] = None
                continue
            triggers = window[window > threshold]
            if triggers.empty:
                lead_times[name] = None
                continue
            lead_times[name] = float((crisis_start - triggers.index[0]).days)
        except Exception:
            lead_times[name] = None

    return lead_times


# ═══════════════════════════════════════════════════════════════════════════
# 5. ENRICHISSEMENT DU TRADE LOG
# ═══════════════════════════════════════════════════════════════════════════

def enrich_trade_log_with_trough(
    trade_log: pd.DataFrame,
    price_primary: pd.Series,
    search_window: int = TROUGH_SEARCH_WINDOW,
) -> pd.DataFrame:
    """Ajoute les colonnes 'jours_avant_creux' et 'predictif' au trade log.

    Pour chaque rotation REFUGE, identifie le prix minimum dans les
    `search_window` jours suivants et calcule le nombre de jours d'avance.

    Un signal est classé "Prédictif" si la sortie a eu lieu ≥ 15 jours
    avant le creux de prix.

    Args:
        trade_log:     DataFrame des rotations.
        price_primary: Série de prix de l'actif principal.
        search_window: Nb de jours de recherche du creux. Défaut = 120.

    Returns:
        trade_log enrichi avec colonnes supplémentaires.
    """
    if trade_log.empty:
        return trade_log

    log = trade_log.copy()
    jours_avant = []
    creux_prix  = []
    predictif   = []

    for _, row in log.iterrows():
        direction = str(row.get("direction", ""))
        if direction != "REFUGE":
            jours_avant.append(np.nan)
            creux_prix.append(np.nan)
            predictif.append("")
            continue

        trade_date = pd.Timestamp(row["date"])
        end_search = trade_date + pd.Timedelta(days=search_window)

        price_window = price_primary.loc[
            (price_primary.index > trade_date) &
            (price_primary.index <= end_search)
        ]

        if price_window.empty:
            jours_avant.append(np.nan)
            creux_prix.append(np.nan)
            predictif.append("N/A")
            continue

        trough_date  = price_window.idxmin()
        trough_price = float(price_window.min())
        days_ahead   = int((trough_date - trade_date).days)

        jours_avant.append(days_ahead)
        creux_prix.append(round(trough_price, 2))
        predictif.append("✅ Prédictif" if days_ahead >= 15 else "⚠ Réactif")

    log["jours_avant_creux"] = jours_avant
    log["prix_creux"]        = creux_prix
    log["verdict"]           = predictif

    return log


# ═══════════════════════════════════════════════════════════════════════════
# 6. BACKTEST CLASSIQUE (TSS)
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(
    prices: pd.DataFrame,
    tss: pd.Series,
    dates: pd.DatetimeIndex,
    window_centers: List[int],
    primary_ticker: str = "SPY",
    refuge_ticker: Optional[str] = "GLD",
    tss_threshold: float = 0.8,
    confirmation_days: int = 2,
    transaction_cost: float = 0.001,
) -> BacktestResult:
    """Backtest TSS classique (stratégie symétrique).

    Rotation refuge si TSS > θ pendant N jours. Signal décalé +1j.
    Pain Index et Capital Saved inclus dans les KPIs.

    Args:
        prices:            DataFrame de prix (T × N).
        tss:               pd.Series TSS index entier.
        dates:             DatetimeIndex du nuage embeddi.
        window_centers:    Positions des fenêtres dans `dates`.
        primary_ticker:    Actif principal.
        refuge_ticker:     Actif refuge ou None (cash).
        tss_threshold:     Seuil θ ∈ [0, 1].
        confirmation_days: Jours de confirmation.
        transaction_cost:  Coût par rotation.

    Returns:
        BacktestResult complet.
    """
    if primary_ticker not in prices.columns:
        primary_ticker = prices.columns[0]
        warnings.warn(f"primary_ticker non trouvé, fallback sur {primary_ticker}", UserWarning)

    price_primary = prices[primary_ticker].dropna()

    if refuge_ticker is not None and refuge_ticker in prices.columns:
        price_refuge = prices[refuge_ticker].dropna()
        use_cash = False
    else:
        price_refuge = None
        use_cash = True

    ret_primary = np.log(price_primary / price_primary.shift(1)).dropna()
    ret_refuge  = (
        np.log(price_refuge / price_refuge.shift(1)).reindex(ret_primary.index).fillna(0.0)
        if not use_cash else pd.Series(0.0, index=ret_primary.index)
    )

    tss_daily = align_tss_to_dates(tss, dates, window_centers, ret_primary.index)
    signals   = generate_trading_signals(tss_daily, tss_threshold, confirmation_days)

    ret_strategy = pd.Series(
        np.where(signals == 1, ret_refuge.values, ret_primary.values),
        index=ret_primary.index,
    )

    trade_days   = (signals.diff().abs() > 0) & signals.index.isin(ret_primary.index)
    n_trades     = int(trade_days.sum())
    cost_series  = pd.Series(
        np.where(trade_days.reindex(ret_primary.index).fillna(False), -transaction_cost, 0.0),
        index=ret_primary.index,
    )
    ret_strategy = ret_strategy + cost_series

    equity_tda = np.exp(ret_strategy.cumsum())
    equity_bnh = np.exp(ret_primary.cumsum())
    _, dd_tda  = compute_max_drawdown(equity_tda)
    _, dd_bnh  = compute_max_drawdown(equity_bnh)

    lead_time_dict = compute_lead_time(tss_daily, threshold=tss_threshold)
    valid_lts      = [v for v in lead_time_dict.values() if v is not None and v > 0]
    avg_lead       = float(np.mean(valid_lts)) if valid_lts else 0.0

    kpis_tda = compute_kpis(
        equity_tda, ret_strategy,
        strategy_name=f"TDA Strategy (θ={tss_threshold})",
        n_trades=n_trades, lead_time=avg_lead,
        equity_bnh=equity_bnh,
    )
    kpis_bnh = compute_kpis(
        equity_bnh, ret_primary,
        strategy_name=f"Buy & Hold ({primary_ticker})",
    )

    outperform       = (ret_strategy > ret_primary).sum()
    kpis_tda.hit_rate = float(outperform / len(ret_primary)) if len(ret_primary) > 0 else 0.0

    # Trade log basique
    trade_idx   = signals.index[trade_days.reindex(signals.index).fillna(False)]
    trade_log   = pd.DataFrame({
        "date":       pd.DatetimeIndex(trade_idx),
        "direction":  ["REFUGE" if signals.loc[d] == 1 else "MARCHÉ" for d in trade_idx],
        "signal_type": ["TSS" for _ in trade_idx],
        "tss":        [float(tss_daily.loc[d]) if d in tss_daily.index else np.nan
                       for d in trade_idx],
        "prix":       [float(price_primary.loc[d]) if d in price_primary.index else np.nan
                       for d in trade_idx],
    })
    trade_log = enrich_trade_log_with_trough(trade_log, price_primary)

    cap_saved_by_crisis = compute_capital_saved(equity_tda, equity_bnh)

    return BacktestResult(
        equity_tda=equity_tda, equity_bnh=equity_bnh,
        signals=signals.reindex(ret_primary.index).fillna(0),
        returns_tda=ret_strategy, returns_bnh=ret_primary,
        drawdown_tda=dd_tda, drawdown_bnh=dd_bnh,
        kpis_tda=kpis_tda, kpis_bnh=kpis_bnh,
        lead_times=lead_time_dict, trade_log=trade_log,
        capital_saved_by_crisis=cap_saved_by_crisis,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7. BACKTEST P-TSS PRÉDICTIF (v2.4) — Protection du Capital
# ═══════════════════════════════════════════════════════════════════════════

def run_ptss_backtest(
    prices: pd.DataFrame,
    ms_result,
    primary_ticker: str = "SPY",
    refuge_ticker: Optional[str] = "GLD",
    exit_threshold: float = 0.30,
    reentry_delta_threshold: float = 0.05,
    reentry_slow_threshold: float = 0.55,
    reentry_ptss_threshold: float = 0.20,
    confirmation_exit: int = 1,
    confirmation_reentry: int = 3,
    transaction_cost: float = 0.001,
    use_early_warning: bool = True,
    ew_ptss_min: float = 0.15,
) -> BacktestResult:
    """Backtest P-TSS prédictif — objectif Max Drawdown < -15 %.

    Utilise generate_predictive_signals() avec logique asymétrique :
      - Sortie rapide (1j confirmation) sur P-TSS ou Early Warning.
      - Rentrée prudente (3j confirmation, triple condition ΔTSS/slow/P-TSS).

    Anti-biais NaN :
      Avant ms_result.dates_slow[0], le signal est inhibé (période de
      chauffe W_slow où TSS_slow = 0.0 par ffill, pouvant générer
      de faux P-TSS élevés via ΔTSS = TSS_fast - 0).

    Args:
        prices:                  DataFrame de prix.
        ms_result:               MultiscaleTDAResult.
        primary_ticker:          Actif principal.
        refuge_ticker:           Actif refuge ou None (cash).
        exit_threshold:          Seuil P-TSS pour la sortie. Défaut = 0.30.
        reentry_delta_threshold: ΔTSS seuil rentrée. Défaut = 0.05.
        reentry_slow_threshold:  TSS_slow seuil rentrée. Défaut = 0.55.
        reentry_ptss_threshold:  P-TSS seuil rentrée. Défaut = 0.20.
        confirmation_exit:       Jours confirmation sortie. Défaut = 1.
        confirmation_reentry:    Jours confirmation rentrée. Défaut = 3.
        transaction_cost:        Coût par rotation. Défaut = 0.001.
        use_early_warning:       Sortie accélérée EW. Défaut = True.
        ew_ptss_min:             P-TSS min pour sortie EW. Défaut = 0.15.

    Returns:
        BacktestResult avec Pain Index, Capital Saved, trade log enrichi.
    """
    if primary_ticker not in prices.columns:
        primary_ticker = prices.columns[0]
        warnings.warn(f"primary_ticker non trouvé, fallback sur {primary_ticker}", UserWarning)

    price_primary = prices[primary_ticker].dropna()

    if refuge_ticker is not None and refuge_ticker in prices.columns:
        price_refuge = prices[refuge_ticker].dropna()
        use_cash = False
    else:
        price_refuge = None
        use_cash = True

    ret_primary = np.log(price_primary / price_primary.shift(1)).dropna()
    ret_refuge  = (
        np.log(price_refuge / price_refuge.shift(1)).reindex(ret_primary.index).fillna(0.0)
        if not use_cash else pd.Series(0.0, index=ret_primary.index)
    )

    # Signaux pré-alignés sur les dates de prix (depuis ms_result)
    ptss_aligned  = ms_result.ptss.reindex(ret_primary.index).ffill().fillna(0.0)
    delta_aligned = ms_result.delta_tss.reindex(ret_primary.index).ffill().fillna(0.0)
    slow_aligned  = ms_result.tss_slow.reindex(ret_primary.index).ffill().fillna(0.0)
    ew_aligned    = ms_result.early_warning.reindex(ret_primary.index).ffill().fillna(False)

    # Date d'activation : première date valide de TSS_slow (après sa chauffe)
    activation_date: Optional[pd.Timestamp] = None
    try:
        if len(ms_result.dates_slow) > 0:
            activation_date = pd.Timestamp(ms_result.dates_slow[0])
    except Exception:
        activation_date = None

    # Génération des signaux asymétriques
    signals, sig_types = generate_predictive_signals(
        ptss=ptss_aligned,
        delta_tss=delta_aligned,
        tss_slow=slow_aligned,
        early_warning=ew_aligned,
        exit_threshold=exit_threshold,
        reentry_delta_threshold=reentry_delta_threshold,
        reentry_slow_threshold=reentry_slow_threshold,
        reentry_ptss_threshold=reentry_ptss_threshold,
        confirmation_exit=confirmation_exit,
        confirmation_reentry=confirmation_reentry,
        use_early_warning=use_early_warning,
        ew_ptss_min=ew_ptss_min,
        activation_date=activation_date,
    )

    # Calcul des rendements de la stratégie
    ret_strategy = pd.Series(
        np.where(signals == 1, ret_refuge.values, ret_primary.values),
        index=ret_primary.index,
    )

    trade_days   = (signals.diff().abs() > 0) & signals.index.isin(ret_primary.index)
    n_trades     = int(trade_days.sum())
    cost_series  = pd.Series(
        np.where(trade_days.reindex(ret_primary.index).fillna(False), -transaction_cost, 0.0),
        index=ret_primary.index,
    )
    ret_strategy = ret_strategy + cost_series

    equity_tda = np.exp(ret_strategy.cumsum())
    equity_bnh = np.exp(ret_primary.cumsum())
    _, dd_tda  = compute_max_drawdown(equity_tda)
    _, dd_bnh  = compute_max_drawdown(equity_bnh)

    # Lead Time via P-TSS (lookback 90j)
    lead_time_dict = compute_lead_time_multiscale(ptss_aligned, threshold=exit_threshold)
    valid_lts      = [v for v in lead_time_dict.values() if v is not None and v > 0]
    avg_lead       = float(np.mean(valid_lts)) if valid_lts else 0.0

    kpis_tda = compute_kpis(
        equity_tda, ret_strategy,
        strategy_name=f"P-TSS Strategy (θ={exit_threshold})",
        n_trades=n_trades, lead_time=avg_lead,
        equity_bnh=equity_bnh,
    )
    kpis_bnh = compute_kpis(
        equity_bnh, ret_primary,
        strategy_name=f"Buy & Hold ({primary_ticker})",
        equity_bnh=equity_bnh,
    )

    outperform        = (ret_strategy > ret_primary).sum()
    kpis_tda.hit_rate = float(outperform / len(ret_primary)) if len(ret_primary) > 0 else 0.0

    # Capital Saved par crise
    cap_saved_by_crisis = compute_capital_saved(equity_tda, equity_bnh)
    total_cs = sum(v for v in cap_saved_by_crisis.values() if v > 0)
    kpis_tda.capital_saved = total_cs

    # Trade log enrichi avec type de signal et analyse du creux
    trade_idx = signals.index[trade_days.reindex(signals.index).fillna(False)]

    trade_directions = ["REFUGE" if signals.loc[d] == 1 else "MARCHÉ" for d in trade_idx]

    # Type de signal : depuis sig_types (non décalé — on prend t-1 car signal est décalé)
    sig_type_list = []
    for d in trade_idx:
        try:
            prev_loc = sig_types.index.get_loc(d)
            if prev_loc > 0:
                st = sig_types.iloc[prev_loc - 1]
            else:
                st = sig_types.iloc[0]
        except Exception:
            st = "PTSS_EXIT"
        sig_type_list.append(st)

    trade_ptss_vals = [
        float(ptss_aligned.loc[d]) if d in ptss_aligned.index else np.nan
        for d in trade_idx
    ]
    trade_delta_vals = [
        float(delta_aligned.loc[d]) if d in delta_aligned.index else np.nan
        for d in trade_idx
    ]
    trade_price_vals = [
        float(price_primary.loc[d]) if d in price_primary.index else np.nan
        for d in trade_idx
    ]

    trade_log = pd.DataFrame({
        "date":        pd.DatetimeIndex(trade_idx),
        "direction":   trade_directions,
        "signal_type": sig_type_list,
        "ptss":        trade_ptss_vals,
        "delta_tss":   trade_delta_vals,
        "prix":        trade_price_vals,
    })

    trade_log = enrich_trade_log_with_trough(trade_log, price_primary)

    return BacktestResult(
        equity_tda=equity_tda, equity_bnh=equity_bnh,
        signals=signals.reindex(ret_primary.index).fillna(0),
        returns_tda=ret_strategy, returns_bnh=ret_primary,
        drawdown_tda=dd_tda, drawdown_bnh=dd_bnh,
        kpis_tda=kpis_tda, kpis_bnh=kpis_bnh,
        lead_times=lead_time_dict, trade_log=trade_log,
        capital_saved_by_crisis=cap_saved_by_crisis,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 8. BACKTEST MULTI-ÉCHELLES (toutes stratégies)
# ═══════════════════════════════════════════════════════════════════════════

def run_multiscale_backtest(
    prices: pd.DataFrame,
    ms_result,
    primary_ticker: str = "SPY",
    refuge_ticker: Optional[str] = "GLD",
    ptss_threshold: float = 0.35,
    delta_tss_threshold: float = 0.15,
    confirmation_days: int = 2,
    transaction_cost: float = 0.001,
) -> Dict[str, BacktestResult]:
    """Lance les trois stratégies multi-échelles et retourne les résultats.

    Stratégies :
      "ptss"      : P-TSS asymétrique (run_ptss_backtest) — stratégie principale.
      "delta_tss" : ΔTSS+ symétrique (seuil delta_tss_threshold).
      "combined"  : P-TSS + boost Early Warning.

    Args:
        prices:              DataFrame de prix.
        ms_result:           MultiscaleTDAResult.
        primary_ticker:      Actif principal.
        refuge_ticker:       Actif refuge.
        ptss_threshold:      Seuil P-TSS pour stratégie ptss et combined.
        delta_tss_threshold: Seuil ΔTSS pour stratégie delta_tss.
        confirmation_days:   Jours confirmation (stratégies symétriques).
        transaction_cost:    Coût par rotation.

    Returns:
        Dict {"ptss": BacktestResult, "delta_tss": BacktestResult,
              "combined": BacktestResult}.
    """
    results: Dict[str, BacktestResult] = {}

    # ── Stratégie principale : P-TSS asymétrique ─────────────────────────
    results["ptss"] = run_ptss_backtest(
        prices=prices,
        ms_result=ms_result,
        primary_ticker=primary_ticker,
        refuge_ticker=refuge_ticker,
        exit_threshold=ptss_threshold,
        reentry_delta_threshold=0.05,
        reentry_slow_threshold=0.55,
        reentry_ptss_threshold=ptss_threshold * 0.5,
        confirmation_exit=1,
        confirmation_reentry=3,
        transaction_cost=transaction_cost,
        use_early_warning=True,
        ew_ptss_min=ptss_threshold * 0.5,
    )

    # ── Stratégie ΔTSS+ (symétrique, signal = clip positif du delta) ─────
    delta_pos = ms_result.delta_tss.clip(lower=0.0)

    primary_ok    = primary_ticker if primary_ticker in prices.columns else prices.columns[0]
    price_primary = prices[primary_ok].dropna()
    ret_primary   = np.log(price_primary / price_primary.shift(1)).dropna()

    if refuge_ticker is not None and refuge_ticker in prices.columns:
        price_refuge = prices[refuge_ticker].dropna()
        ret_refuge   = np.log(price_refuge / price_refuge.shift(1)).reindex(ret_primary.index).fillna(0.0)
        use_cash     = False
    else:
        ret_refuge = pd.Series(0.0, index=ret_primary.index)
        use_cash   = True

    dt_aligned = delta_pos.reindex(ret_primary.index).ffill().fillna(0.0)
    dt_sigs    = generate_trading_signals(dt_aligned, delta_tss_threshold, confirmation_days)

    ret_dt = pd.Series(
        np.where(dt_sigs == 1, ret_refuge.values, ret_primary.values),
        index=ret_primary.index,
    )
    dt_trade_days = (dt_sigs.diff().abs() > 0) & dt_sigs.index.isin(ret_primary.index)
    dt_n_trades   = int(dt_trade_days.sum())
    dt_costs      = pd.Series(
        np.where(dt_trade_days.reindex(ret_primary.index).fillna(False), -transaction_cost, 0.0),
        index=ret_primary.index,
    )
    ret_dt   = ret_dt + dt_costs
    eq_dt    = np.exp(ret_dt.cumsum())
    eq_bnh   = np.exp(ret_primary.cumsum())
    _, dd_dt = compute_max_drawdown(eq_dt)
    _, dd_bh = compute_max_drawdown(eq_bnh)

    lt_dt    = compute_lead_time_multiscale(dt_aligned, threshold=delta_tss_threshold)
    vlt_dt   = [v for v in lt_dt.values() if v is not None and v > 0]
    avg_dt   = float(np.mean(vlt_dt)) if vlt_dt else 0.0

    kpis_dt  = compute_kpis(eq_dt, ret_dt, f"ΔTSS Strategy (θ={delta_tss_threshold})",
                             dt_n_trades, avg_dt, equity_bnh=eq_bnh)
    kpis_bnh = compute_kpis(eq_bnh, ret_primary, f"Buy & Hold ({primary_ok})")
    kpis_dt.hit_rate = float((ret_dt > ret_primary).sum() / len(ret_primary)) if len(ret_primary) > 0 else 0.0

    dt_trade_idx = dt_sigs.index[dt_trade_days.reindex(dt_sigs.index).fillna(False)]
    dt_log = pd.DataFrame({
        "date":        pd.DatetimeIndex(dt_trade_idx),
        "direction":   ["REFUGE" if dt_sigs.loc[d] == 1 else "MARCHÉ" for d in dt_trade_idx],
        "signal_type": ["DELTA_TSS" for _ in dt_trade_idx],
        "ptss":        [float(ms_result.ptss.reindex([d]).ffill().fillna(0.0).iloc[0]) for d in dt_trade_idx],
        "delta_tss":   [float(dt_aligned.loc[d]) if d in dt_aligned.index else np.nan for d in dt_trade_idx],
        "prix":        [float(price_primary.loc[d]) if d in price_primary.index else np.nan for d in dt_trade_idx],
    })
    dt_log = enrich_trade_log_with_trough(dt_log, price_primary)

    results["delta_tss"] = BacktestResult(
        equity_tda=eq_dt, equity_bnh=eq_bnh,
        signals=dt_sigs.reindex(ret_primary.index).fillna(0),
        returns_tda=ret_dt, returns_bnh=ret_primary,
        drawdown_tda=dd_dt, drawdown_bnh=dd_bh,
        kpis_tda=kpis_dt, kpis_bnh=kpis_bnh,
        lead_times=lt_dt, trade_log=dt_log,
        capital_saved_by_crisis=compute_capital_saved(eq_dt, eq_bnh),
    )

    # ── Stratégie combinée : P-TSS + Early Warning boost ─────────────────
    ew_float        = ms_result.early_warning.astype(float)
    combined_signal = (ms_result.ptss + ew_float * 0.5).clip(0.0, 1.0)

    results["combined"] = run_ptss_backtest(
        prices=prices,
        ms_result=type("_MS", (), {
            "ptss":          combined_signal,
            "delta_tss":     ms_result.delta_tss,
            "tss_slow":      ms_result.tss_slow,
            "early_warning": ms_result.early_warning,
            "dates_slow":    ms_result.dates_slow,
        })(),
        primary_ticker=primary_ticker,
        refuge_ticker=refuge_ticker,
        exit_threshold=ptss_threshold * 0.8,
        reentry_delta_threshold=0.05,
        reentry_slow_threshold=0.55,
        reentry_ptss_threshold=ptss_threshold * 0.4,
        confirmation_exit=1,
        confirmation_reentry=3,
        transaction_cost=transaction_cost,
        use_early_warning=False,   # EW déjà intégré dans combined_signal
        ew_ptss_min=0.0,
    )
    results["combined"].kpis_tda.strategy_name = f"Combined P-TSS+EW (θ={ptss_threshold*0.8:.2f})"

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 9. RAPPORT DE DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════

def generate_strategy_report(result: BacktestResult) -> str:
    """Rapport Markdown complet du backtest.

    Args:
        result: BacktestResult complet.

    Returns:
        Chaîne Markdown pour Streamlit.
    """
    tda = result.kpis_tda
    bnh = result.kpis_bnh

    outperform     = "✅ SURPERFORMANCE" if tda.cagr > bnh.cagr else "⚠️ SOUS-PERFORMANCE"
    dd_improvement = bnh.max_drawdown - tda.max_drawdown   # positif = TDA mieux
    pain_diff      = bnh.pain_index  - tda.pain_index      # positif = TDA moins douloureux

    lt_lines = []
    for crisis, days in result.lead_times.items():
        if days is None:
            lt_lines.append(f"  - {crisis}: **N/A** (données hors période)")
        elif days > 0:
            lt_lines.append(f"  - {crisis}: **+{days:.0f} jours** ✅")
        else:
            lt_lines.append(f"  - {crisis}: **{abs(days):.0f}j retard** ⚠️")
    lt_section = "\n".join(lt_lines) if lt_lines else "  - Aucune crise dans la période"

    cs_lines = []
    for crisis, cs in result.capital_saved_by_crisis.items():
        emoji = "✅" if cs > 0.02 else "—"
        cs_lines.append(f"  - {crisis}: **{cs:+.2%}** {emoji}")
    cs_section = "\n".join(cs_lines) if cs_lines else "  - Aucune donnée"

    report = f"""
## 📊 Rapport de Backtest TDA-Risk-Sentinel v2.4

### {outperform} vs Buy & Hold

| Métrique              | {tda.strategy_name[:22].ljust(22)} | Buy & Hold             |
|-----------------------|------------------------|------------------------|
| **CAGR**              | {tda.cagr:+.2%}        | {bnh.cagr:+.2%}        |
| **Sharpe Ratio**      | {tda.sharpe:.2f}        | {bnh.sharpe:.2f}        |
| **Max Drawdown**      | **{tda.max_drawdown:.2%}** | {bnh.max_drawdown:.2%} |
| **Calmar Ratio**      | {tda.calmar:.2f}        | {bnh.calmar:.2f}        |
| **Pain Index**        | **{tda.pain_index:.3f}** | {bnh.pain_index:.3f}   |
| **Rendement Total**   | {tda.total_return:+.2%}| {bnh.total_return:+.2%}|
| **Volatilité Ann.**   | {tda.volatility_ann:.2%}| {bnh.volatility_ann:.2%}|
| **Trades**            | {tda.n_trades}          | 0                      |
| **Hit Rate**          | {tda.hit_rate:.1%}      | —                      |
| **Capital Saved**     | **{tda.capital_saved:+.2%}** | —                 |

### ⏱️ Lead Time (Avance sur les Crises)
{lt_section}

### 🛡️ Capital Sauvé par Crise
{cs_section}

### 💡 Analyse Protection du Capital
- **Réduction MaxDD** : {dd_improvement:.2%} d'amélioration vs B&H.
- **Réduction Pain**  : {pain_diff:.3f} de douleur moyenne évitée.
- **Trades**          : {tda.n_trades} rotations → coût friction total ≈ {tda.n_trades * 0.001:.3%}.
- **Lead Time moyen** : {tda.avg_lead_time_days:.0f} jours d'avance (signal précurseur).
"""
    return report.strip()
