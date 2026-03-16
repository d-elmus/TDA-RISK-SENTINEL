"""
app.py — TDA-Risk-Sentinel Pro v2.0
Interface Streamlit "Finance Pro" — Détection Prédictive des Krachs

Architecture du dashboard :
  ┌─────────────────────────────────────────────────────┐
  │  HEADER — Titre + Status Bar + Métriques clés       │
  ├─────────────────────────────────────────────────────┤
  │  SIDEBAR — Paramètres, Actifs, Backtest             │
  ├─────────────┬───────────────────────────────────────┤
  │  TSS Gauge  │  Décomposition Radar                  │
  │  (30%)      │  (70%)                                │
  ├─────────────┴───────────────────────────────────────┤
  │  Vortex de Marché 3D Animé                          │
  ├─────────────────────┬───────────────────────────────┤
  │  TSS + Sensibilité  │  Radar Topologique (Barcode)  │
  ├─────────────────────┴───────────────────────────────┤
  │  Prix + Risk Overlay                                │
  ├─────────────────────────────────────────────────────┤
  │  [BACKTEST] Équité + Drawdown + KPIs + Rapport      │
  ├─────────────────────────────────────────────────────┤
  │  [EXPERT] Composantes + Diagnostic + Tests          │
  └─────────────────────────────────────────────────────┘

Usage : streamlit run app.py
"""

import time
import json
import warnings
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ── Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TDA-Risk-Sentinel Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "TDA-Risk-Sentinel Pro v2.0 — Détection de régime de marché par Homologie Persistante.",
    },
)

# ── CSS Finance Pro Dark Theme ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Inter:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080c14;
  }

  .main { background-color: #080c14; }
  .block-container { padding: 0.8rem 1.2rem; max-width: 100%; }

  /* Header principal */
  .sentinel-header {
    background: linear-gradient(135deg, #0d1421 0%, #111827 50%, #0d1929 100%);
    border: 1px solid #1a2235;
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .sentinel-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8eaf6;
    letter-spacing: 0.08em;
    margin: 0;
  }
  .sentinel-subtitle {
    font-size: 0.75rem;
    color: #78909c;
    margin: 4px 0 0 0;
    letter-spacing: 0.04em;
  }

  /* Status badges */
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
  }
  .status-safe     { background: rgba(0,230,118,0.12);  border: 1px solid #00e676; color: #00e676; }
  .status-warn     { background: rgba(255,193,7,0.12);  border: 1px solid #ffc107; color: #ffc107; }
  .status-danger   { background: rgba(255,23,68,0.15);  border: 1px solid #ff1744; color: #ff1744; }
  .status-critical { background: rgba(213,0,0,0.20);    border: 1px solid #d50000; color: #ff5252;
                     animation: pulse 1.5s infinite; }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.65; }
  }

  /* Metric cards */
  .metric-row { display: flex; gap: 10px; margin: 10px 0; }
  .metric-card {
    flex: 1;
    background: linear-gradient(145deg, #0d1421, #111827);
    border: 1px solid #1a2235;
    border-radius: 8px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #00b0ff);
  }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
    color: var(--accent, #e8eaf6);
  }
  .metric-label { font-size: 0.72rem; color: #78909c; text-transform: uppercase;
                  letter-spacing: 0.08em; margin-top: 4px; }
  .metric-delta { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
                  margin-top: 6px; }

  /* Section headers */
  .section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #00b0ff;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 18px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1a2235;
  }

  /* Alert boxes */
  .alert {
    padding: 12px 16px;
    border-radius: 6px;
    margin: 8px 0;
    font-size: 0.88rem;
    line-height: 1.6;
  }
  .alert-critical { background: rgba(213,0,0,0.12);   border-left: 3px solid #d50000; color: #ff8a80; }
  .alert-danger   { background: rgba(255,23,68,0.10);  border-left: 3px solid #ff1744; color: #ff8a80; }
  .alert-warn     { background: rgba(255,193,7,0.10);  border-left: 3px solid #ffc107; color: #ffd54f; }
  .alert-safe     { background: rgba(0,230,118,0.08);  border-left: 3px solid #00e676; color: #69f0ae; }
  .alert-info     { background: rgba(0,176,255,0.08);  border-left: 3px solid #00b0ff; color: #80d8ff; }

  /* KPI Table */
  .kpi-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .kpi-table th {
    background: #111827;
    color: #78909c;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 8px 12px;
    border-bottom: 1px solid #1a2235;
    text-align: left;
  }
  .kpi-table td {
    padding: 7px 12px;
    border-bottom: 1px solid #0d1421;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #e8eaf6;
  }
  .kpi-better { color: #00e676; font-weight: 600; }
  .kpi-worse  { color: #ff5252; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #080c14;
    border-right: 1px solid #1a2235;
  }
  [data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: #00b0ff;
    border-bottom: 1px solid #1a2235;
    padding-bottom: 4px;
  }

  /* Divider */
  .pro-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1a2235 20%, #00b0ff40 50%, #1a2235 80%, transparent);
    margin: 16px 0;
  }
</style>
""", unsafe_allow_html=True)

# ── Imports locaux ─────────────────────────────────────────────────────────
from data_pipeline import fetch_market_data, get_data_summary, DEFAULT_TICKERS, _simulate_all
from tda_engine import (
    run_tda_pipeline_full,
    sensitivity_analysis,
    run_multiscale_pipeline,
    MULTISCALE_CONFIG,
)
from backtester import run_backtest, generate_strategy_report, run_multiscale_backtest
from visualizations import (
    plot_market_vortex,
    plot_topological_radar,
    plot_prices_risk_overlay,
    plot_tss_with_sensitivity,
    plot_backtest_dashboard,
    plot_tss_gauge,
    plot_components_radar,
    plot_multiscale_tss,
    plot_multiscale_overlay,
)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

ALL_TICKERS = ["SPY", "QQQ", "BTC-USD", "GLD", "GSG", "VNQ", "TLT", "SHY", "VIXY"]
REFUGE_OPTIONS = {"Or (GLD)": "GLD", "Bons du Trésor (TLT)": "TLT",
                  "Cash (0%)": None, "Obligations CT (SHY)": "SHY"}

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 8px'>
      <div style='font-family:JetBrains Mono,monospace;font-size:1.1rem;
                  color:#00b0ff;letter-spacing:0.12em'>TDA-SENTINEL</div>
      <div style='font-size:0.65rem;color:#455a64;letter-spacing:0.15em'>PRO v2.4</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📦 Univers d'Actifs")

    tickers_sel = st.multiselect(
        "Actifs analysés",
        options=ALL_TICKERS,
        default=DEFAULT_TICKERS,
        help="3 à 7 actifs pour une géométrie de corrélation significative.",
    )
    if len(tickers_sel) < 2:
        st.error("Minimum 2 actifs requis.")
        tickers_sel = DEFAULT_TICKERS

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Début", value=pd.to_datetime("2019-01-01"))
    with col_d2:
        end_date = st.date_input("Fin", value=pd.to_datetime("2024-12-31"))

    use_sim = st.checkbox("Mode simulation (hors-ligne)", value=False)

    st.markdown("---")
    st.markdown("### 🔬 Moteur TDA")

    window_size  = st.slider("Fenêtre W (jours)", 30, 120, 60, 5,
                              help="Fenêtre Z-score et homologie.")
    step_size    = st.slider("Pas entre fenêtres", 1, 20, 5, 1,
                              help="↓ Pas = ↑ résolution temporelle, ↑ temps de calcul.")
    embed_dim    = st.select_slider("Dimension Takens d", [2, 3, 4, 5], 3)
    embed_delay  = st.slider("Délai Takens τ", 1, 5, 1)
    thresh_meth  = st.selectbox(
        "Seuillage persistance",
        ["iqr", "median", "pct90"],
        index=0,
        help="IQR = robuste. Median = équilibré. Pct90 = strict.",
    )
    tss_weights_preset = st.selectbox(
        "Pondération TSS",
        ["Équilibrée (45/30/25)", "Vitesse (60/25/15)", "Entropie (30/20/50)"],
        index=0,
    )
    _w_map = {
        "Équilibrée (45/30/25)": (0.45, 0.30, 0.25),
        "Vitesse (60/25/15)":    (0.60, 0.25, 0.15),
        "Entropie (30/20/50)":   (0.30, 0.20, 0.50),
    }
    tss_weights = _w_map[tss_weights_preset]

    run_sensitivity = st.checkbox(
        "Analyse de sensibilité",
        value=False,
        help="Teste W±5, τ∈{1,2,3}. Ajoute ~20s de calcul.",
    )

    st.markdown("---")
    st.markdown("### ⚡ Multi-Échelles TDA")

    run_multiscale = st.checkbox(
        "Activer Multi-Échelles",
        value=True,
        help="ΔTSS + P-TSS + Early Warning pour augmenter le Lead Time.",
    )
    if run_multiscale:
        ms_w_fast = st.slider(
            "W_fast (fenêtre courte)",
            min_value=10, max_value=60,
            value=int(MULTISCALE_CONFIG["W_FAST"]),
            step=5,
            help="Capture les micro-instabilités topologiques.",
        )
        ms_w_slow = st.slider(
            "W_slow (fenêtre longue)",
            min_value=60, max_value=250,
            value=int(MULTISCALE_CONFIG["W_SLOW"]),
            step=10,
            help="Représente la structure de fond du marché.",
        )
        ms_ew_k = st.slider(
            "Coefficient EW k (Early Warning σ)",
            min_value=1.0, max_value=3.5,
            value=float(MULTISCALE_CONFIG["EW_ACCEL_K"]),
            step=0.5,
            help="ΔTSS > μ + k·σ déclenche l'alerte précoce.",
        )
        ms_ptss_thr = st.slider(
            "Seuil P-TSS (backtest)",
            min_value=0.10, max_value=0.60,
            value=0.35, step=0.05,
            help="Seuil de déclenchement du backtest P-TSS.",
        )
    else:
        ms_w_fast = int(MULTISCALE_CONFIG["W_FAST"])
        ms_w_slow = int(MULTISCALE_CONFIG["W_SLOW"])
        ms_ew_k   = float(MULTISCALE_CONFIG["EW_ACCEL_K"])
        ms_ptss_thr = 0.35

    st.markdown("---")
    st.markdown("### 💹 Backtest")

    run_bt      = st.checkbox("Activer le backtest", value=True)
    primary_tkr = st.selectbox("Actif principal", tickers_sel, index=0)
    refuge_lbl  = st.selectbox("Actif refuge", list(REFUGE_OPTIONS.keys()), index=0)
    refuge_tkr  = REFUGE_OPTIONS[refuge_lbl]
    bt_threshold = st.slider("Seuil d'alerte TSS (θ)", 0.50, 0.95, 0.80, 0.05,
                              help="TSS > θ déclenche la rotation vers l'actif refuge.")
    bt_confirm   = st.slider("Jours de confirmation", 1, 5, 2,
                              help="Nombre de jours consécutifs au-dessus du seuil.")
    bt_cost      = st.number_input("Coût de transaction", 0.0, 0.01, 0.001, 0.0005,
                                    format="%.4f")

    st.markdown("---")
    st.markdown("### 🎨 Affichage")

    selected_ticker = st.selectbox("Actif overlay prix", tickers_sel, index=0)
    show_animation  = st.checkbox("Animer le Vortex 3D", value=True,
                                   help="Désactiver pour performances (données longues).")
    show_expert     = st.checkbox("Section Expert", value=True)

    st.markdown("---")
    run_btn = st.button("▶ ANALYSER", type="primary", use_container_width=True)

    st.markdown("""
    <div style='font-size:0.65rem;color:#37474f;margin-top:12px;line-height:1.8'>
    TDA-Risk-Sentinel Pro v2.4<br>
    Multi-Échelles · P-TSS · Early Warning<br>
    Homologie Persistante · Entropie · Wasserstein
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="sentinel-header">
  <div>
    <div class="sentinel-title">🛡️ TDA-RISK-SENTINEL PRO</div>
    <div class="sentinel-subtitle">
      Détection Prédictive des Krachs · Homologie Persistante · Espace des Phases
    </div>
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#455a64;text-align:right">
    v2.4 · Claude Sonnet 4.6<br>
    Multi-Échelles · P-TSS · Early Warning
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state["results"] = None
if "bt_result" not in st.session_state:
    st.session_state["bt_result"] = None
if "sensitivity" not in st.session_state:
    st.session_state["sensitivity"] = None
if "ms_result" not in st.session_state:
    st.session_state["ms_result"] = None
if "ms_bt_results" not in st.session_state:
    st.session_state["ms_bt_results"] = None


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE AVEC CACHE
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_pipeline(
    tickers, start, end, window, step, dim, delay,
    thresh_method, weights, simulated
):
    """Pipeline TDA complet mis en cache."""
    warnings.filterwarnings("ignore")
    tickers = list(tickers)

    if simulated:
        prices = _simulate_all(tickers, start, end)
    else:
        prices = fetch_market_data(tickers, start=start, end=end)

    result = run_tda_pipeline_full(
        prices,
        window=window,
        step=step,
        embed_dim=dim,
        embed_delay=delay,
        threshold_method=thresh_method,
        tss_weights=weights,
    )
    result["prices"] = prices
    result["summary"] = get_data_summary(prices)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_sensitivity(tickers, start, end, window, delay, step, dim, simulated):
    """Analyse de sensibilité mise en cache."""
    warnings.filterwarnings("ignore")
    if simulated:
        prices = _simulate_all(list(tickers), start, end)
    else:
        prices = fetch_market_data(list(tickers), start=start, end=end)
    return sensitivity_analysis(prices, base_window=window, base_delay=delay, step=step, embed_dim=dim)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_multiscale(
    tickers, start, end, w_fast, w_slow, dim, delay,
    thresh_method, weights, ew_k, simulated
):
    """Pipeline Multi-Échelles TDA mis en cache."""
    warnings.filterwarnings("ignore")
    if simulated:
        prices = _simulate_all(list(tickers), start, end)
    else:
        prices = fetch_market_data(list(tickers), start=start, end=end)
    return run_multiscale_pipeline(
        prices,
        w_fast=w_fast,
        w_slow=w_slow,
        step_fast=max(3, w_fast // 4),
        step_slow=max(5, w_slow // 10),
        embed_dim=dim,
        embed_delay=delay,
        threshold_method=thresh_method,
        tss_weights=weights,
        ew_accel_window=max(10, w_fast),
        ew_accel_k=ew_k,
    )


# ═══════════════════════════════════════════════════════════════════════════
# EXÉCUTION
# ═══════════════════════════════════════════════════════════════════════════

if run_btn or st.session_state["results"] is not None:

    if run_btn:
        progress = st.progress(0, text="Initialisation du pipeline TDA...")

        try:
            t0 = time.time()

            progress.progress(5,  text="Chargement des données de marché...")
            results = _cached_pipeline(
                tuple(tickers_sel), str(start_date), str(end_date),
                window_size, step_size, embed_dim, embed_delay,
                thresh_meth, tss_weights, use_sim,
            )

            progress.progress(45, text="Homologie persistante & entropie...")
            time.sleep(0.05)

            progress.progress(65, text="Calcul du TSS composite...")
            bt_result = None
            if run_bt:
                from backtester import run_backtest
                bt_result = run_backtest(
                    prices=results["prices"],
                    tss=results["tss"],
                    dates=results["dates"],
                    window_centers=results["window_centers"],
                    primary_ticker=primary_tkr,
                    refuge_ticker=refuge_tkr,
                    tss_threshold=bt_threshold,
                    confirmation_days=bt_confirm,
                    transaction_cost=bt_cost,
                )
                progress.progress(80, text="Backtest terminé...")

            sensitivity = None
            if run_sensitivity:
                progress.progress(82, text="Analyse de sensibilité (W±5, τ∈{1,2,3})...")
                sensitivity = _cached_sensitivity(
                    tuple(tickers_sel), str(start_date), str(end_date),
                    window_size, embed_delay, step_size, embed_dim, use_sim,
                )

            ms_result    = None
            ms_bt_results = None
            if run_multiscale:
                progress.progress(86, text=f"Pipeline Multi-Échelles (W_fast={ms_w_fast}, W_slow={ms_w_slow})...")
                try:
                    ms_result = _cached_multiscale(
                        tuple(tickers_sel), str(start_date), str(end_date),
                        ms_w_fast, ms_w_slow, embed_dim, embed_delay,
                        thresh_meth, tss_weights, ms_ew_k, use_sim,
                    )
                    progress.progress(94, text="Backtest Multi-Échelles (P-TSS, ΔTSS, Combined)...")
                    ms_bt_results = run_multiscale_backtest(
                        prices=results["prices"],
                        ms_result=ms_result,
                        primary_ticker=primary_tkr,
                        refuge_ticker=refuge_tkr,
                        ptss_threshold=ms_ptss_thr,
                        confirmation_days=bt_confirm,
                        transaction_cost=bt_cost,
                    )
                except Exception as e_ms:
                    warnings.warn(f"Multi-échelles échoué : {e_ms}", RuntimeWarning)

            elapsed = time.time() - t0
            progress.progress(100, text=f"✓ Pipeline complet en {elapsed:.1f}s")
            time.sleep(0.25)
            progress.empty()

            st.session_state["results"]       = results
            st.session_state["bt_result"]     = bt_result
            st.session_state["sensitivity"]   = sensitivity
            st.session_state["ms_result"]     = ms_result
            st.session_state["ms_bt_results"] = ms_bt_results
            st.session_state["elapsed"]       = elapsed
            st.session_state["params"]        = dict(
                tickers=tickers_sel, primary=primary_tkr, refuge_lbl=refuge_lbl,
                threshold=bt_threshold, ticker_display=selected_ticker,
                animate=show_animation, expert=show_expert,
                ms_ptss_thr=ms_ptss_thr,
            )

        except Exception as e:
            progress.empty()
            st.error(f"**Erreur pipeline** : {e}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            st.stop()

    # ── Récupérer les résultats ──
    results       = st.session_state["results"]
    bt_result     = st.session_state["bt_result"]
    sensitivity   = st.session_state["sensitivity"]
    ms_result     = st.session_state.get("ms_result")
    ms_bt_results = st.session_state.get("ms_bt_results")
    elapsed       = st.session_state.get("elapsed", 0)
    params        = st.session_state.get("params", {})

    prices         = results["prices"]
    tss            = results["tss"]
    point_cloud    = results["point_cloud"]
    dates          = results["dates"]
    diagrams_list  = results["diagrams_list"]
    window_centers = results["window_centers"]
    components_df  = results["components_df"]
    descriptors    = results["descriptors"]
    summary        = results["summary"]

    current_tss    = float(tss.iloc[-1]) if len(tss) > 0 else 0.0
    max_tss        = float(tss.max())    if len(tss) > 0 else 0.0
    prev_tss_5     = float(tss.iloc[-6]) if len(tss) > 6 else float(tss.iloc[0])
    tss_delta      = current_tss - prev_tss_5
    current_entropy = float(descriptors[-1].entropy) if descriptors else 0.0
    current_n_h1    = int(descriptors[-1].n_h1)      if descriptors else 0

    # Régime
    if current_tss >= 0.80:
        regime_cls  = "status-critical"
        regime_txt  = "⚡ CRITIQUE"
        alert_cls   = "alert-critical"
        alert_icon  = "🔴"
    elif current_tss >= 0.70:
        regime_cls  = "status-danger"
        regime_txt  = "⚠ STRESS"
        alert_cls   = "alert-danger"
        alert_icon  = "🟠"
    elif current_tss >= 0.40:
        regime_cls  = "status-warn"
        regime_txt  = "⚡ TRANSITION"
        alert_cls   = "alert-warn"
        alert_icon  = "🟡"
    else:
        regime_cls  = "status-safe"
        regime_txt  = "✅ CALME"
        alert_cls   = "alert-safe"
        alert_icon  = "🟢"

    trend_sym = "↗" if tss_delta > 0.03 else "↘" if tss_delta < -0.03 else "→"

    # ─────────────────────────────────────────────────────────────────────
    # STATUS BAR
    # ─────────────────────────────────────────────────────────────────────

    src = "SIMULATION" if use_sim else "YAHOO FINANCE"
    st.markdown(f"""
    <div style='display:flex;gap:10px;align-items:center;margin-bottom:14px;flex-wrap:wrap'>
      <span class='status-badge {regime_cls}'>{regime_txt}</span>
      <span class='status-badge' style='background:rgba(0,176,255,0.08);
            border:1px solid #1a2235;color:#78909c'>
        {summary['n_days']}j · {summary['n_assets']}A · {src}
      </span>
      <span class='status-badge' style='background:rgba(0,176,255,0.08);
            border:1px solid #1a2235;color:#78909c'>
        ⏱ {elapsed:.1f}s · {len(diagrams_list)} fenêtres TDA
      </span>
      <span style='margin-left:auto;font-family:JetBrains Mono,monospace;
                   font-size:0.72rem;color:#37474f'>
        {summary['start']} → {summary['end']}
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # MÉTRIQUES PRINCIPALES
    # ─────────────────────────────────────────────────────────────────────

    tss_color   = "#ff1744" if current_tss > 0.7 else "#ffc107" if current_tss > 0.4 else "#00e676"
    ent_color   = "#00e676" if current_entropy > 2.0 else "#ffc107" if current_entropy > 1.0 else "#ff1744"
    delta_color = "#ff5252" if tss_delta > 0 else "#69f0ae"

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card" style="--accent:{tss_color}">
        <div class="metric-label">TSS Actuel</div>
        <div class="metric-value" style="color:{tss_color}">{current_tss:.3f}</div>
        <div class="metric-delta" style="color:{delta_color}">{trend_sym} {abs(tss_delta):.3f} (Δ5)</div>
      </div>
      <div class="metric-card" style="--accent:{ent_color}">
        <div class="metric-label">Entropie H₁</div>
        <div class="metric-value" style="color:{ent_color}">{current_entropy:.2f}</div>
        <div class="metric-delta" style="color:#78909c">bits · {'Diversifié' if current_entropy > 2 else 'Monolithique'}</div>
      </div>
      <div class="metric-card" style="--accent:#7c4dff">
        <div class="metric-label">Cycles H₁ Stables</div>
        <div class="metric-value" style="color:#7c4dff">{current_n_h1}</div>
        <div class="metric-delta" style="color:#78909c">après seuillage IQR</div>
      </div>
      <div class="metric-card" style="--accent:#ff6d00">
        <div class="metric-label">TSS Pic Historique</div>
        <div class="metric-value" style="color:#ff6d00">{max_tss:.3f}</div>
        <div class="metric-delta" style="color:#78909c">{'⚠ Proche du pic' if current_tss > 0.85 * max_tss else 'Normal'}</div>
      </div>
      <div class="metric-card" style="--accent:#00b0ff">
        <div class="metric-label">Données</div>
        <div class="metric-value" style="font-size:1.3rem;color:#00b0ff">
          {summary['n_days']}j
        </div>
        <div class="metric-delta" style="color:#78909c">{elapsed:.1f}s · {len(window_centers)} fenêtres</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # GAUGE + RADAR COMPONENTS
    # ─────────────────────────────────────────────────────────────────────

    col_gauge, col_radar_comp = st.columns([0.28, 0.72])

    with col_gauge:
        st.plotly_chart(
            plot_tss_gauge(current_tss, "TSS Actuel"),
            use_container_width=True,
        )

    with col_radar_comp:
        st.plotly_chart(
            plot_components_radar(components_df, idx=-1),
            use_container_width=True,
        )

    # ─────────────────────────────────────────────────────────────────────
    # VORTEX DE MARCHÉ 3D
    # ─────────────────────────────────────────────────────────────────────

    st.markdown('<div class="section-header">🌀 Vortex de Marché — Espace des Phases</div>',
                unsafe_allow_html=True)

    st.plotly_chart(
        plot_market_vortex(
            point_cloud, dates, tss, window_centers,
            animate=params.get("animate", True),
            frame_step=max(10, len(point_cloud) // 50),
        ),
        use_container_width=True,
    )

    # ─────────────────────────────────────────────────────────────────────
    # TSS + BARCODE (côte à côte)
    # ─────────────────────────────────────────────────────────────────────

    col_tss, col_bc = st.columns([0.58, 0.42])

    with col_tss:
        st.markdown('<div class="section-header">📈 TSS — Série Temporelle & Robustesse</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            plot_tss_with_sensitivity(
                tss, dates, window_centers,
                sensitivity=sensitivity,
                components_df=components_df,
            ),
            use_container_width=True,
        )

    with col_bc:
        st.markdown('<div class="section-header">🔬 Radar Topologique — Dernière Fenêtre</div>',
                    unsafe_allow_html=True)
        if diagrams_list:
            last_diag = diagrams_list[-1]
            last_desc = descriptors[-1] if descriptors else None
            entropy_val = last_desc.entropy   if last_desc else 0.0
            threshold_v = last_desc.threshold if last_desc else 0.0

            st.plotly_chart(
                plot_topological_radar(
                    last_diag,
                    entropy=entropy_val,
                    threshold=threshold_v,
                    title="Radar de Risque Topologique — H₀ & H₁",
                ),
                use_container_width=True,
            )

    # ─────────────────────────────────────────────────────────────────────
    # MULTI-ÉCHELLES TDA
    # ─────────────────────────────────────────────────────────────────────

    if ms_result is not None:
        st.markdown(
            '<div class="section-header">⚡ Multi-Échelles TDA — Divergence Topologique & Prédiction</div>',
            unsafe_allow_html=True,
        )

        # Métriques Multi-Échelles
        cur_delta  = float(ms_result.delta_tss.iloc[-1])
        cur_ptss   = float(ms_result.ptss.iloc[-1])
        cur_ew     = bool(ms_result.early_warning.iloc[-1])
        pct_ew     = float(ms_result.early_warning.mean() * 100)
        ptss_max   = float(ms_result.ptss.max())

        delta_color = "#ff6d00" if cur_delta > 0.10 else "#00e676" if cur_delta < -0.05 else "#ffc107"
        ptss_color  = "#ff1744" if cur_ptss > 0.60 else "#ffc107" if cur_ptss > 0.30 else "#00e676"
        ew_color    = "#d50000" if cur_ew else "#00e676"
        ew_txt      = "⚡ ACTIF" if cur_ew else "✅ Normal"

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card" style="--accent:{delta_color}">
            <div class="metric-label">ΔTSS Actuel</div>
            <div class="metric-value" style="color:{delta_color}">{cur_delta:+.3f}</div>
            <div class="metric-delta" style="color:#78909c">
              {'↑ Fracture en cours' if cur_delta > 0.10 else '↘ Convergence' if cur_delta < -0.05 else '→ Équilibré'}
            </div>
          </div>
          <div class="metric-card" style="--accent:{ptss_color}">
            <div class="metric-label">P-TSS (Prédictif)</div>
            <div class="metric-value" style="color:{ptss_color}">{cur_ptss:.3f}</div>
            <div class="metric-delta" style="color:#78909c">Max historique : {ptss_max:.3f}</div>
          </div>
          <div class="metric-card" style="--accent:{ew_color}">
            <div class="metric-label">Early Warning</div>
            <div class="metric-value" style="font-size:1.3rem;color:{ew_color}">{ew_txt}</div>
            <div class="metric-delta" style="color:#78909c">{pct_ew:.1f}% du temps actif</div>
          </div>
          <div class="metric-card" style="--accent:#7c4dff">
            <div class="metric-label">Divergence Scale</div>
            <div class="metric-value" style="font-size:1.2rem;color:#7c4dff">
              W{ms_result.config.get('W_FAST',20)}·W{ms_result.config.get('W_SLOW',100)}
            </div>
            <div class="metric-delta" style="color:#78909c">
              k={ms_result.config.get('EW_ACCEL_K',2.0)}σ · {len(ms_result.diagrams_fast)} / {len(ms_result.diagrams_slow)} fenêtres
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Graphique Multi-Échelles (3 panneaux)
        st.plotly_chart(
            plot_multiscale_tss(
                ms_result.tss_fast,
                ms_result.tss_slow,
                ms_result.delta_tss,
                ms_result.ptss,
                ms_result.early_warning,
                ms_result.config,
            ),
            use_container_width=True,
        )

        # Lead Time Multi-Échelles (comparaison des 3 stratégies)
        if ms_bt_results is not None:
            st.markdown(
                '<div class="section-header">⏱ Lead Time Multi-Échelles — P-TSS vs ΔTSS vs TSS Classique</div>',
                unsafe_allow_html=True,
            )

            # Tableau comparatif Lead Times
            _crises_names = list(ms_bt_results["ptss"].lead_times.keys())
            lt_tss_classic = (
                bt_result.lead_times if bt_result is not None else {}
            )

            lt_rows = []
            for cr in _crises_names:
                lt_classic = lt_tss_classic.get(cr)
                lt_ptss    = ms_bt_results["ptss"].lead_times.get(cr)
                lt_delta   = ms_bt_results["delta_tss"].lead_times.get(cr)
                lt_comb    = ms_bt_results["combined"].lead_times.get(cr)

                def _fmt_lt(v):
                    if v is None: return "N/A"
                    return f"+{v:.0f}j ✅" if v > 0 else f"{v:.0f}j ⚠"

                lt_rows.append({
                    "Crise": cr,
                    "TSS Classique": _fmt_lt(lt_classic),
                    "P-TSS": _fmt_lt(lt_ptss),
                    "ΔTSS+": _fmt_lt(lt_delta),
                    "Combined": _fmt_lt(lt_comb),
                })

            if lt_rows:
                st.dataframe(
                    pd.DataFrame(lt_rows),
                    use_container_width=True, hide_index=True,
                )

            # Résumé KPIs backtest P-TSS
            ptss_kpis = ms_bt_results["ptss"].kpis_tda
            comb_kpis = ms_bt_results["combined"].kpis_tda
            avg_lt_ptss = ptss_kpis.avg_lead_time_days
            avg_lt_comb = comb_kpis.avg_lead_time_days
            avg_lt_classic = bt_result.kpis_tda.avg_lead_time_days if bt_result else 0.0

            best_lt = max(avg_lt_ptss, avg_lt_comb, avg_lt_classic)
            if best_lt > 0:
                if avg_lt_ptss >= avg_lt_comb and avg_lt_ptss >= avg_lt_classic:
                    best_strategy = "P-TSS"
                elif avg_lt_comb >= avg_lt_classic:
                    best_strategy = "Combined P-TSS+EW"
                else:
                    best_strategy = "TSS Classique"

                improvement = max(0.0, best_lt - avg_lt_classic)
                st.markdown(f"""
                <div class="alert {'alert-safe' if improvement > 5 else 'alert-warn' if improvement > 0 else 'alert-info'}">
                  <b>🎯 Meilleure stratégie : {best_strategy}</b><br>
                  Lead Time moyen P-TSS : <b>{avg_lt_ptss:.0f}j</b> · Combined : <b>{avg_lt_comb:.0f}j</b> · Classique : <b>{avg_lt_classic:.0f}j</b><br>
                  {'Gain de <b>' + f'{improvement:.0f}j</b> vs TSS classique → anticipation améliorée.' if improvement > 0 else 'Performances similaires — données insuffisantes ou crises hors période.'}
                </div>
                """, unsafe_allow_html=True)

            # ── P-TSS Capital Protection (v2.4) ──────────────────────────
            ptss_bt      = ms_bt_results["ptss"]
            ptss_kpis_v  = ptss_bt.kpis_tda
            bnh_kpis_v   = ptss_bt.kpis_bnh

            st.markdown(
                '<div class="section-header">🛡️ Protection du Capital — P-TSS Asymétrique (v2.4)</div>',
                unsafe_allow_html=True,
            )

            # Precompute CSS classes and deltas
            def _kb(v_tda, v_bnh, hi=True):
                return "kpi-better" if (v_tda > v_bnh if hi else v_tda < v_bnh) else "kpi-worse"

            dd_red = ptss_kpis_v.max_drawdown - bnh_kpis_v.max_drawdown
            pi_red = ptss_kpis_v.pain_index   - bnh_kpis_v.pain_index

            st.markdown(f"""
            <table class="kpi-table">
              <thead><tr>
                <th>Métrique</th><th>P-TSS Asym.</th><th>Buy &amp; Hold</th><th>Protection</th>
              </tr></thead>
              <tbody>
                <tr>
                  <td>Max Drawdown</td>
                  <td class="{_kb(ptss_kpis_v.max_drawdown, bnh_kpis_v.max_drawdown, False)}">{ptss_kpis_v.max_drawdown:+.2%}</td>
                  <td>{bnh_kpis_v.max_drawdown:+.2%}</td>
                  <td class="{'kpi-better' if dd_red > 0.05 else 'kpi-worse'}">{-dd_red:+.2%} protégé</td>
                </tr>
                <tr>
                  <td>Pain Index</td>
                  <td class="{_kb(ptss_kpis_v.pain_index, bnh_kpis_v.pain_index, False)}">{ptss_kpis_v.pain_index:.3f}</td>
                  <td>{bnh_kpis_v.pain_index:.3f}</td>
                  <td class="{'kpi-better' if pi_red < 0 else 'kpi-worse'}">{abs(pi_red):.3f} {'↓ réduit' if pi_red < 0 else '↑ augmenté'}</td>
                </tr>
                <tr>
                  <td>CAGR</td>
                  <td class="{_kb(ptss_kpis_v.cagr, bnh_kpis_v.cagr)}">{ptss_kpis_v.cagr:+.2%}</td>
                  <td>{bnh_kpis_v.cagr:+.2%}</td>
                  <td class="{_kb(ptss_kpis_v.cagr, bnh_kpis_v.cagr)}">{ptss_kpis_v.cagr - bnh_kpis_v.cagr:+.2%}</td>
                </tr>
                <tr>
                  <td>Sharpe Ratio</td>
                  <td class="{_kb(ptss_kpis_v.sharpe, bnh_kpis_v.sharpe)}">{ptss_kpis_v.sharpe:.2f}</td>
                  <td>{bnh_kpis_v.sharpe:.2f}</td>
                  <td class="{_kb(ptss_kpis_v.sharpe, bnh_kpis_v.sharpe)}">{ptss_kpis_v.sharpe - bnh_kpis_v.sharpe:+.2f}</td>
                </tr>
                <tr>
                  <td>Calmar Ratio</td>
                  <td class="{_kb(ptss_kpis_v.calmar, bnh_kpis_v.calmar)}">{ptss_kpis_v.calmar:.2f}</td>
                  <td>{bnh_kpis_v.calmar:.2f}</td>
                  <td class="{_kb(ptss_kpis_v.calmar, bnh_kpis_v.calmar)}">{ptss_kpis_v.calmar - bnh_kpis_v.calmar:+.2f}</td>
                </tr>
                <tr>
                  <td>Lead Time Moyen</td>
                  <td class="kpi-better">{ptss_kpis_v.avg_lead_time_days:.0f} jours</td>
                  <td>—</td>
                  <td>Anticipation prédictive</td>
                </tr>
                <tr>
                  <td>Capital Sauvé Total</td>
                  <td class="{'kpi-better' if ptss_kpis_v.capital_saved > 0 else 'kpi-worse'}">{ptss_kpis_v.capital_saved:+.2%}</td>
                  <td>—</td>
                  <td>Σ crises vs B&amp;H</td>
                </tr>
              </tbody>
            </table>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Capital Sauvé par Crise ───────────────────────────────────
            import math as _math
            cs_by_crisis = getattr(ptss_bt, "capital_saved_by_crisis", {})
            if cs_by_crisis:
                st.markdown(
                    '<div class="section-header">💰 Capital Sauvé par Crise</div>',
                    unsafe_allow_html=True,
                )
                cs_cols = st.columns(max(1, len(cs_by_crisis)))
                for col_cs, (crisis_name, cs_val) in zip(cs_cols, cs_by_crisis.items()):
                    _is_nan = (cs_val is None) or (
                        isinstance(cs_val, float) and _math.isnan(cs_val)
                    )
                    with col_cs:
                        if _is_nan:
                            cs_color = "#37474f"
                            cs_str   = "N/A"
                            cs_sub   = "Hors période"
                        elif cs_val > 0.05:
                            cs_color = "#00e676"
                            cs_str   = f"+{cs_val:.1%}"
                            cs_sub   = "✅ Capital protégé"
                        elif cs_val > 0:
                            cs_color = "#ffc107"
                            cs_str   = f"+{cs_val:.1%}"
                            cs_sub   = "⚡ Protection partielle"
                        else:
                            cs_color = "#ff5252"
                            cs_str   = f"{cs_val:.1%}"
                            cs_sub   = "⚠ Non protégé"
                        st.markdown(f"""
                        <div class="metric-card" style="--accent:{cs_color}">
                          <div class="metric-label">{crisis_name}</div>
                          <div class="metric-value" style="color:{cs_color}">{cs_str}</div>
                          <div class="metric-delta" style="color:#78909c">{cs_sub}</div>
                        </div>""", unsafe_allow_html=True)

            # ── Journal des Trades P-TSS (enrichi) ───────────────────────
            ptss_log = ptss_bt.trade_log
            if not ptss_log.empty:
                _verdict_col = "verdict" in ptss_log.columns
                predictif_count = (
                    int((ptss_log["verdict"] == "✅ Prédictif").sum())
                    if _verdict_col else 0
                )
                total_refuge = (
                    int((ptss_log["direction"] == "REFUGE").sum())
                    if "direction" in ptss_log.columns else len(ptss_log)
                )
                exp_title = (
                    f"📋 Journal des Trades P-TSS ({len(ptss_log)} rotations"
                    + (
                        f" · ✅ {predictif_count}/{total_refuge} prédictifs ≥15j"
                        if _verdict_col else ""
                    )
                    + ")"
                )
                with st.expander(exp_title, expanded=False):
                    styled_log = ptss_log.style
                    if "direction" in ptss_log.columns:
                        styled_log = styled_log.map(
                            lambda v: (
                                "color:#ff6d00;font-weight:bold" if v == "REFUGE"
                                else "color:#00b0ff;font-weight:bold"
                            ),
                            subset=["direction"],
                        )
                    if "signal_type" in ptss_log.columns:
                        styled_log = styled_log.map(
                            lambda v: (
                                "color:#d50000;font-weight:bold" if "EW" in str(v)
                                else "color:#ff6d00;font-weight:bold" if "P-TSS" in str(v)
                                else "color:#00b0ff" if any(
                                    x in str(v) for x in ["REENTRY", "RETOUR", "RETURN"]
                                )
                                else "color:#78909c"
                            ),
                            subset=["signal_type"],
                        )
                    if _verdict_col:
                        styled_log = styled_log.map(
                            lambda v: (
                                "color:#00e676;font-weight:bold" if v == "✅ Prédictif"
                                else "color:#ffc107" if "Réactif" in str(v)
                                else ""
                            ),
                            subset=["verdict"],
                        )
                    st.dataframe(styled_log, use_container_width=True)

            # ── Rapport Complet P-TSS ─────────────────────────────────────
            with st.expander("📄 Rapport Capital Protection P-TSS", expanded=False):
                from backtester import generate_strategy_report as _gsr_ptss
                st.markdown(_gsr_ptss(ptss_bt))

    # ─────────────────────────────────────────────────────────────────────
    # PRIX + RISK OVERLAY
    # ─────────────────────────────────────────────────────────────────────

    st.markdown('<div class="section-header">📊 Prix & Zones d\'Exclusion de Risque</div>',
                unsafe_allow_html=True)

    ticker_display = params.get("ticker_display", tickers_sel[0])
    signals_for_overlay = bt_result.signals if bt_result is not None else None

    if ms_result is not None:
        # Version enrichie : TSS_fast/slow superposés + P-TSS + zones EW
        ms_signals = (
            ms_bt_results["ptss"].signals
            if ms_bt_results is not None else None
        )
        st.plotly_chart(
            plot_multiscale_overlay(
                prices,
                ms_result.tss_fast,
                ms_result.tss_slow,
                ms_result.ptss,
                ms_result.early_warning,
                ticker=ticker_display,
                signals=ms_signals,
                config=ms_result.config,
            ),
            use_container_width=True,
        )
    else:
        st.plotly_chart(
            plot_prices_risk_overlay(
                prices, tss, dates, window_centers,
                ticker=ticker_display,
                signals=signals_for_overlay,
            ),
            use_container_width=True,
        )

    # ─────────────────────────────────────────────────────────────────────
    # BACKTEST
    # ─────────────────────────────────────────────────────────────────────

    if bt_result is not None:
        st.markdown('<div class="section-header">💼 Backtest — TDA Strategy vs Buy & Hold</div>',
                    unsafe_allow_html=True)

        st.plotly_chart(
            plot_backtest_dashboard(bt_result),
            use_container_width=True,
        )

        # KPIs comparatifs
        tda = bt_result.kpis_tda
        bnh = bt_result.kpis_bnh

        def _fmt_pct(v): return f"{v:+.2%}"
        def _fmt_f2(v):  return f"{v:.2f}"
        def _cls(v_tda, v_bnh, higher_better=True):
            better = v_tda > v_bnh if higher_better else v_tda < v_bnh
            return "kpi-better" if better else "kpi-worse"

        st.markdown(f"""
        <table class="kpi-table">
          <thead>
            <tr>
              <th>Métrique</th>
              <th>TDA Strategy</th>
              <th>Buy & Hold</th>
              <th>Avantage</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>CAGR</td>
              <td class="{_cls(tda.cagr, bnh.cagr)}">{_fmt_pct(tda.cagr)}</td>
              <td>{_fmt_pct(bnh.cagr)}</td>
              <td class="{_cls(tda.cagr-bnh.cagr,0)}">{_fmt_pct(tda.cagr-bnh.cagr)}</td>
            </tr>
            <tr>
              <td>Sharpe Ratio</td>
              <td class="{_cls(tda.sharpe, bnh.sharpe)}">{_fmt_f2(tda.sharpe)}</td>
              <td>{_fmt_f2(bnh.sharpe)}</td>
              <td class="{_cls(tda.sharpe-bnh.sharpe,0)}">{_fmt_f2(tda.sharpe-bnh.sharpe)}</td>
            </tr>
            <tr>
              <td>Max Drawdown</td>
              <td class="{_cls(tda.max_drawdown, bnh.max_drawdown, higher_better=False)}">{_fmt_pct(tda.max_drawdown)}</td>
              <td>{_fmt_pct(bnh.max_drawdown)}</td>
              <td class="{_cls(bnh.max_drawdown-tda.max_drawdown,0)}">{_fmt_pct(bnh.max_drawdown-tda.max_drawdown)}</td>
            </tr>
            <tr>
              <td>Calmar Ratio</td>
              <td class="{_cls(tda.calmar, bnh.calmar)}">{_fmt_f2(tda.calmar)}</td>
              <td>{_fmt_f2(bnh.calmar)}</td>
              <td class="{_cls(tda.calmar-bnh.calmar,0)}">{_fmt_f2(tda.calmar-bnh.calmar)}</td>
            </tr>
            <tr>
              <td>Volatilité Ann.</td>
              <td class="{_cls(tda.volatility_ann, bnh.volatility_ann, higher_better=False)}">{_fmt_pct(tda.volatility_ann)}</td>
              <td>{_fmt_pct(bnh.volatility_ann)}</td>
              <td>—</td>
            </tr>
            <tr>
              <td>Rendement Total</td>
              <td class="{_cls(tda.total_return, bnh.total_return)}">{_fmt_pct(tda.total_return)}</td>
              <td>{_fmt_pct(bnh.total_return)}</td>
              <td>—</td>
            </tr>
            <tr>
              <td>Rotations (Trades)</td>
              <td>{tda.n_trades}</td>
              <td>0</td>
              <td style="color:#78909c">Low turnover</td>
            </tr>
            <tr>
              <td>Hit Rate</td>
              <td>{tda.hit_rate:.1%}</td>
              <td>—</td>
              <td>—</td>
            </tr>
            <tr>
              <td>Lead Time Moyen</td>
              <td class="kpi-better">{tda.avg_lead_time_days:.0f} jours</td>
              <td>—</td>
              <td>Avance sur les crises</td>
            </tr>
          </tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Lead Times détaillés
        st.markdown('<div class="section-header">⏱ Lead Time — Avance sur les Crises</div>',
                    unsafe_allow_html=True)
        lt_cols = st.columns(len(bt_result.lead_times))
        for col, (crisis, days) in zip(lt_cols, bt_result.lead_times.items()):
            with col:
                if days is None:
                    st.markdown(f"""
                    <div class="metric-card" style="--accent:#37474f">
                      <div class="metric-label">{crisis}</div>
                      <div class="metric-value" style="font-size:1rem;color:#455a64">N/A</div>
                      <div class="metric-delta" style="color:#37474f">Hors période</div>
                    </div>""", unsafe_allow_html=True)
                elif days > 0:
                    st.markdown(f"""
                    <div class="metric-card" style="--accent:#00e676">
                      <div class="metric-label">{crisis}</div>
                      <div class="metric-value" style="color:#00e676">{days:.0f}j</div>
                      <div class="metric-delta" style="color:#00e676">✅ En avance</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card" style="--accent:#ff5252">
                      <div class="metric-label">{crisis}</div>
                      <div class="metric-value" style="color:#ff5252">{abs(days):.0f}j</div>
                      <div class="metric-delta" style="color:#ff5252">⚠ En retard</div>
                    </div>""", unsafe_allow_html=True)

        # Rapport textuel
        with st.expander("📄 Rapport Complet (Markdown)", expanded=False):
            report = generate_strategy_report(bt_result)
            st.markdown(report)

        # Trade Log
        if not bt_result.trade_log.empty:
            with st.expander(f"📋 Journal des Trades ({len(bt_result.trade_log)} rotations)", expanded=False):
                st.dataframe(
                    bt_result.trade_log.style.map(
                        lambda v: "color:#ff6d00;font-weight:bold" if v == "REFUGE"
                                  else "color:#00b0ff;font-weight:bold",
                        subset=["direction"],
                    ),
                    use_container_width=True,
                )

    # ─────────────────────────────────────────────────────────────────────
    # SECTION EXPERT
    # ─────────────────────────────────────────────────────────────────────

    if params.get("expert", True):
        st.markdown('<div class="section-header">🧬 Section Expert — Analyse Avancée</div>',
                    unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧮 Composantes TSS",
            "📐 Statistiques Topologiques",
            "🔍 Sensibilité (W, τ)",
            "✅ Tests Unitaires",
            "🔬 Stabilité Dimensionnelle",
        ])

        with tab1:
            if len(components_df) > 0:
                # Graphique des composantes
                import plotly.express as px
                centers_dates = [dates[min(wc, len(dates) - 1)] for wc in components_df.index]
                comp_plot = components_df.copy()
                comp_plot.index = pd.DatetimeIndex(centers_dates)

                fig_comp = px.line(
                    comp_plot[["wasserstein_norm", "amplitude_norm", "entropy_collapse", "tss_final"]],
                    template="plotly_dark",
                    color_discrete_map={
                        "wasserstein_norm":   "#00b0ff",
                        "amplitude_norm":     "#7c4dff",
                        "entropy_collapse":   "#ff1744",
                        "tss_final":          "#ffc107",
                    },
                    labels={"value": "Score normalisé", "variable": "Composante"},
                    title="Décomposition temporelle du TSS",
                )
                fig_comp.update_layout(
                    paper_bgcolor="#080c14", plot_bgcolor="#0d1421",
                    height=300, margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # Table des dernières valeurs
                last_vals = components_df.iloc[-1]
                st.markdown(f"""
                **Valeurs courantes :**
                - Wasserstein W₁ (normalisé) : `{last_vals.get('wasserstein_norm', 0):.4f}`
                - Amplitude H₁ (normalisée)  : `{last_vals.get('amplitude_norm', 0):.4f}`
                - Collapse Entropie          : `{last_vals.get('entropy_collapse', 0):.4f}`
                - Entropie brute H₁          : `{last_vals.get('entropy_raw', 0):.4f}` bits
                - Seuil θ (IQR)              : `{last_vals.get('threshold', 0):.4f}`
                - Cycles H₁ stables          : `{int(last_vals.get('n_h1', 0))}`
                """)

        with tab2:
            tss_arr = tss.values
            pct_stress  = float((tss_arr > 0.7).mean() * 100)
            pct_transit = float(((tss_arr > 0.4) & (tss_arr <= 0.7)).mean() * 100)
            pct_calm    = float((tss_arr <= 0.4).mean() * 100)

            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("% Régime Stress",      f"{pct_stress:.1f}%")
            col_s2.metric("% Régime Transition",   f"{pct_transit:.1f}%")
            col_s3.metric("% Régime Calme",        f"{pct_calm:.1f}%")

            # Distribution TSS
            fig_dist = px.histogram(
                x=tss_arr, nbins=50,
                labels={"x": "TSS", "y": "Fréquence"},
                title="Distribution du TSS",
                color_discrete_sequence=["#00b0ff"],
                template="plotly_dark",
                opacity=0.8,
            )
            for thr, col_name, lbl in [
                (0.7, "#ff1744", "Stress"),
                (0.4, "#ffc107", "Transition"),
            ]:
                fig_dist.add_vline(
                    x=thr, line_dash="dash",
                    line_color=col_name, line_width=1.5,
                    annotation_text=lbl,
                    annotation_font=dict(color=col_name),
                )
            fig_dist.update_layout(
                paper_bgcolor="#080c14", plot_bgcolor="#0d1421",
                height=250, margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Stats descriptives
            stats_df = pd.DataFrame({
                "Statistique": ["Moyenne", "Médiane", "Écart-type", "Asymétrie", "Kurtosis",
                                 "Min", "Max", "Q25", "Q75"],
                "Valeur": [
                    f"{np.mean(tss_arr):.4f}",
                    f"{np.median(tss_arr):.4f}",
                    f"{np.std(tss_arr):.4f}",
                    f"{float(pd.Series(tss_arr).skew()):.4f}",
                    f"{float(pd.Series(tss_arr).kurtosis()):.4f}",
                    f"{tss_arr.min():.4f}",
                    f"{tss_arr.max():.4f}",
                    f"{np.percentile(tss_arr, 25):.4f}",
                    f"{np.percentile(tss_arr, 75):.4f}",
                ],
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        with tab3:
            if sensitivity is not None and len(sensitivity.tss_mean) > 0:
                st.success("✅ Analyse de sensibilité disponible")

                n_configs = len(sensitivity.tss_matrix)
                mean_std  = float(sensitivity.tss_std.mean())
                max_spread = float((sensitivity.tss_upper - sensitivity.tss_lower).max())

                col_sa1, col_sa2, col_sa3 = st.columns(3)
                col_sa1.metric("Configurations testées", n_configs)
                col_sa2.metric("Std TSS moyen",   f"{mean_std:.4f}",
                                help="Faible → signal robuste")
                col_sa3.metric("Spread max (U-L)", f"{max_spread:.4f}")

                if mean_std < 0.05:
                    st.markdown("""<div class="alert alert-safe">
                    ✅ Signal TSS <b>hautement robuste</b> — variance faible entre les configurations.
                    </div>""", unsafe_allow_html=True)
                elif mean_std < 0.12:
                    st.markdown("""<div class="alert alert-warn">
                    ⚡ Robustesse <b>modérée</b> — quelques variations selon les paramètres.
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="alert alert-danger">
                    ⚠️ Signal <b>sensible aux hyperparamètres</b> — interpréter avec prudence.
                    </div>""", unsafe_allow_html=True)

                # Tableau des configurations
                cfg_data = []
                for (W, tau), series in sensitivity.tss_matrix.items():
                    cfg_data.append({
                        "W (fenêtre)": W,
                        "τ (délai)": tau,
                        "TSS moyen": f"{series.mean():.4f}",
                        "TSS max": f"{series.max():.4f}",
                        "TSS std": f"{series.std():.4f}",
                    })
                st.dataframe(pd.DataFrame(cfg_data), use_container_width=True, hide_index=True)
            else:
                st.info("Activez 'Analyse de sensibilité' dans la sidebar pour voir les résultats.")

        with tab4:
            # Tests unitaires sur le moteur TDA
            st.markdown("**Tests unitaires des calculs d'entropie et du moteur TDA**")

            test_results = {}

            # Test 1 : Entropie nulle pour 1 feature
            try:
                from tda_engine import compute_persistence_entropy
                dummy_diag = np.array([[0.0, 1.0, 1]])  # 1 feature H1
                ent = compute_persistence_entropy(dummy_diag, dim=1)
                passed = ent == 0.0
                test_results["Entropie nulle (1 feature)"] = (
                    "✅ PASS" if passed else f"❌ FAIL — got {ent:.4f}",
                    passed,
                )
            except Exception as e:
                test_results["Entropie nulle (1 feature)"] = (f"❌ ERROR: {e}", False)

            # Test 2 : Entropie maximale (2 features équiprobables)
            try:
                diag_eq = np.array([[0.0, 1.0, 1], [0.0, 1.0, 1]])
                ent_max = compute_persistence_entropy(diag_eq, dim=1)
                passed = abs(ent_max - 1.0) < 1e-6  # log2(2) = 1 bit
                test_results["Entropie max (2 features équiprobables)"] = (
                    f"✅ PASS (E={ent_max:.4f})" if passed else f"❌ FAIL — got {ent_max:.4f}",
                    passed,
                )
            except Exception as e:
                test_results["Entropie max (2 features équiprobables)"] = (f"❌ ERROR: {e}", False)

            # Test 3 : Seuil dynamique IQR
            try:
                from tda_engine import dynamic_persistence_threshold
                diag_iqr = np.array([
                    [0, 0.1, 1], [0, 0.1, 1], [0, 0.1, 1],  # bruit
                    [0, 2.0, 1],  # signal
                ])
                theta = dynamic_persistence_threshold(diag_iqr, dim=1, method="iqr")
                passed = theta > 0.0
                test_results["Seuil IQR > 0"] = (
                    f"✅ PASS (θ={theta:.4f})" if passed else "❌ FAIL",
                    passed,
                )
            except Exception as e:
                test_results["Seuil IQR > 0"] = (f"❌ ERROR: {e}", False)

            # Test 4 : Wasserstein identique
            # Tolérance 1e-5 (pas 1e-9) : persim retourne un float64 avec
            # une erreur numérique ~1e-7 due à l'algorithme d'assignation
            # optimale (Hungarian matching). 0.0000 affiché ≠ 0.0 exact.
            try:
                from tda_engine import wasserstein_distance_h1
                d = np.array([[0, 0.5, 1], [0.1, 0.8, 1]])
                w = wasserstein_distance_h1(d, d)
                passed = abs(w) < 1e-5
                test_results["Wasserstein(D,D) = 0"] = (
                    f"✅ PASS (W={w:.2e})" if passed else f"❌ FAIL — got {w:.2e}",
                    passed,
                )
            except Exception as e:
                test_results["Wasserstein(D,D) = 0"] = (f"❌ ERROR: {e}", False)

            # Test 5 : Z-score moyenne ≈ 0
            try:
                from tda_engine import compute_log_returns, rolling_zscore
                prices_test = pd.DataFrame(
                    {"A": np.exp(np.cumsum(np.random.randn(200) * 0.01))},
                    index=pd.bdate_range("2020-01-01", periods=200),
                )
                ret_test = compute_log_returns(prices_test)
                z_test   = rolling_zscore(ret_test, window=60)
                mean_abs = abs(z_test.mean().mean())
                passed   = mean_abs < 0.5
                test_results["Z-score moyenne ≈ 0"] = (
                    f"✅ PASS (|μ|={mean_abs:.4f})" if passed else f"❌ FAIL",
                    passed,
                )
            except Exception as e:
                test_results["Z-score moyenne ≈ 0"] = (f"❌ ERROR: {e}", False)

            # Affichage des résultats
            passed_count = sum(1 for _, (_, p) in test_results.items() if p)
            total_count  = len(test_results)

            overall_color = "#00e676" if passed_count == total_count else \
                            "#ffc107" if passed_count >= total_count * 0.7 else "#ff1744"

            st.markdown(f"""
            <div style='font-family:JetBrains Mono,monospace;font-size:0.82rem;
                        color:{overall_color};margin-bottom:12px'>
              Résultat global : {passed_count}/{total_count} tests passés
            </div>
            """, unsafe_allow_html=True)

            for test_name, (result_str, passed) in test_results.items():
                color = "#00e676" if passed else "#ff5252"
                st.markdown(f"""
                <div style='display:flex;gap:12px;padding:6px 10px;
                            background:#0d1421;border-radius:4px;
                            margin-bottom:4px;font-family:JetBrains Mono,monospace;font-size:0.8rem'>
                  <span style='color:{color};min-width:90px'>{result_str.split(' ')[0]}</span>
                  <span style='color:#78909c'>{test_name}</span>
                  <span style='color:#455a64;margin-left:auto'>{result_str.split(' ',1)[1] if ' ' in result_str else ''}</span>
                </div>
                """, unsafe_allow_html=True)

            # Sauvegarder le résultat des tests
            try:
                tests_json = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "passed": passed_count,
                    "total":  total_count,
                    "tests": {k: {"result": v[0], "passed": v[1]} for k, v in test_results.items()},
                }
                with open("tests.json", "w", encoding="utf-8") as f:
                    json.dump(tests_json, f, indent=2, ensure_ascii=False)
                st.caption("Tests sauvegardés dans `tests.json`")
            except Exception:
                pass

        with tab5:
            # ── Stabilité Dimensionnelle ──────────────────────────────────
            st.markdown("""
            **Objectif :** Prouver que le TSS est une réalité topologique, pas un artefact
            du choix de la dimension d'embedding de Takens.

            Si le TSS moyen reste **stable** (Δ < 0.1 entre dimensions), le signal
            est robuste. Si l'écart-type **inter-dimensions** dépasse 0.1, le modèle
            est sensible au choix de d → interpréter avec prudence.
            """)

            # Sélection des dimensions à tester
            col_dim_sel, col_dim_run = st.columns([0.7, 0.3])
            with col_dim_sel:
                dims_to_test = st.multiselect(
                    "Dimensions d à tester",
                    options=[2, 3, 4, 5],
                    default=[2, 3, 4, 5],
                    help="d=5 peut être lent (~60-120s) — une barre de progression s'affiche.",
                )
                if not dims_to_test:
                    st.warning("Sélectionnez au moins une dimension.")

            with col_dim_run:
                st.markdown("<br>", unsafe_allow_html=True)
                run_stability = st.button(
                    "🔍 Run Sensitivity Check",
                    type="primary",
                    use_container_width=True,
                    disabled=not dims_to_test,
                )

            if run_stability and dims_to_test:
                from tda_engine import run_tda_pipeline as _rp

                if 5 in dims_to_test:
                    st.markdown(
                        '<div class="alert alert-warn">⚡ Dimension 5 incluse — '
                        'calcul plus long (~60-120s). Barre de progression active.</div>',
                        unsafe_allow_html=True,
                    )

                # Barre de progression par dimension (séquentiel : Streamlit
                # n'est pas thread-safe, on ne peut pas mettre à jour st.progress
                # depuis un worker joblib background).
                prog_bar = st.progress(0, text="Initialisation...")
                stability_rows = []
                n_dims = len(dims_to_test)

                for i, d in enumerate(sorted(dims_to_test)):
                    prog_bar.progress(
                        int(i / n_dims * 100),
                        text=f"Calcul dimension d={d} ({i+1}/{n_dims})…"
                        + (" ⏳ patience" if d >= 5 else ""),
                    )
                    t0 = time.perf_counter()
                    try:
                        tss_d, _, _, _, _ = _rp(
                            prices,
                            window=window_size,
                            step=step_size,
                            embed_dim=d,
                            embed_delay=embed_delay,
                            threshold_method=thresh_meth,
                            tss_weights=tss_weights,
                        )
                    except Exception as e_d:
                        stability_rows.append({
                            "Dimension d": d,
                            "TSS Moyen":  np.nan,
                            "Écart-type": np.nan,
                            "TSS Min":    np.nan,
                            "TSS Max":    np.nan,
                            "Temps (s)":  round(time.perf_counter() - t0, 1),
                            "Statut":     f"❌ {str(e_d)[:60]}",
                        })
                        continue

                    elapsed_d = time.perf_counter() - t0
                    tss_vals = tss_d.values
                    stability_rows.append({
                        "Dimension d": d,
                        "TSS Moyen":  round(float(np.mean(tss_vals)),  4),
                        "Écart-type": round(float(np.std(tss_vals)),   4),
                        "TSS Min":    round(float(np.min(tss_vals)),   4),
                        "TSS Max":    round(float(np.max(tss_vals)),   4),
                        "Temps (s)":  round(elapsed_d, 1),
                        "Statut":     "✅ OK",
                    })

                prog_bar.progress(100, text="✓ Analyse complète")
                time.sleep(0.3)
                prog_bar.empty()

                stab_df = pd.DataFrame(stability_rows)
                st.session_state["stability_df"] = stab_df

            # Affichage des résultats (persistants entre les reruns)
            stab_df = st.session_state.get("stability_df")
            if stab_df is not None and len(stab_df) > 0:

                st.markdown("#### Résultats par Dimension d'Embedding")
                st.dataframe(stab_df, use_container_width=True, hide_index=True)

                # Interprétation automatique
                valid = stab_df.dropna(subset=["TSS Moyen"])
                if len(valid) >= 2:
                    mean_values  = valid["TSS Moyen"].values
                    std_of_means = float(np.std(mean_values))   # variabilité INTER-dim
                    mean_of_stds = float(valid["Écart-type"].mean())  # variabilité INTRA-dim

                    if std_of_means < 0.05:
                        verdict_cls  = "alert-safe"
                        verdict_icon = "✅"
                        verdict_main = "Modèle Stable"
                        verdict_text = (
                            f"L'écart inter-dimensions est de <b>{std_of_means:.4f}</b> (< 0.05). "
                            "Le TSS est <b>indépendant du choix de d</b> → signal topologique réel."
                        )
                    elif std_of_means < 0.10:
                        verdict_cls  = "alert-warn"
                        verdict_icon = "⚡"
                        verdict_main = "Stabilité Modérée"
                        verdict_text = (
                            f"L'écart inter-dimensions est de <b>{std_of_means:.4f}</b> (0.05–0.10). "
                            "Le signal présente une sensibilité modérée à d → utiliser d=3 (défaut)."
                        )
                    else:
                        verdict_cls  = "alert-danger"
                        verdict_icon = "⚠️"
                        verdict_main = "Sensibilité Élevée"
                        verdict_text = (
                            f"L'écart inter-dimensions est de <b>{std_of_means:.4f}</b> (> 0.10). "
                            "Le TSS varie significativement avec d → "
                            "<b>interpréter avec prudence</b>, calibrer d sur données historiques."
                        )

                    st.markdown(
                        f'<div class="alert {verdict_cls}">'
                        f'<b>{verdict_icon} {verdict_main}</b><br>'
                        f'{verdict_text}<br>'
                        f'<small>Std intra-dim moyen : {mean_of_stds:.4f}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Graphique comparatif des TSS moyens par dimension
                    import plotly.graph_objects as go
                    fig_stab = go.Figure()
                    fig_stab.add_trace(go.Bar(
                        x=[f"d={int(r)}" for r in valid["Dimension d"]],
                        y=valid["TSS Moyen"].tolist(),
                        error_y=dict(
                            type="data",
                            array=valid["Écart-type"].tolist(),
                            visible=True,
                            color="#ffc107",
                            thickness=2,
                        ),
                        marker=dict(
                            color=valid["TSS Moyen"].tolist(),
                            colorscale="RdYlGn_r",
                            cmin=0, cmax=1,
                            line=dict(color="#1a2235", width=1),
                        ),
                        text=[f"{v:.4f}" for v in valid["TSS Moyen"]],
                        textposition="outside",
                        textfont=dict(size=11, color="#e8eaf6"),
                        hovertemplate=(
                            "d=%{x}<br>TSS moyen: %{y:.4f}<br>"
                            "<extra></extra>"
                        ),
                    ))
                    fig_stab.add_hline(
                        y=float(np.mean(mean_values)),
                        line_dash="dash", line_color="#ffc107", line_width=1.5,
                        annotation_text=f"Moyenne = {np.mean(mean_values):.4f}",
                        annotation_font=dict(color="#ffc107", size=10),
                        annotation_position="right",
                    )
                    fig_stab.update_layout(
                        title=dict(
                            text="TSS Moyen par Dimension d'Embedding de Takens",
                            font=dict(size=13, color="#e8eaf6"),
                        ),
                        paper_bgcolor="#080c14",
                        plot_bgcolor="#0d1421",
                        xaxis=dict(
                            title="Dimension d (Takens)",
                            gridcolor="#1a2235",
                            tickfont=dict(color="#78909c"),
                        ),
                        yaxis=dict(
                            title="TSS Moyen [0–1]",
                            range=[0, 1.1],
                            gridcolor="#1a2235",
                            tickfont=dict(color="#78909c"),
                        ),
                        font=dict(color="#e8eaf6"),
                        height=300,
                        margin=dict(l=10, r=80, t=50, b=40),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_stab, use_container_width=True)

                elif len(valid) == 1:
                    st.info("Une seule dimension valide — impossible de comparer la stabilité inter-dimensions.")
                else:
                    st.error("Aucune dimension n'a produit de résultat valide.")
            else:
                st.info("Cliquez sur **🔍 Run Sensitivity Check** pour lancer l'analyse.")

    # ─────────────────────────────────────────────────────────────────────
    # DIAGNOSTIC AUTOMATIQUE
    # ─────────────────────────────────────────────────────────────────────

    st.markdown('<div class="section-header">🧠 Diagnostic Automatisé</div>',
                unsafe_allow_html=True)

    entropy_interp = (
        "diversifiée (marché résilient)" if current_entropy > 2.5 else
        "modérée (régime en transition)" if current_entropy > 1.2 else
        "effondrée (marché monolithique — FRAGILE)"
    )
    entropy_risk = (
        "faible" if current_entropy > 2.5 else
        "modéré" if current_entropy > 1.2 else
        "ÉLEVÉ"
    )

    h1_interp = (
        "complexe (multiple attracteurs)" if current_n_h1 > 5 else
        "structuré (régime distinct)" if current_n_h1 > 1 else
        "simple (attracteur unique)"
    )

    trend_interp = (
        f"en hausse rapide (+{tss_delta:.3f}) → VIGILANCE ACCRUE"
        if tss_delta > 0.05 else
        f"stable ({tss_delta:+.3f})"
        if abs(tss_delta) <= 0.05 else
        f"en baisse ({tss_delta:.3f}) → résilience croissante"
    )

    # ── Diagnostic Multi-Échelles (si disponible) ──────────────────────
    ms_diag_section = ""
    if ms_result is not None:
        cur_delta_diag = float(ms_result.delta_tss.iloc[-1])
        cur_ptss_diag  = float(ms_result.ptss.iloc[-1])
        cur_ew_diag    = bool(ms_result.early_warning.iloc[-1])

        # Convergence/divergence inter-échelles
        if cur_delta_diag > 0.15:
            ms_regime = "FRACTURE TOPOLOGIQUE"
            ms_color_cls = "alert-danger"
            ms_interp = (
                f"ΔTSS = **{cur_delta_diag:+.3f}** → L'instabilité micro (W_fast) "
                f"devance significativement la structure macro (W_slow). "
                "Ce découplage inter-échelles est un précurseur classique de régime de crise."
            )
        elif cur_delta_diag > 0.05:
            ms_regime = "DIVERGENCE MODÉRÉE"
            ms_color_cls = "alert-warn"
            ms_interp = (
                f"ΔTSS = **{cur_delta_diag:+.3f}** → Pression locale émergente "
                "non encore consolidée à l'échelle macro. Surveillance recommandée."
            )
        elif cur_delta_diag < -0.05:
            ms_regime = "CONVERGENCE"
            ms_color_cls = "alert-safe"
            ms_interp = (
                f"ΔTSS = **{cur_delta_diag:+.3f}** → Les deux échelles se resserrent. "
                "Phase de stabilisation ou de détente généralisée."
            )
        else:
            ms_regime = "ÉQUILIBRE"
            ms_color_cls = "alert-info"
            ms_interp = (
                f"ΔTSS = **{cur_delta_diag:+.3f}** → Cohérence entre micro et macro. "
                "Régime homogène, sans fracture détectable."
            )

        ew_note = (
            "⚡ **Early Warning ACTIF** — L'accélération de ΔTSS dépasse le seuil k·σ. "
            "Probabilité d'événement de marché à court terme accrue.\n\n"
            if cur_ew_diag else ""
        )

        ms_diag_section = f"""
**📡 Diagnostic Multi-Échelles** (W_fast={ms_result.config.get('W_FAST',20)}j · W_slow={ms_result.config.get('W_SLOW',100)}j)

{ew_note}**Régime inter-échelles : {ms_regime}** — {ms_interp}

**P-TSS = {cur_ptss_diag:.3f}** → {'Signal prédictif élevé — fracture non encore systémique → fenêtre d\'anticipation.' if cur_ptss_diag > 0.40 else 'Pas de fracture topologique significative détectée à cette échelle.'}
"""

    diag_text = f"""
**Analyse du régime de marché actuel** — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

**TSS = {current_tss:.3f}** | Régime : **{regime_txt}** | Tendance : {trend_interp}

**Lecture topologique (mono-échelle) :**
- L'**entropie de persistance H₁** est **{current_entropy:.2f} bits** ({entropy_interp}).
  → Risque d'effondrement corrélationnel : **{entropy_risk}**.
- **{current_n_h1} cycle(s) H₁** stables détectés dans l'espace des phases.
  → Géométrie des corrélations : {h1_interp}.
{ms_diag_section}"""

    if current_tss >= 0.80:
        diag_text += """
**⚡ SIGNAL CRITIQUE** : Le TSS dépasse le seuil d'exclusion (θ=0.80).
La topologie des corrélations présente les signatures typiques d'un précurseur de crise :
synchronisation des actifs (H₁ persistants) + effondrement de la diversité (entropie basse).
**Action recommandée** : Rotation partielle vers actif refuge selon tolérance au risque.
"""
    elif current_tss >= 0.40:
        diag_text += """
**⚡ VIGILANCE** : Structure topologique en mutation. Des cycles H₁ instables émergent,
suggérant une réorganisation des corrélations inter-actifs.
**Action recommandée** : Surveiller l'évolution du TSS. Réduire optionnellement l'exposition.
"""
    else:
        diag_text += """
**✅ RÉGIME NORMAL** : La géométrie des corrélations est stable et diversifiée.
Les features H₁ ont une faible persistance — aucune structure de crise détectée.
**Action recommandée** : Maintenir les positions selon le plan d'investissement.
"""

    st.markdown(f'<div class="alert {alert_cls}">{diag_text}</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="pro-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.65rem;color:#37474f;text-align:center;padding:8px'>
      TDA-Risk-Sentinel Pro v2.4 · ripser · persim · Multi-Échelles · Vietoris-Rips · Entropie · Wasserstein<br>
      ⚠️ Outil d'analyse quantitative — Ne constitue pas un conseil en investissement.
    </div>
    """, unsafe_allow_html=True)

else:
    # ─────────────────────────────────────────────────────────────────────
    # ÉCRAN D'ACCUEIL
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="pro-divider"></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("""
        ### 🔬 Architecture TDA v2.0

        ```
        PRIX AJUSTÉS (Yahoo Finance)
               ↓
        LOG-RENDEMENTS  r_t = ln(P_t/P_{t-1})
               ↓
        Z-SCORE GLISSANT  (fenêtre W, sans look-ahead)
               ↓
        TAKENS EMBEDDING  (dimension d, délai τ)
        → Nuage de points 3D dans l'espace des phases
               ↓
        VIETORIS-RIPS (filtration multi-échelle)
        → Diagrammes de persistance H₀, H₁
               ↓
        SEUIL IQR DYNAMIQUE  (filtrage bruit blanc)
               ↓
        DESCRIPTEURS TOPOLOGIQUES
          • Distance de Wasserstein W₁ (vitesse)
          • Amplitude H₁ max (complexité)
          • Entropie de Persistance (diversité)
               ↓
        TSS = 0.45·W₁ + 0.30·Amp + 0.25·(1-Entropie)
               ↓
        BACKTEST (Rotation refuge si TSS > θ)
        → Sharpe · MaxDD · CAGR · Lead Time
        ```
        """)

    with col_r:
        st.markdown("""
        ### 📐 Innovations Pro v2.0

        **1. Entropie de Persistance**
        > E = -Σᵢ pᵢ log₂(pᵢ)
        > Une chute d'entropie → marché monolithique → signal de crise.

        **2. Seuil Dynamique IQR**
        > θ = Q₁ + 1.5·IQR des persistances H₁
        > Filtre le bruit blanc topologique automatiquement.

        **3. Analyse de Sensibilité**
        > Tests W ∈ {W-5, W, W+5} × τ ∈ {1,2,3}
        > Enveloppe TSS ± σ pour valider la robustesse.

        **4. Vortex 3D Animé**
        > Trajectoire de l'état du marché dans l'espace des phases.
        > Animation temporelle : du calme au vortex de crise.

        **5. Backtesting Complet**
        > Stratégie de rotation systématique TSS > θ.
        > Sharpe, MaxDD, Calmar, Lead Time vs B&H.
        """)

    st.markdown('<div class="pro-divider"></div>', unsafe_allow_html=True)
    st.info("👈  Configurez les paramètres dans la barre latérale et cliquez sur **▶ ANALYSER**.")
