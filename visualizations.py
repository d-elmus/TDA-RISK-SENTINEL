"""
visualizations.py — Visualisations Pro TDA-Risk-Sentinel v2.3

Composants graphiques :
  1. Vortex de Marché 3D animé (Plotly frames) — "Wow Effect" central.
  2. Radar de Risque Topologique (Barcode amélioré avec entropie).
  3. Prix + Risk Overlay (heatmap TSS + zones d'exclusion).
  4. Courbe TSS avec enveloppe de sensibilité.
  5. Dashboard de Backtest (équités + drawdown + signaux).
  6. Composantes du TSS (waterfall / spider chart).
  7. Multi-Échelles : TSS_fast/TSS_slow + ruban Divergence + P-TSS + EW.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Finance Pro Dark Theme
# ═══════════════════════════════════════════════════════════════════════════

C = {
    # Fonds
    "bg":          "#080c14",
    "bg_card":     "#0d1421",
    "bg_panel":    "#111827",
    "grid":        "#1a2235",
    # Signaux
    "safe":        "#00e676",
    "warn":        "#ffc107",
    "danger":      "#ff1744",
    "critical":    "#d50000",
    # Accents
    "accent":      "#00b0ff",
    "accent2":     "#7c4dff",
    "accent3":     "#ff6d00",
    # Texte
    "text":        "#e8eaf6",
    "text_dim":    "#78909c",
    "text_bright": "#ffffff",
    # Colorscales nommées
    "cs_risk":     "RdYlGn_r",
    "cs_heat":     "Inferno",
}

FONT_FAMILY = "JetBrains Mono, Fira Code, Consolas, monospace"

CRISIS_PERIODS = {
    "COVID Crash": ("2020-02-20", "2020-03-23"),
    "Inflation Shock": ("2022-01-05", "2022-10-13"),
    "SVB Crisis": ("2023-03-09", "2023-03-24"),
    "Crypto Winter": ("2021-11-10", "2022-06-18"),
}


def _base_layout(**kwargs) -> dict:
    """Retourne le layout de base partagé par tous les graphiques."""
    base = dict(
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg_card"],
        font=dict(family=FONT_FAMILY, color=C["text"], size=11),
        legend=dict(
            bgcolor="rgba(8,12,20,0.8)",
            bordercolor=C["grid"],
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    base.update(kwargs)
    return base


def _axis_style(**kwargs) -> dict:
    """Style d'axe standardisé Finance Pro."""
    base = dict(
        gridcolor=C["grid"],
        gridwidth=0.5,
        showgrid=True,
        zeroline=False,
        tickfont=dict(size=10, color=C["text_dim"]),
        linecolor=C["grid"],
    )
    base.update(kwargs)
    return base


# ═══════════════════════════════════════════════════════════════════════════
# 1. VORTEX DE MARCHÉ 3D ANIMÉ
# ═══════════════════════════════════════════════════════════════════════════

def plot_market_vortex(
    point_cloud: np.ndarray,
    dates: pd.DatetimeIndex,
    tss: Optional[pd.Series] = None,
    window_centers: Optional[List[int]] = None,
    animate: bool = True,
    frame_step: int = 20,
) -> go.Figure:
    """Visualise le Vortex de Marché — nuage de points 3D animé.

    Le "vortex" représente la trajectoire de l'état du marché dans
    l'espace des phases de Takens. En régime calme, la trajectoire
    tourne autour d'un attracteur stable (anneau). En régime de crise,
    elle part en spirale vers de nouvelles régions de l'espace → "vortex".

    Animation :
      Chaque frame ajoute N points à la trajectoire, révélant la
      dynamique temporelle. La couleur encode le TSS (vert → rouge).

    Args:
        point_cloud:    Array (T, 3) du nuage embeddi.
        dates:          DatetimeIndex de longueur T.
        tss:            pd.Series TSS (optionnel).
        window_centers: Indices TSS dans point_cloud.
        animate:        Activer l'animation Plotly. Défaut = True.
        frame_step:     Points par frame d'animation. Défaut = 20.

    Returns:
        go.Figure avec animation et contrôles lecture/pause.
    """
    T = len(point_cloud)

    # Interpoler TSS sur tous les points
    if tss is not None and window_centers is not None:
        tss_full = np.interp(np.arange(T), window_centers, tss.values)
    else:
        tss_full = np.linspace(0, 0.5, T)

    # Hover text
    hover = [
        f"<b>{dates[i].strftime('%Y-%m-%d')}</b><br>"
        f"TSS: {tss_full[i]:.3f}<br>"
        f"PC1: {point_cloud[i, 0]:.3f}<br>"
        f"PC2: {point_cloud[i, 1]:.3f}<br>"
        f"PC3: {point_cloud[i, 2]:.3f}"
        for i in range(T)
    ]

    fig = go.Figure()

    # ── Trace complète (fond, désaturée) ──
    fig.add_trace(go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode="lines",
        line=dict(color="rgba(100,130,180,0.12)", width=1),
        name="Trajectoire complète",
        hoverinfo="skip",
        showlegend=False,
    ))

    # ── Trace principale colorée ──
    fig.add_trace(go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode="lines+markers",
        line=dict(
            color=tss_full,
            colorscale=C["cs_risk"],
            width=4,
            cmin=0, cmax=1,
        ),
        marker=dict(
            size=np.where(tss_full > 0.7, 5, 2).tolist(),
            color=tss_full,
            colorscale=C["cs_risk"],
            colorbar=dict(
                title=dict(text="TSS", font=dict(size=11)),
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=["Calme", "0.25", "0.50", "0.75", "Stress"],
                thickness=12,
                len=0.7,
                x=1.05,
            ),
            cmin=0, cmax=1,
            opacity=0.9,
        ),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name="État du marché",
    ))

    # ── Point courant (dernier état) ──
    fig.add_trace(go.Scatter3d(
        x=[point_cloud[-1, 0]],
        y=[point_cloud[-1, 1]],
        z=[point_cloud[-1, 2]],
        mode="markers",
        marker=dict(
            size=14,
            color=_tss_rgb(tss_full[-1]),
            symbol="diamond",
            line=dict(color="white", width=2),
        ),
        name="État actuel",
        hovertemplate=f"<b>Maintenant</b><br>TSS: {tss_full[-1]:.3f}<extra></extra>",
    ))

    # ── Animation (frames progressives) ──
    if animate and T > frame_step * 3:
        frames = []
        frame_indices = list(range(frame_step, T + 1, frame_step))
        if frame_indices[-1] != T:
            frame_indices.append(T)

        for fi in frame_indices:
            frames.append(go.Frame(
                data=[
                    go.Scatter3d(
                        x=point_cloud[:fi, 0],
                        y=point_cloud[:fi, 1],
                        z=point_cloud[:fi, 2],
                        mode="lines+markers",
                        line=dict(
                            color=tss_full[:fi],
                            colorscale=C["cs_risk"],
                            width=4, cmin=0, cmax=1,
                        ),
                        marker=dict(
                            size=np.where(tss_full[:fi] > 0.7, 5, 2).tolist(),
                            color=tss_full[:fi],
                            colorscale=C["cs_risk"],
                            cmin=0, cmax=1,
                        ),
                    )
                ],
                traces=[1],
                name=str(fi),
            ))

        fig.frames = frames
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0.02,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="▶ Lancer",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=80, redraw=True),
                            fromcurrent=True,
                            mode="immediate",
                        )],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                        )],
                    ),
                ],
                font=dict(size=12, color=C["text"]),
                bgcolor=C["bg_panel"],
                bordercolor=C["grid"],
            )],
        )

    # ── Layout scène ──
    scene_style = dict(
        bgcolor=C["bg"],
        xaxis=dict(
            title="PC1 — Composante Principale",
            gridcolor=C["grid"], showbackground=False,
            tickfont=dict(size=8),
        ),
        yaxis=dict(
            title="PC2",
            gridcolor=C["grid"], showbackground=False,
            tickfont=dict(size=8),
        ),
        zaxis=dict(
            title="PC3",
            gridcolor=C["grid"], showbackground=False,
            tickfont=dict(size=8),
        ),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2),
            center=dict(x=0, y=0, z=-0.1),
        ),
    )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="🌀 Vortex de Marché — Espace des Phases (Takens Embedding 3D)",
                font=dict(size=15, color=C["text_bright"]),
                x=0.01,
            ),
            height=560,
            margin=dict(l=0, r=0, t=55, b=60),
        ),
        scene=scene_style,
    )

    return fig


def _tss_rgb(v: float) -> str:
    """Interpolation couleur Vert → Jaune → Rouge pour TSS ∈ [0,1]."""
    v = float(np.clip(v, 0, 1))
    if v < 0.5:
        r, g = int(255 * v * 2), 200
    else:
        r, g = 200, int(200 * (1 - (v - 0.5) * 2))
    return f"rgb({r},{g},20)"


# ═══════════════════════════════════════════════════════════════════════════
# 2. RADAR DE RISQUE TOPOLOGIQUE (Barcode amélioré)
# ═══════════════════════════════════════════════════════════════════════════

def plot_topological_radar(
    diagram: np.ndarray,
    entropy: float = 0.0,
    threshold: float = 0.0,
    title: str = "Radar de Risque Topologique",
) -> go.Figure:
    """Visualise les barcodes de persistance avec annotations d'entropie.

    Le "Radar Topologique" combine :
      - Barcodes H₀ (bleu) et H₁ (rouge) triés par persistance.
      - Ligne de seuil dynamique θ (IQR).
      - Annotation d'entropie (badge flottant).
      - Encadrement visuel du risque via couleurs d'arrière-plan.

    Args:
        diagram:   Array (n, 3) [birth, death, dim].
        entropy:   Entropie de persistance H₁ calculée (pour affichage).
        threshold: Seuil dynamique θ (ligne verticale).
        title:     Titre du graphique.

    Returns:
        go.Figure avec barcodes annotés.
    """
    fig = go.Figure()

    dim_cfg = {
        0: dict(color=C["accent"],  name="H₀ — Composantes connexes", max_bars=20),
        1: dict(color=C["danger"],  name="H₁ — Cycles topologiques",  max_bars=25),
    }

    y_cursor = 0
    h1_signif = 0

    for dim in [0, 1]:
        pts = diagram[diagram[:, 2] == dim]
        if len(pts) == 0:
            continue

        pers = pts[:, 1] - pts[:, 0]
        # Gérer les infinis (H0 : composante principale)
        max_finite = np.nanmax(pers[np.isfinite(pers)]) if np.any(np.isfinite(pers)) else 1.0
        pers_display = np.where(np.isinf(pers), max_finite * 1.2, pers)

        order    = np.argsort(pers_display)[::-1][:dim_cfg[dim]["max_bars"]]
        cfg      = dim_cfg[dim]

        # Zone de fond pour ce groupe
        fig.add_hrect(
            y0=y_cursor - 0.5,
            y1=y_cursor + len(order) - 0.5,
            fillcolor=f"rgba({20 if dim == 0 else 40},{20},{40},0.3)",
            line_width=0,
        )

        for rank, i in enumerate(order):
            birth = float(pts[i, 0])
            death = float(pts[i, 0] + pers_display[i])
            p     = float(pers_display[i])
            above_threshold = (threshold > 0 and p > threshold)

            if dim == 1 and above_threshold:
                h1_signif += 1

            line_color = cfg["color"] if above_threshold or threshold == 0 else "rgba(100,120,160,0.4)"
            lw = 5 if above_threshold else 2

            y = y_cursor + rank
            fig.add_trace(go.Scatter(
                x=[birth, death],
                y=[y, y],
                mode="lines",
                line=dict(color=line_color, width=lw),
                name=cfg["name"] if rank == 0 else None,
                showlegend=(rank == 0),
                hovertemplate=(
                    f"<b>{cfg['name']}</b><br>"
                    f"Birth: {birth:.4f}<br>"
                    f"Death: {death:.4f}<br>"
                    f"Persistance: {p:.4f}"
                    f"{'  ★ SIGNIFICATIF' if above_threshold else ''}"
                    "<extra></extra>"
                ),
            ))

        # Label de groupe
        fig.add_annotation(
            x=0, y=y_cursor + len(order) / 2,
            text=f"<b>{'H₀' if dim == 0 else 'H₁'}</b>",
            showarrow=False,
            xref="paper", yref="y",
            font=dict(color=cfg["color"], size=13),
            xanchor="left",
        )

        y_cursor += len(order) + 3

    # ── Seuil dynamique ──
    if threshold > 0:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=C["warn"],
            line_width=1.5,
            annotation_text=f"θ = {threshold:.3f}",
            annotation_position="top right",
            annotation_font=dict(color=C["warn"], size=10),
        )

    # ── Badge Entropie ──
    entropy_color = C["safe"] if entropy > 2.0 else C["warn"] if entropy > 1.0 else C["danger"]
    entropy_label = "Élevée" if entropy > 2.0 else "Modérée" if entropy > 1.0 else "Faible"
    fig.add_annotation(
        x=1.0, y=1.0,
        text=f"<b>Entropie H₁</b><br>{entropy:.2f} bits<br>({entropy_label})",
        showarrow=False,
        xref="paper", yref="paper",
        font=dict(color=entropy_color, size=11),
        bgcolor=f"rgba(8,12,20,0.9)",
        bordercolor=entropy_color,
        borderwidth=1,
        borderpad=6,
        align="center",
    )

    # ── Badge H₁ significatifs ──
    fig.add_annotation(
        x=1.0, y=0.82,
        text=f"<b>Cycles H₁ stables</b><br>{h1_signif}",
        showarrow=False,
        xref="paper", yref="paper",
        font=dict(color=C["danger"] if h1_signif > 3 else C["text"], size=11),
        bgcolor="rgba(8,12,20,0.9)",
        bordercolor=C["grid"],
        borderwidth=1,
        borderpad=6,
        align="center",
    )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=13, color=C["text_bright"])),
            height=420,
            margin=dict(l=10, r=140, t=50, b=30),
        ),
        xaxis=_axis_style(title="Échelle ε (Filtration Vietoris-Rips)"),
        yaxis=dict(showticklabels=False, **_axis_style(showgrid=False)),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3. PRIX + RISK OVERLAY AMÉLIORÉ
# ═══════════════════════════════════════════════════════════════════════════

def plot_prices_risk_overlay(
    prices: pd.DataFrame,
    tss: pd.Series,
    dates: pd.DatetimeIndex,
    window_centers: List[int],
    ticker: str = "SPY",
    signals: Optional[pd.Series] = None,
) -> go.Figure:
    """Prix interactif avec heatmap TSS et zones d'exclusion.

    Zones colorées en fond :
      - Vert  (TSS ≤ 0.4) : zone sûre.
      - Jaune (0.4 < TSS ≤ 0.7) : zone de vigilance.
      - Rouge (TSS > 0.7) : zone d'exclusion.

    Les zones d'exclusion correspondent aux périodes où la stratégie
    TDA serait hors du marché (position refuge/cash).

    Args:
        prices:         DataFrame de prix.
        tss:            pd.Series TSS (index entier).
        dates:          DatetimeIndex du nuage.
        window_centers: Positions des fenêtres.
        ticker:         Actif à afficher.
        signals:        Signaux de trading (0/1, optionnel) pour overlay.

    Returns:
        go.Figure 2 sous-graphes : prix + TSS.
    """
    # Mapper TSS → dates
    centers_dates = [dates[min(wc, len(dates) - 1)] for wc in window_centers]
    tss_dated = pd.Series(tss.values, index=pd.DatetimeIndex(centers_dates))

    price_col = prices[ticker] if ticker in prices.columns else prices.iloc[:, 0]
    ticker_label = ticker if ticker in prices.columns else prices.columns[0]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.68, 0.32],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=["", ""],
    )

    # ── Heatmap de fond par segments ──
    for i in range(len(tss_dated) - 1):
        t0 = tss_dated.index[i]
        t1 = tss_dated.index[i + 1]
        v  = float(tss_dated.iloc[i])

        if v < 0.40:
            r, g, b, a = 0, 200, 100, max(0.04, v * 0.15)
        elif v < 0.70:
            r, g, b, a = 255, 170, 0,   max(0.06, v * 0.20)
        else:
            r, g, b, a = 255, 30,  30,  max(0.08, v * 0.30)

        fig.add_vrect(
            x0=t0, x1=t1,
            fillcolor=f"rgba({r},{g},{b},{a:.3f})",
            layer="below", line_width=0,
            row=1, col=1,
        )

    # ── Annotations crises historiques ──
    for crisis_name, (c_start, c_end) in CRISIS_PERIODS.items():
        try:
            ts, te = pd.Timestamp(c_start), pd.Timestamp(c_end)
            if ts >= price_col.index[0] and ts <= price_col.index[-1]:
                fig.add_vrect(
                    x0=ts, x1=min(te, price_col.index[-1]),
                    annotation_text=f"  {crisis_name}",
                    annotation_position="top left",
                    annotation_font=dict(size=8, color="#ccc"),
                    fillcolor="rgba(160,80,80,0.06)",
                    line=dict(color="rgba(200,100,100,0.35)", width=1, dash="dot"),
                    row=1, col=1,
                )
        except Exception:
            pass

    # ── Signal de trading (zones refuges) ──
    if signals is not None:
        refuge_blocks = []
        in_block = False
        block_start = None

        for dt, sig in signals.items():
            if sig == 1 and not in_block:
                in_block = True
                block_start = dt
            elif sig == 0 and in_block:
                refuge_blocks.append((block_start, dt))
                in_block = False
        if in_block:
            refuge_blocks.append((block_start, signals.index[-1]))

        for bs, be in refuge_blocks:
            fig.add_vrect(
                x0=bs, x1=be,
                fillcolor="rgba(124,77,255,0.12)",
                layer="below", line_width=0,
                row=1, col=1,
            )

    # ── Courbe de prix ──
    fig.add_trace(
        go.Scatter(
            x=price_col.index, y=price_col.values,
            mode="lines",
            line=dict(color=C["accent"], width=1.8),
            name=ticker_label,
            hovertemplate="%{x|%Y-%m-%d}  $%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ── TSS ──
    fig.add_trace(
        go.Scatter(
            x=tss_dated.index, y=tss_dated.values,
            mode="lines", fill="tozeroy",
            line=dict(color=C["warn"], width=2.2),
            fillcolor="rgba(255,193,7,0.12)",
            name="TSS",
            hovertemplate="%{x|%Y-%m-%d}  TSS: %{y:.3f}<extra></extra>",
        ),
        row=2, col=1,
    )

    # Seuils TSS
    for thr, col, lbl in [(0.7, C["danger"], "Exclusion"), (0.4, C["warn"], "Vigilance")]:
        fig.add_hline(
            y=thr, line_dash="dot", line_color=col, line_width=1.2,
            annotation_text=lbl,
            annotation_position="right",
            annotation_font=dict(size=9, color=col),
            row=2, col=1,
        )

    # La clé 'legend' est passée DANS _base_layout() via **kwargs afin qu'elle
    # écrase la valeur par défaut via base.update(kwargs). Cela évite le
    # TypeError "multiple values for keyword argument 'legend'" qui surviendrait
    # si 'legend' était extrait du dict par **_base_layout() ET passé en argument
    # explicite simultanément à update_layout().
    fig.update_layout(
        **_base_layout(
            height=530,
            hovermode="x unified",
            legend=dict(
                bgcolor="rgba(8,12,20,0.8)",
                bordercolor=C["grid"], borderwidth=1,
                x=0.01, y=0.99,
            ),
        ),
    )

    for row in [1, 2]:
        fig.update_xaxes(**_axis_style(), row=row, col=1)
        fig.update_yaxes(**_axis_style(), row=row, col=1)

    fig.update_yaxes(title_text=f"Prix {ticker_label} ($)", row=1, col=1)
    fig.update_yaxes(title_text="TSS [0–1]", range=[0, 1.08], row=2, col=1)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. TSS AVEC ENVELOPPE DE SENSIBILITÉ
# ═══════════════════════════════════════════════════════════════════════════

def plot_tss_with_sensitivity(
    tss: pd.Series,
    dates: pd.DatetimeIndex,
    window_centers: List[int],
    sensitivity=None,
    components_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Courbe TSS avec bande de robustesse (analyse de sensibilité).

    La bande mean ± std autour du TSS central visualise la robustesse
    du signal : une bande étroite = signal stable → confiance élevée.

    Args:
        tss:             pd.Series TSS central.
        dates:           DatetimeIndex du nuage.
        window_centers:  Positions des fenêtres.
        sensitivity:     SensitivityResult (optionnel, pour la bande).
        components_df:   DataFrame des composantes normalisées (optionnel).

    Returns:
        go.Figure avec TSS + enveloppe + composantes.
    """
    centers_dates = [dates[min(wc, len(dates) - 1)] for wc in window_centers]
    tss_dated = pd.Series(tss.values, index=pd.DatetimeIndex(centers_dates))
    x = tss_dated.index
    y = tss_dated.values

    fig = go.Figure()

    # ── Zones de régime (fond) ──
    fig.add_hrect(y0=0,   y1=0.40, fillcolor="rgba(0,200,100,0.04)", line_width=0)
    fig.add_hrect(y0=0.40, y1=0.70, fillcolor="rgba(255,170,0,0.05)",  line_width=0)
    fig.add_hrect(y0=0.70, y1=1.05, fillcolor="rgba(255,30,30,0.06)",  line_width=0)

    # ── Enveloppe de sensibilité ──
    if sensitivity is not None and len(sensitivity.tss_mean) > 0:
        # Aligner sur les mêmes indices
        sens_idx = sensitivity.tss_mean.index
        sens_len = min(len(sens_idx), len(x))
        idx_common = list(range(sens_len))

        # Interpoler vers les dates
        up_vals  = sensitivity.tss_upper.values[:sens_len]
        low_vals = sensitivity.tss_lower.values[:sens_len]

        fig.add_trace(go.Scatter(
            x=x[:sens_len].tolist() + x[:sens_len].tolist()[::-1],
            y=up_vals.tolist() + low_vals.tolist()[::-1],
            fill="toself",
            fillcolor="rgba(0,176,255,0.08)",
            line=dict(color="rgba(0,176,255,0.0)"),
            name="Bande de Sensibilité (W±5, τ∈{1,2,3})",
            hoverinfo="skip",
        ))

    # ── TSS principal ──
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=C["warn"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(255,193,7,0.09)",
        name="TSS Central",
        hovertemplate="%{x|%Y-%m-%d}  TSS: %{y:.3f}<extra></extra>",
    ))

    # ── Composantes (optionnel, traces secondaires) ──
    if components_df is not None and len(components_df) > 0:
        comp_dates = [dates[min(wc, len(dates) - 1)] for wc in components_df.index]
        comp_dated_idx = pd.DatetimeIndex(comp_dates)

        comp_traces = [
            ("wasserstein_norm", C["accent"],  "W₁ Wasserstein", "dot"),
            ("amplitude_norm",   C["accent2"], "Amplitude H₁",   "dot"),
            ("entropy_collapse", C["danger"],  "Collapse Entropie", "dot"),
        ]
        for col, color, label, dash in comp_traces:
            if col in components_df.columns:
                fig.add_trace(go.Scatter(
                    x=comp_dated_idx,
                    y=components_df[col].values,
                    mode="lines",
                    line=dict(color=color, width=1, dash=dash),
                    opacity=0.55,
                    name=label,
                    hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
                    visible="legendonly",
                ))

    # Seuils
    for thr, col, lbl in [
        (0.70, C["danger"],  "Exclusion (θ=0.70)"),
        (0.40, C["warn"],    "Vigilance (θ=0.40)"),
    ]:
        fig.add_hline(
            y=thr, line_dash="dash",
            line_color=col, line_width=1.2,
            annotation_text=lbl,
            annotation_position="right",
            annotation_font=dict(size=9, color=col),
        )

    # Pic maximal
    max_idx = int(np.argmax(y))
    fig.add_annotation(
        x=x[max_idx], y=y[max_idx],
        text=f"⚠ Pic TSS<br>{y[max_idx]:.3f}",
        showarrow=True, arrowhead=2,
        arrowcolor=C["danger"],
        font=dict(color=C["danger"], size=10),
        bgcolor="rgba(255,23,68,0.12)",
        bordercolor=C["danger"], borderpad=4,
        ay=-35,
    )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="📈 Topological Stress Score — Série Temporelle & Robustesse",
                font=dict(size=13, color=C["text_bright"]),
            ),
            height=320,
            margin=dict(l=10, r=110, t=50, b=10),
        ),
        xaxis=_axis_style(),
        yaxis=_axis_style(title="TSS [0–1]", range=[0, 1.08]),
        showlegend=True,
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5. DASHBOARD DE BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

def plot_backtest_dashboard(result) -> go.Figure:
    """Graphique complet du backtest : équités + drawdown + signaux.

    Structure :
      Row 1 (60%) : Courbes d'équité TDA vs Buy & Hold.
      Row 2 (25%) : Drawdown comparatif.
      Row 3 (15%) : Signaux de rotation (0/1).

    Args:
        result: BacktestResult du module backtester.

    Returns:
        go.Figure 3-sous-graphes.
    """
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.55, 0.28, 0.17],
        shared_xaxes=True,
        vertical_spacing=0.025,
        subplot_titles=["", "", ""],
    )

    eq_tda = result.equity_tda
    eq_bnh = result.equity_bnh
    dd_tda = result.drawdown_tda
    dd_bnh = result.drawdown_bnh
    signals = result.signals

    # ── Équités ──
    fig.add_trace(
        go.Scatter(
            x=eq_tda.index, y=eq_tda.values,
            mode="lines",
            line=dict(color=C["accent"], width=2.2),
            fill="tozeroy",
            fillcolor="rgba(0,176,255,0.06)",
            name=f"TDA Strategy (θ={result.kpis_tda.strategy_name.split('=')[-1].strip(')')})",
            hovertemplate="%{x|%Y-%m-%d}  %{y:.3f}×<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=eq_bnh.index, y=eq_bnh.values,
            mode="lines",
            line=dict(color=C["text_dim"], width=1.5, dash="dot"),
            name="Buy & Hold",
            hovertemplate="%{x|%Y-%m-%d}  %{y:.3f}×<extra></extra>",
        ),
        row=1, col=1,
    )

    # Ligne de référence (équité = 1)
    fig.add_hline(y=1.0, line_dash="dash", line_color=C["grid"], line_width=1, row=1, col=1)

    # ── Drawdowns ──
    fig.add_trace(
        go.Scatter(
            x=dd_tda.index, y=dd_tda.values * 100,
            mode="lines", fill="tozeroy",
            line=dict(color=C["accent"], width=1.5),
            fillcolor="rgba(0,176,255,0.08)",
            name="DD TDA",
            hovertemplate="%{x|%Y-%m-%d}  DD: %{y:.1f}%<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dd_bnh.index, y=dd_bnh.values * 100,
            mode="lines", fill="tozeroy",
            line=dict(color=C["danger"], width=1.2, dash="dot"),
            fillcolor="rgba(255,23,68,0.06)",
            name="DD B&H",
            hovertemplate="%{x|%Y-%m-%d}  DD: %{y:.1f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    # ── Signaux de rotation ──
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals.values,
            mode="lines",
            line=dict(color=C["accent2"], width=1),
            fill="tozeroy",
            fillcolor="rgba(124,77,255,0.18)",
            name="Signal Refuge (1=actif)",
            hovertemplate="%{x|%Y-%m-%d}  Signal: %{y}<extra></extra>",
        ),
        row=3, col=1,
    )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="💼 Backtest TDA — Performance vs Buy & Hold",
                font=dict(size=14, color=C["text_bright"]),
            ),
            height=560,
            hovermode="x unified",
            margin=dict(l=10, r=10, t=55, b=10),
        ),
    )

    for row in [1, 2, 3]:
        fig.update_xaxes(**_axis_style(), row=row, col=1)

    fig.update_yaxes(title_text="Équité (×1.0)", **_axis_style(), row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", **_axis_style(), row=2, col=1)
    fig.update_yaxes(title_text="Signal", range=[-0.05, 1.3],
                     **_axis_style(showgrid=False), row=3, col=1)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. COMPOSANTES TSS (Spider / Gauge)
# ═══════════════════════════════════════════════════════════════════════════

def plot_tss_gauge(tss_value: float, label: str = "TSS Actuel") -> go.Figure:
    """Gauge semi-circulaire du TSS courant.

    Args:
        tss_value: Valeur TSS ∈ [0, 1].
        label:     Label affiché sous la jauge.

    Returns:
        go.Figure avec indicateur gauge.
    """
    if tss_value >= 0.7:
        color = C["danger"]
        regime = "STRESS CRITIQUE"
    elif tss_value >= 0.4:
        color = C["warn"]
        regime = "TRANSITION"
    else:
        color = C["safe"]
        regime = "RÉGIME CALME"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=tss_value * 100,
        title=dict(
            text=f"<b>{label}</b><br><span style='font-size:0.8em;color:{color}'>{regime}</span>",
            font=dict(size=14, color=C["text"]),
        ),
        number=dict(
            suffix="%",
            font=dict(size=28, color=color),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickvals=[0, 25, 40, 70, 100],
                ticktext=["0", "25", "40⚡", "70⚠", "100"],
                tickfont=dict(size=9, color=C["text_dim"]),
            ),
            bar=dict(color=color, thickness=0.3),
            bgcolor=C["bg_card"],
            borderwidth=1,
            bordercolor=C["grid"],
            steps=[
                dict(range=[0, 40],   color="rgba(0,200,100,0.12)"),
                dict(range=[40, 70],  color="rgba(255,170,0,0.12)"),
                dict(range=[70, 100], color="rgba(255,30,30,0.15)"),
            ],
            threshold=dict(
                line=dict(color=C["critical"], width=3),
                thickness=0.85,
                value=80,
            ),
        ),
    ))

    fig.update_layout(
        **_base_layout(height=230, margin=dict(l=20, r=20, t=60, b=10)),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 7. MULTI-ÉCHELLES — Divergence Topologique & Predictive-TSS (v2.3)
# ═══════════════════════════════════════════════════════════════════════════

def plot_multiscale_tss(
    tss_fast: pd.Series,
    tss_slow: pd.Series,
    delta_tss: pd.Series,
    ptss: pd.Series,
    early_warning: pd.Series,
    config: Dict,
) -> go.Figure:
    """Visualisation multi-panneau du système TDA Multi-Échelles.

    Structure (3 lignes) :
      Row 1 (50%) : TSS_fast & TSS_slow avec ruban de divergence.
        - Ligne bleue (TSS_fast, W_fast) — micro-instabilités.
        - Ligne verte atténuée (TSS_slow, W_slow) — structure de fond.
        - Ruban entre les deux : orange si TSS_fast > TSS_slow (fracture),
          bleu-vert si TSS_fast < TSS_slow (convergence / détente).
      Row 2 (30%) : ΔTSS (divergence brute) & P-TSS (score prédictif).
        - ΔTSS = TSS_fast − TSS_slow : indique la direction de la fracture.
        - P-TSS ∈ [0, 1] : normalise la fracture (déclencheur de stratégie).
      Row 3 (20%) : Signal Early Warning (accélération de ΔTSS).
        - Barre pleine rouge lorsque ΔTSS > μ(ΔTSS) + k·σ(ΔTSS).

    Args:
        tss_fast:      pd.Series TSS(W_fast) aligné sur les dates de prix.
        tss_slow:      pd.Series TSS(W_slow) aligné sur les dates de prix.
        delta_tss:     pd.Series ΔTSS ∈ [-1, 1].
        ptss:          pd.Series P-TSS ∈ [0, 1].
        early_warning: pd.Series[bool] signal EW.
        config:        Dict des hyperparamètres (pour les labels).

    Returns:
        go.Figure 3-sous-graphes.
    """
    w_fast = config.get("W_FAST", 20)
    w_slow = config.get("W_SLOW", 100)
    ew_k   = config.get("EW_ACCEL_K", 2.0)

    # Aligner les séries sur un index commun propre
    common_idx = tss_fast.index
    tf  = tss_fast.reindex(common_idx)
    ts  = tss_slow.reindex(common_idx)
    dt  = delta_tss.reindex(common_idx)
    pt  = ptss.reindex(common_idx)
    ew  = early_warning.reindex(common_idx).fillna(False).astype(float)

    dates = common_idx

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.50, 0.30, 0.20],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=["", "", ""],
    )

    # ── Row 1 : TSS_fast vs TSS_slow + Ruban ────────────────────────────

    # Séparer le ruban en deux zones (fracture vs convergence) via polygon
    x_ribbon = list(dates) + list(dates[::-1])

    # Zone de fracture : TSS_fast > TSS_slow → orange
    tf_vals = tf.values
    ts_vals = ts.values
    top_frac    = np.where(tf_vals >= ts_vals, tf_vals, ts_vals)
    bottom_frac = np.where(tf_vals >= ts_vals, ts_vals, tf_vals)

    # Fracture positive (fast > slow)
    y_frac = list(top_frac) + list(bottom_frac[::-1])
    fig.add_trace(go.Scatter(
        x=x_ribbon, y=y_frac,
        fill="toself",
        fillcolor="rgba(255,109,0,0.18)",   # orange = divergence/fracture
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="Fracture (fast > slow)",
        showlegend=True,
    ), row=1, col=1)

    # Convergence (slow > fast — pas de risque immédiat)
    top_conv    = np.where(ts_vals >= tf_vals, ts_vals, tf_vals)
    bottom_conv = np.where(ts_vals >= tf_vals, tf_vals, ts_vals)
    y_conv = list(top_conv) + list(bottom_conv[::-1])
    fig.add_trace(go.Scatter(
        x=x_ribbon, y=y_conv,
        fill="toself",
        fillcolor="rgba(0,176,255,0.10)",   # bleu = convergence/détente
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="Convergence (slow > fast)",
        showlegend=True,
    ), row=1, col=1)

    # TSS_slow (fond, ligne atténuée)
    fig.add_trace(go.Scatter(
        x=dates, y=ts_vals,
        mode="lines",
        line=dict(color="rgba(0,200,100,0.65)", width=1.5, dash="dot"),
        name=f"TSS_slow (W={w_slow}j)",
        hovertemplate="%{x|%Y-%m-%d}  TSS_slow: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # TSS_fast (premier plan, ligne principale)
    fig.add_trace(go.Scatter(
        x=dates, y=tf_vals,
        mode="lines",
        line=dict(color=C["accent"], width=2.2),
        name=f"TSS_fast (W={w_fast}j)",
        hovertemplate="%{x|%Y-%m-%d}  TSS_fast: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # Seuils de référence Row 1
    for thr, col_ref, lbl in [(0.70, C["danger"], "Stress"), (0.40, C["warn"], "Vigilance")]:
        fig.add_hline(
            y=thr, line_dash="dot", line_color=col_ref, line_width=0.8,
            annotation_text=lbl, annotation_position="right",
            annotation_font=dict(size=8, color=col_ref),
            row=1, col=1,
        )

    # ── Row 2 : ΔTSS & P-TSS ────────────────────────────────────────────

    # Ligne zéro
    fig.add_hline(y=0, line_color=C["grid"], line_width=0.8, row=2, col=1)

    # ΔTSS avec fill positif/négatif
    dt_pos = dt.clip(lower=0.0)
    dt_neg = dt.clip(upper=0.0)

    fig.add_trace(go.Scatter(
        x=dates, y=dt_pos.values,
        mode="lines", fill="tozeroy",
        line=dict(color=C["accent3"], width=1.5),
        fillcolor="rgba(255,109,0,0.15)",
        name="ΔTSS⁺ (fracture)",
        hovertemplate="ΔTSS: %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=dt_neg.values,
        mode="lines", fill="tozeroy",
        line=dict(color=C["accent"], width=1.5),
        fillcolor="rgba(0,176,255,0.10)",
        name="ΔTSS⁻ (convergence)",
        hovertemplate="ΔTSS: %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    # P-TSS (ligne vive)
    fig.add_trace(go.Scatter(
        x=dates, y=pt.values,
        mode="lines",
        line=dict(color=C["danger"], width=2.2),
        name="P-TSS (prédictif)",
        hovertemplate="P-TSS: %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    # Seuil P-TSS à 0.35
    fig.add_hline(
        y=0.35, line_dash="dash", line_color=C["danger"], line_width=1.0,
        annotation_text="θ P-TSS=0.35", annotation_position="right",
        annotation_font=dict(size=8, color=C["danger"]),
        row=2, col=1,
    )

    # ── Row 3 : Early Warning ───────────────────────────────────────────

    fig.add_trace(go.Scatter(
        x=dates, y=ew.values,
        mode="lines", fill="tozeroy",
        line=dict(color=C["critical"], width=1),
        fillcolor="rgba(213,0,0,0.30)",
        name=f"⚡ Early Warning (k={ew_k}σ)",
        hovertemplate="%{x|%Y-%m-%d}  EW: %{y}<extra></extra>",
    ), row=3, col=1)

    # ── Annotations crises ──────────────────────────────────────────────
    for crisis_name, (c_start, c_end) in CRISIS_PERIODS.items():
        try:
            ts_c = pd.Timestamp(c_start)
            te_c = pd.Timestamp(c_end)
            if ts_c >= dates[0] and ts_c <= dates[-1]:
                for row_n in [1, 2, 3]:
                    fig.add_vrect(
                        x0=ts_c,
                        x1=min(te_c, dates[-1]),
                        fillcolor="rgba(160,80,80,0.05)",
                        line=dict(color="rgba(200,100,100,0.3)", width=1, dash="dot"),
                        row=row_n, col=1,
                    )
                # Label uniquement sur row 1
                fig.add_annotation(
                    x=ts_c, y=1.04,
                    text=crisis_name,
                    showarrow=False,
                    xref="x", yref="paper",
                    font=dict(size=8, color="rgba(200,120,120,0.75)"),
                    textangle=-60,
                    xanchor="left",
                )
        except Exception:
            pass

    # ── Layout global ────────────────────────────────────────────────────
    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"⚡ Multi-Échelles TDA — TSS_fast(W={w_fast}) · TSS_slow(W={w_slow}) · P-TSS · Early Warning",
                font=dict(size=13, color=C["text_bright"]),
            ),
            height=560,
            hovermode="x unified",
            margin=dict(l=10, r=120, t=55, b=10),
            legend=dict(
                bgcolor="rgba(8,12,20,0.85)",
                bordercolor=C["grid"], borderwidth=1,
                x=1.01, y=1.0, xanchor="left",
                font=dict(size=9),
            ),
        ),
    )

    for row_n in [1, 2, 3]:
        fig.update_xaxes(**_axis_style(), row=row_n, col=1)

    fig.update_yaxes(title_text="TSS [0–1]",    range=[0, 1.1],  **_axis_style(), row=1, col=1)
    fig.update_yaxes(title_text="ΔTSS & P-TSS", range=[-1, 1.1], **_axis_style(), row=2, col=1)
    fig.update_yaxes(title_text="EW",            range=[-0.05, 1.3],
                     **_axis_style(showgrid=False, showticklabels=False), row=3, col=1)

    return fig


def plot_multiscale_overlay(
    prices: pd.DataFrame,
    tss_fast: pd.Series,
    tss_slow: pd.Series,
    ptss: pd.Series,
    early_warning: pd.Series,
    ticker: str = "SPY",
    signals: Optional[pd.Series] = None,
    config: Optional[Dict] = None,
) -> go.Figure:
    """Prix + superposition des deux TSS + P-TSS + zones EW.

    Version enrichie de plot_prices_risk_overlay() avec :
      - Deux courbes TSS (fast/slow) dans le panneau TSS.
      - P-TSS en ligne rouge pointillée.
      - Zones Early Warning en fond violet sur le panneau prix.

    Args:
        prices:        DataFrame de prix.
        tss_fast:      pd.Series TSS(W_fast) aligné sur les dates de prix.
        tss_slow:      pd.Series TSS(W_slow) aligné sur les dates de prix.
        ptss:          pd.Series P-TSS ∈ [0, 1].
        early_warning: pd.Series[bool] signal EW.
        ticker:        Actif à afficher.
        signals:       Signaux P-TSS (optionnel, pour zones refuge).
        config:        Dict hyperparamètres multi-échelles.

    Returns:
        go.Figure 2 sous-graphes : prix enrichi + TSS triple.
    """
    if config is None:
        config = {}
    w_fast = config.get("W_FAST", 20)
    w_slow = config.get("W_SLOW", 100)

    price_col = prices[ticker] if ticker in prices.columns else prices.iloc[:, 0]
    ticker_label = ticker if ticker in prices.columns else prices.columns[0]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.60, 0.40],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=["", ""],
    )

    # ── Heatmap de fond via P-TSS ───────────────────────────────────────
    ptss_reindexed = ptss.reindex(price_col.index).ffill().fillna(0.0)
    for i in range(len(ptss_reindexed) - 1):
        t0 = ptss_reindexed.index[i]
        t1 = ptss_reindexed.index[i + 1]
        v  = float(ptss_reindexed.iloc[i])
        if v < 0.30:
            r, g, b, a = 0, 200, 100, max(0.03, v * 0.10)
        elif v < 0.60:
            r, g, b, a = 255, 130, 0,  max(0.05, v * 0.18)
        else:
            r, g, b, a = 255, 23,  68, max(0.08, v * 0.28)
        fig.add_vrect(
            x0=t0, x1=t1,
            fillcolor=f"rgba({r},{g},{b},{a:.3f})",
            layer="below", line_width=0, row=1, col=1,
        )

    # ── Early Warning zones ──────────────────────────────────────────────
    ew_reindexed = early_warning.reindex(price_col.index).ffill().fillna(False)
    ew_blocks, in_block, block_start = [], False, None
    for dt_idx, ew_val in ew_reindexed.items():
        if ew_val and not in_block:
            in_block, block_start = True, dt_idx
        elif not ew_val and in_block:
            ew_blocks.append((block_start, dt_idx))
            in_block = False
    if in_block:
        ew_blocks.append((block_start, ew_reindexed.index[-1]))
    for bs, be in ew_blocks:
        fig.add_vrect(
            x0=bs, x1=be,
            fillcolor="rgba(213,0,0,0.12)",
            layer="below", line_width=0, row=1, col=1,
        )

    # ── Annotations crises ───────────────────────────────────────────────
    for crisis_name, (c_start, c_end) in CRISIS_PERIODS.items():
        try:
            ts_c, te_c = pd.Timestamp(c_start), pd.Timestamp(c_end)
            if ts_c >= price_col.index[0] and ts_c <= price_col.index[-1]:
                fig.add_vrect(
                    x0=ts_c, x1=min(te_c, price_col.index[-1]),
                    annotation_text=f"  {crisis_name}",
                    annotation_position="top left",
                    annotation_font=dict(size=8, color="#ccc"),
                    fillcolor="rgba(160,80,80,0.05)",
                    line=dict(color="rgba(200,100,100,0.35)", width=1, dash="dot"),
                    row=1, col=1,
                )
        except Exception:
            pass

    # ── Signal refuge (P-TSS) ────────────────────────────────────────────
    if signals is not None:
        refuge_blocks, in_block, block_start = [], False, None
        for dt_idx, sig in signals.items():
            if sig == 1 and not in_block:
                in_block, block_start = True, dt_idx
            elif sig == 0 and in_block:
                refuge_blocks.append((block_start, dt_idx))
                in_block = False
        if in_block:
            refuge_blocks.append((block_start, signals.index[-1]))
        for bs, be in refuge_blocks:
            fig.add_vrect(
                x0=bs, x1=be,
                fillcolor="rgba(124,77,255,0.10)",
                layer="below", line_width=0, row=1, col=1,
            )

    # ── Courbe de prix ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=price_col.index, y=price_col.values,
        mode="lines",
        line=dict(color=C["accent"], width=1.8),
        name=ticker_label,
        hovertemplate="%{x|%Y-%m-%d}  $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # ── TSS_slow (fond, vert pointillé) ──────────────────────────────────
    fig.add_trace(go.Scatter(
        x=tss_slow.index, y=tss_slow.values,
        mode="lines",
        line=dict(color="rgba(0,200,100,0.6)", width=1.5, dash="dot"),
        name=f"TSS_slow (W={w_slow}j)",
        hovertemplate=f"TSS_slow: %{{y:.3f}}<extra></extra>",
    ), row=2, col=1)

    # ── TSS_fast (avant-plan, jaune) ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=tss_fast.index, y=tss_fast.values,
        mode="lines", fill="tozeroy",
        line=dict(color=C["warn"], width=2.0),
        fillcolor="rgba(255,193,7,0.09)",
        name=f"TSS_fast (W={w_fast}j)",
        hovertemplate=f"TSS_fast: %{{y:.3f}}<extra></extra>",
    ), row=2, col=1)

    # ── P-TSS (rouge pointillé vif) ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ptss.index, y=ptss.values,
        mode="lines",
        line=dict(color=C["danger"], width=1.8, dash="dashdot"),
        name="P-TSS (prédictif)",
        hovertemplate="P-TSS: %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    # Seuils Row 2
    for thr, col_ref, lbl in [
        (0.70, C["danger"], "Exclusion"), (0.35, C["accent3"], "P-TSS θ"), (0.40, C["warn"], "Vigilance"),
    ]:
        fig.add_hline(
            y=thr, line_dash="dot", line_color=col_ref, line_width=1.0,
            annotation_text=lbl, annotation_position="right",
            annotation_font=dict(size=9, color=col_ref),
            row=2, col=1,
        )

    fig.update_layout(
        **_base_layout(
            height=550,
            hovermode="x unified",
            legend=dict(
                bgcolor="rgba(8,12,20,0.85)",
                bordercolor=C["grid"], borderwidth=1,
                x=0.01, y=0.99,
            ),
        ),
    )

    for row_n in [1, 2]:
        fig.update_xaxes(**_axis_style(), row=row_n, col=1)
        fig.update_yaxes(**_axis_style(), row=row_n, col=1)

    fig.update_yaxes(title_text=f"Prix {ticker_label} ($)", row=1, col=1)
    fig.update_yaxes(title_text="TSS [0–1]", range=[0, 1.1], row=2, col=1)

    return fig


def plot_components_radar(components_df: pd.DataFrame, idx: int = -1) -> go.Figure:
    """Graphique radar des 3 composantes du TSS pour une fenêtre donnée.

    Args:
        components_df: DataFrame des composantes normalisées.
        idx:           Index de la fenêtre à afficher. Défaut = -1 (dernière).

    Returns:
        go.Figure radar (polar).
    """
    if len(components_df) == 0:
        return go.Figure()

    row = components_df.iloc[idx]
    categories = ["Wasserstein\n(Vitesse)", "Amplitude\nH₁", "Collapse\nEntropie"]
    values     = [
        float(row.get("wasserstein_norm", 0)),
        float(row.get("amplitude_norm", 0)),
        float(row.get("entropy_collapse", 0)),
    ]
    values_plot = values + [values[0]]  # Fermer le polygone

    tss_val = float(row.get("tss_final", sum(values) / 3))
    fill_color = (
        "rgba(255,30,30,0.18)"  if tss_val > 0.7 else
        "rgba(255,193,7,0.15)"  if tss_val > 0.4 else
        "rgba(0,200,100,0.12)"
    )
    line_color = C["danger"] if tss_val > 0.7 else C["warn"] if tss_val > 0.4 else C["safe"]

    fig = go.Figure(go.Scatterpolar(
        r=values_plot,
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2),
        marker=dict(size=7, color=line_color),
        hovertemplate="<b>%{theta}</b><br>%{r:.3f}<extra></extra>",
        name="Composantes TSS",
    ))

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="Décomposition TSS",
                font=dict(size=12, color=C["text_bright"]),
                x=0.5,
            ),
            height=280,
            margin=dict(l=40, r=40, t=50, b=10),
        ),
        polar=dict(
            bgcolor=C["bg_card"],
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                gridcolor=C["grid"],
                linecolor=C["grid"],
                tickfont=dict(size=8, color=C["text_dim"]),
            ),
            angularaxis=dict(
                linecolor=C["grid"],
                gridcolor=C["grid"],
                tickfont=dict(size=10, color=C["text"]),
            ),
        ),
        showlegend=False,
    )

    return fig
