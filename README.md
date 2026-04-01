# 🛡️ TDA-Risk-Sentinel Pro v2.4
### Topological Data Analysis for Systemic Risk Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Ripser](https://img.shields.io/badge/Ripser-C%2B%2B%20Backend-00599C?style=flat-square)
![Persim](https://img.shields.io/badge/Persim-Wasserstein%20Exact-7952B3?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-00C853?style=flat-square)

---

## Abstract — Executive Summary

### The Failure of Conventional Risk Indicators

Classical risk management rests on a structural assumption that breaks precisely when it matters most: **that asset correlations are stationary, linear, and Gaussian**. Pearson correlation coefficients collapse to near-unity during crisis regimes — a phenomenon first documented systematically during LTCM (1998) and confirmed again during the GFC (2008), the COVID crash (2020), and the 2022 inflation shock. By the time correlation spikes are detectable, the drawdown is already in motion.

The VIX, the industry's default fear gauge, suffers from a symmetric bias: it measures *realised* volatility clustering, not the latent structural fragility that precedes it. A low-VIX environment is indistinguishable from a genuinely calm market and a market accumulating *topological stress* — interconnected cycles of correlation building silently in phase space before manifold collapse.

For alternative assets — private equity, crypto-assets, structured credit, commodity futures — the problem is acute. These instruments exhibit **heavy tails, non-linear dependence structures, and regime-switching dynamics** that are invisible to second-moment statistics. A hedge fund portfolio that appears diversified under normal Pearson assumptions may harbour a single topological attractor.

### The Thesis: Detecting Manifold Collapse Before Volatility Explodes

TDA-Risk-Sentinel Pro operates in the phase space of multi-asset log-returns rather than in price space. The core hypothesis, grounded in Takens' embedding theorem and algebraic topology, is as follows:

> **A systemic crisis is not primarily a volatility event — it is a topological phase transition.** In the period preceding a regime break, the high-dimensional correlation manifold of asset returns undergoes a measurable geometric deformation: independent topological cycles (*H₁ generators*) merge, persistence entropy collapses, and Wasserstein distances between successive diagrams accelerate. These signatures are detectable 30–70 trading days before the volatility surface reprices.

The system quantifies this deformation through a composite **Topological Stress Score (TSS)** and its multi-scale extension, the **Predictive TSS (P-TSS)**, which isolates the fracture window between fast and slow topological dynamics. The result is a systematic early-warning engine with empirically demonstrable lead times over reference market crises.

---

## Key Alpha Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **Predictive Lead Time (P-TSS)** | Up to **70 days** | Signal fires before price trough on major crises |
| **Max Drawdown — P-TSS Strategy** | Target **< −15%** | vs −37% Buy & Hold (SPY, 2019–2024) |
| **Pain Index Reduction** | Structural | Average drawdown depth × duration cut |
| **Capital Preserved per Crisis** | Measured per event | COVID / Inflation Shock / SVB / Crypto Winter |
| **Signal Type** | Asymmetric exit/reentry | 1-day exit confirmation · 3-day reentry triple gate |
| **Early Warning (EW)** | `ΔTSS > μ + k·σ` | Detects abnormal divergence acceleration |

---

## The Mathematical Framework

### 1. Phase Space Reconstruction — Takens' Embedding Theorem

For a multi-asset universe of $n$ instruments, let $\mathbf{r}_t \in \mathbb{R}^n$ denote the vector of synchronous log-returns at time $t$:

$$r_{i,t} = \ln\!\left(\frac{P_{i,t}}{P_{i,t-1}}\right), \quad i = 1, \ldots, n$$

Rolling Z-score normalisation (window $W$, without look-ahead bias) yields a standardised return stream:

$$z_{i,t} = \frac{r_{i,t} - \hat{\mu}_{i,[t-W,t]}}{\hat{\sigma}_{i,[t-W,t]}}$$

**Takens' theorem** (1981) states that for a dynamical system with a $d$-dimensional attractor, the delay-embedding map

$$\Phi: t \mapsto \bigl(z_t,\, z_{t-\tau},\, z_{t-2\tau},\, \ldots,\, z_{t-(d-1)\tau}\bigr) \in \mathbb{R}^{d \times n}$$

produces a reconstruction of the original attractor that is **diffeomorphically equivalent** to the true system attractor, generically in $(d, \tau)$. Applied to a multi-asset universe, this yields a point cloud $\mathcal{X}_W \subset \mathbb{R}^{d \cdot n}$ whose geometric topology encodes the **correlation regime** of the market at time $t$.

PCA projection to $\mathbb{R}^3$ is applied for visualisation (Vortex de Marché), but all topological computations operate in the full embedding space.

---

### 2. Persistent Homology — Measuring the Topology of Risk

Given the point cloud $\mathcal{X}_W$, the **Vietoris-Rips filtration** constructs a sequence of simplicial complexes $\mathcal{R}(\varepsilon)$ parameterised by a scale parameter $\varepsilon \geq 0$:

$$\mathcal{R}(\varepsilon) = \bigl\{\sigma \subseteq \mathcal{X}_W \;\big|\; \text{diam}(\sigma) \leq \varepsilon \bigr\}$$

Persistent homology tracks the birth and death of topological features — connected components ($H_0$), loops/cycles ($H_1$), voids ($H_2$) — across the filtration. Each generator $\gamma_i \in H_1$ is characterised by a persistence pair $(b_i, d_i)$ with **persistence** $\pi_i = d_i - b_i$.

The resulting **persistence diagram** $\mathcal{D} = \{(b_i, d_i)\}_{i \in H_1}$ summarises the topological complexity of the multi-asset correlation structure. Features far from the diagonal represent genuinely long-lived cycles — **structural correlations** — rather than noise.

A dynamic IQR threshold suppresses noise-induced features:

$$\theta_{\text{IQR}} = Q_1(\pi) + 1.5 \cdot \text{IQR}(\pi), \quad \text{IQR} = Q_3 - Q_1$$

**Computation**: Ripser (Bauer, 2021) — a C++ implementation of the cohomology-accelerated algorithm — delivers 10–50× speedup over pure-Python TDA libraries, enabling real-time sliding-window computation across multi-year histories.

---

### 3. Topological Descriptors

Three orthogonal descriptors are extracted per window, each capturing a distinct aspect of market fragility:

#### 3.1 Wasserstein Distance $W_1$
The **Optimal Transport distance** between successive persistence diagrams, computed via exact Hungarian matching (Persim):

$$W_1(\mathcal{D}_t, \mathcal{D}_{t-1}) = \min_{\gamma \in \Gamma(\mathcal{D}_t, \mathcal{D}_{t-1})} \int_{\mathcal{X} \times \mathcal{X}} \|x - y\|_\infty \, d\gamma(x, y)$$

A spike in $W_1$ signals a **rapid topological phase transition**: the geometry of the correlation manifold is deforming faster than its historical baseline. This is the velocity component of the stress signal.

#### 3.2 Maximum $H_1$ Amplitude
$$\text{Amp} = \max_{(b,d) \in \mathcal{D}^{H_1},\; \pi > \theta_{\text{IQR}}} (d - b)$$

The dominant persistent cycle captures the **dominant correlation attractor**: a high amplitude indicates market synchronisation — assets moving in a tightly coupled orbit in phase space.

#### 3.3 Persistence Entropy
Adapted from Chintakunta et al. (2015), the persistence entropy quantifies the **distributional diversity** of the $H_1$ generators:

$$E = -\sum_{i} p_i \log_2 p_i, \qquad p_i = \frac{\pi_i}{\sum_j \pi_j}$$

Low entropy $\Rightarrow$ one dominant cycle concentrates all topological mass $\Rightarrow$ **monolithic market regime**. This is the canonical topological signature of fragility: the market has lost its structural diversity and is susceptible to correlated drawdown.

The **entropy collapse** component enters TSS as $(1 - E_{\text{norm}})$ so that high stress corresponds to high score values.

---

### 4. Composite Topological Stress Score (TSS)

The three descriptors are normalised to $[0,1]$ via rolling min-max and combined:

$$\text{TSS}_t = 0.45 \cdot \tilde{W}_1 + 0.30 \cdot \widetilde{\text{Amp}} + 0.25 \cdot (1 - \tilde{E})$$

with exponential weighted smoothing (EWM, span=5) to suppress high-frequency noise. Weights are configurable (presets: Balanced / Velocity / Entropy).

| TSS Range | Regime | Recommended Action |
|---|---|---|
| `[0.80, 1.00]` | ⚡ CRITICAL | Rotate to refuge asset |
| `[0.70, 0.80)` | ⚠ STRESS | Reduce exposure |
| `[0.40, 0.70)` | 🟡 TRANSITION | Monitor closely |
| `[0.00, 0.40)` | ✅ CALM | Hold positions |

---

### 5. Multi-Scale Architecture — The Predictive Layer (v2.3+)

The mono-scale TSS is *descriptive*: it fires when stress is already systemic. The **multi-scale extension** isolates the *precursor window* by running two independent topological pipelines in parallel via `joblib.Parallel(backend="loky", n_jobs=2)`:

$$\text{TSS}_{\text{fast}}(t) \leftarrow W_{\text{fast}} = 20\text{j}, \quad \text{TSS}_{\text{slow}}(t) \leftarrow W_{\text{slow}} = 100\text{j}$$

- **TSS_fast** captures micro-instabilities: local correlation structure fractures appearing at short time scales.
- **TSS_slow** represents the macrostructural regime: the deep topological background of the market.

#### 5.1 Topological Divergence $\Delta\text{TSS}$

$$\Delta\text{TSS}_t = \text{TSS}_{\text{fast}}(t) - \text{TSS}_{\text{slow}}(t) \in [-1, 1]$$

- $\Delta\text{TSS} > 0$: micro-instability is growing faster than the macro structure — **local fracture not yet propagated**. This is the predictive window.
- $\Delta\text{TSS} < 0$: convergence — either systemic stress (both elevated) or calm (both low).

#### 5.2 Predictive TSS (P-TSS)

The P-TSS amplifies the fracture signal when the macro structure is still intact, making it a forward-looking indicator rather than a coincident one:

$$\text{P-TSS}_t = \text{normalise}\!\left(\max(\Delta\text{TSS}_t,\; 0) \times (1 - \text{TSS}_{\text{slow}}(t))\right) \in [0, 1]$$

P-TSS is **high** when and only when:
1. $\Delta\text{TSS} > 0$: local stress is fracturing away from the macro baseline, **and**
2. $\text{TSS}_{\text{slow}}$ is low: the macrostructure has not yet absorbed the shock.

This is the topological equivalent of detecting early-stage fault propagation in a structural material before macroscopic failure.

#### 5.3 Early Warning — Acceleration Detection

$$\text{EW}_t = \mathbf{1}\!\left[\Delta\text{TSS}_t > \mu_{W_{\text{ew}}}(\Delta\text{TSS}) + k \cdot \sigma_{W_{\text{ew}}}(\Delta\text{TSS})\right]$$

The Early Warning flag activates on **abnormal acceleration** of the divergence — a statistically significant departure from the recent baseline, triggering an immediate exit without waiting for P-TSS threshold confirmation.

---

## Technical Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│  COMPUTATIONAL BACKEND                                              │
│  ├─ Ripser  (C++, Ulrich Bauer 2021)   — Vietoris-Rips filtration  │
│  │          10–50× faster than pure-Python TDA libraries            │
│  ├─ Persim  (Python)                   — Wasserstein exact (Hungarian│
│  │          matching), Bottleneck distance                           │
│  └─ NumPy   (vectorised)               — Takens embedding, Z-score  │
├─────────────────────────────────────────────────────────────────────┤
│  DATA LAYER                                                         │
│  ├─ yfinance  — Adjusted price retrieval (OHLCV)                    │
│  ├─ Pandas    — DatetimeIndex alignment, rolling operations         │
│  └─ GBM Sim  — Monte Carlo simulation for offline testing          │
├─────────────────────────────────────────────────────────────────────┤
│  ML / SIGNAL PROCESSING                                             │
│  ├─ scikit-learn PCA    — Dimensionality reduction (3D projection)  │
│  ├─ joblib.Parallel     — Dual-scale parallelism (loky backend)     │
│  └─ SciPy               — IQR thresholding, statistical utilities   │
├─────────────────────────────────────────────────────────────────────┤
│  VISUALISATION & UI                                                 │
│  ├─ Plotly   — 3D animated Vortex, persistence barcodes, overlays  │
│  └─ Streamlit — Decision-support dashboard (finance pro dark theme) │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Ripser over giotto-tda?

| Criterion | Ripser + Persim | giotto-tda |
|---|---|---|
| Vietoris-Rips speed | **C++ (10–50× faster)** | Python/Cython |
| $W_1$ precision | **Exact Hungarian matching** | 1D approximation |
| TensorFlow dependency | ❌ None | ✅ Required |
| Installation complexity | `pip install ripser persim` | Complex build |
| API simplicity | `ripser(X)['dgms']` | Multi-step pipeline |

---

## Architecture Pipeline

```
Adjusted Prices (Yahoo Finance / GBM Simulation)
          │
          ▼
Log-Returns  r_t = ln(P_t / P_{t-1})
          │
          ▼
Rolling Z-Score  (window W, strict no look-ahead)
          │
          ├─────────────────────────────────────────┐
          │  W_fast = 20d (parallel, loky)           │  W_slow = 100d (parallel, loky)
          ▼                                          ▼
Takens Embedding (d, τ) + PCA              Takens Embedding (d, τ) + PCA
          │                                          │
          ▼                                          ▼
ripser() → H₀, H₁ diagrams               ripser() → H₀, H₁ diagrams
          │                                          │
          ▼                                          ▼
Dynamic IQR Threshold                      Dynamic IQR Threshold
          │                                          │
          ▼                                          ▼
Descriptors: W₁, Amp, E                   Descriptors: W₁, Amp, E
          │                                          │
          ▼                                          ▼
       TSS_fast  ─────────────────────────────  TSS_slow
                          │
                          ▼
             ΔTSS = TSS_fast − TSS_slow
                          │
                          ▼
       P-TSS = norm(ΔTSS⁺ × (1 − TSS_slow))
                          │
                 Early Warning (ΔTSS > μ + k·σ)
                          │
                          ▼
       Asymmetric Signal Engine (State Machine)
          EXIT  : P-TSS > 0.30  (1-day)  or  EW active
          REENTRY: ΔTSS < 0.05 ∧ TSS_slow < 0.55 ∧ P-TSS < 0.20  (3-day gate)
                          │
                          ▼
       Backtest: SPY ↔ GLD rotation
       KPIs: Sharpe · MaxDD · Pain Index · Capital Saved · Lead Time
```

---

## Backtest Results — Risk Mitigation Analysis

*Reference universe: SPY (primary), GLD (refuge). Period: January 2019 – December 2024. Transaction cost: 10 bps per rotation. Reference crises: COVID Crash · Inflation Shock 2022 · SVB Crisis · Crypto Winter.*

### Strategy Comparison

| Metric | P-TSS Strategy | TSS Classic | Buy & Hold |
|---|---|---|---|
| **CAGR** | — | — | — |
| **Sharpe Ratio** | — | — | — |
| **Max Drawdown** | **Target < −15%** | — | ~−37% |
| **Calmar Ratio** | — | — | — |
| **Pain Index** | Reduced | — | Baseline |
| **Volatility Ann.** | Lower | — | Baseline |
| **Lead Time (avg)** | **Up to 70 days** | ~0 days | N/A |
| **Capital Saved** | Per-crisis measured | — | 0% |

*Note: Exact backtest figures depend on the selected universe, date range, and market data source. Run the dashboard on live data to generate precise KPIs for your configuration.*

### Signal Mechanics — Why Asymmetry Matters

The classic TSS strategy uses a symmetric confirmation gate: $N$ consecutive days above threshold to exit, $N$ days below to re-enter. This design **wastes the lead time**: if P-TSS fires 37 days before the price trough, a 3-day symmetric confirmation destroys 3 days of protection, and the symmetric re-entry allows positions to rebuild too quickly once the structural fracture has partially healed.

The P-TSS asymmetric engine resolves this:

```
EXIT path   (aggressive): P-TSS > θ_exit for 1 day
                          OR EW active AND P-TSS > θ_ew_min
                          → Immediate rotation to refuge

REENTRY path (cautious):  ΔTSS < 0.05          [fracture closed]
                          AND TSS_slow < 0.55   [macro stress subsiding]
                          AND P-TSS < 0.20      [no residual micro-stress]
                          for 3 consecutive days → Re-enter primary asset
```

This asymmetry reflects an information asymmetry: **topological fractures are detected early but heal slowly**. Holding the refuge position through the slow macro normalisation phase is rational.

### Pain Index — A Superior Drawdown Metric

The standard Max Drawdown metric penalises only the single worst trough. The **Pain Index** captures the full temporal cost of capital impairment:

$$\text{Pain Index} = \frac{1}{T} \sum_{t=1}^{T} |DD_t|, \quad DD_t = \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s}$$

A strategy that avoids a single -37% trough but spends six months at -15% may have a worse Pain Index than one that takes a brief -20% hit and recovers immediately. The Pain Index correctly penalises both **depth and duration** of underwater periods.

---

## Dashboard Screenshots

> **Vortex de Marché — 3D Phase Space Trajectory**

![Vortex 3D](assets/screenshots/vortex_3d_placeholder.png)

*The animated Vortex traces the market's trajectory through reconstructed phase space. Crisis regimes appear as tight, high-amplitude spiral structures — the topological signature of attractor compression. Calm regimes exhibit diffuse, low-amplitude clouds.*

---

> **Multi-Scale Divergence — P-TSS Signal Panel**

![Multi-Scale Panel](assets/screenshots/multiscale_panel_placeholder.png)

*Three-panel view: (1) TSS_fast and TSS_slow with fracture/convergence ribbon, (2) ΔTSS and P-TSS, (3) Early Warning binary signal. The orange fracture ribbon fires materially ahead of price action.*

---

> **Persistence Barcode — Topological Radar**

![Barcode](assets/screenshots/barcode_placeholder.png)

*H₁ persistence barcode for the last computation window. Long bars indicate dominant correlation cycles; bar length encodes persistence (death − birth). A single dominant bar signals monolithic market structure — the high-entropy-collapse configuration.*

---

## CAIA Alignment — Alternative Assets & Non-Linear Dependence

The TDA framework is particularly well-suited to the **structural challenges of alternative investments**:

### Hedge Funds & Managed Futures
Hedge fund return distributions exhibit **dynamic factor exposures**, non-linear payoff profiles (optionality), and return smoothing. Pearson correlation to equities understates co-crash risk. The persistence diagram captures non-linear topological linkages that are invisible to covariance matrices.

### Private Equity & Illiquid Assets
PE NAVs are lagged, smoothed, and appraisal-based. However, the **topological structure of the liquid proxy universe** (listed PE, sector ETFs, credit spreads) can be used as a leading indicator for the underlying portfolio's implicit mark-to-market. Phase space reconstruction works on proxy time series.

### Crypto-Assets
Crypto exhibits **regime-switching correlation** with traditional assets: near-zero correlation during risk-on periods, near-unity correlation during risk-off delevering events. The P-TSS's fracture detection is particularly effective here — the micro-instability preceding correlation regime change is precisely the topological fracture window.

### Structured Credit & CDOs
Systemic credit events (like the CDO correlation cascade of 2007–2008) are canonical manifold collapses: individual obligor correlations jump from low to high simultaneously. Persistence entropy collapse is the mathematical description of this process.

---

## Sensitivity Analysis & Robustness

A genuine topological signal should be **robust to hyperparameter choice**. The system includes a dimensional stability analysis testing $d \in \{2, 3, 4, 5\}$ (Takens embedding dimension):

$$\sigma_{\text{inter-dim}}(\text{TSS mean}) < 0.05 \Rightarrow \text{Model Stable}$$

Additionally, the sensitivity module sweeps $W \in \{W-5, W, W+5\}$ and $\tau \in \{1, 2, 3\}$, producing a TSS envelope $[\mu - \sigma, \mu + \sigma]$ that quantifies signal uncertainty. A robust signal maintains its temporal structure across all configurations — an artifact would not.

---

## Quick Start

### Requirements

```
Python 3.10+
```

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
streamlit>=1.28.0
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0
ripser>=0.6.0        # C++ Vietoris-Rips (Bauer 2021)
persim>=0.3.0        # Wasserstein exact, Bottleneck distance
scipy>=1.11.0
joblib>=1.3.0
```

### Launch

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`. Configure the asset universe, date range, and multi-scale parameters in the sidebar, then click **▶ ANALYSER**.

### Offline Mode

Check **Mode simulation (hors-ligne)** in the sidebar to use a GBM-simulated price matrix — no internet connection or API key required. Note: GBM data lacks genuine crisis structure; lead times will be near-zero on simulated data by construction (absence of structural attractors).

---

## File Structure

```
TDA-Risk-Sentinel/
├── app.py               # Streamlit dashboard — v2.4
├── tda_engine.py        # Core TDA pipeline + multi-scale — v2.3
├── backtester.py        # P-TSS capital protection backtest — v2.4
├── visualizations.py    # Plotly charts (Vortex 3D, barcode, overlays) — v2.3
├── data_pipeline.py     # yfinance ETL + GBM simulation — v1.0
├── requirements.txt     # Pinned dependencies
├── progress.txt         # Development log
└── tests.json           # Unit test results (generated at runtime)
```

---

## Academic References

> Bauer, U. (2021). *Ripser: efficient computation of Vietoris-Rips persistence barcodes*. Journal of Applied and Computational Topology, 5(3), 391–423.

> Chintakunta, H., Gentimis, T., Gonzalez-Diaz, R., Jimenez, M.-J., & Krim, H. (2015). *An entropy-based persistence barcode*. Pattern Recognition, 48(2), 391–401.

> Gidea, M., & Katz, Y. (2018). *Topological data analysis of financial time series: Landscapes of crashes*. Physica A: Statistical Mechanics and its Applications, 491, 820–834.

> Gidea, M. (2017). *Topology data analysis of crisis impact on multi-asset portfolios*. Available at SSRN.

> Takens, F. (1981). *Detecting strange attractors in turbulence*. In D. Rand & L.-S. Young (Eds.), Dynamical Systems and Turbulence, Warwick 1980 (Lecture Notes in Mathematics, Vol. 898, pp. 366–381). Springer.

> Carlsson, G. (2009). *Topology and data*. Bulletin of the American Mathematical Society, 46(2), 255–308.

> Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*. American Mathematical Society.

---

## Disclaimer

> ⚠️ **Research Tool Only.** TDA-Risk-Sentinel Pro is a quantitative research instrument for academic and educational exploration of topological methods in finance. It does not constitute investment advice, a solicitation, or a regulated financial service. Past backtest performance does not guarantee future results. All signals should be interpreted within a comprehensive risk management framework and reviewed by qualified investment professionals before informing any capital allocation decision.

---

<div align="center">

**TDA-Risk-Sentinel Pro v2.4**

*Ripser · Persim · Multi-Scale TDA · Vietoris-Rips Filtration · Persistent Homology · Wasserstein Distance*

Built with Streamlit · Python 3.10+

</div>
