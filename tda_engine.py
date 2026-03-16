"""
tda_engine.py — Moteur Topologique TDA-Risk-Sentinel Pro v2.3
Backend : ripser (Vietoris-Rips ultra-rapide) + persim (Wasserstein exact)

Avantages ripser vs giotto-tda :
  - Implémentation C++ de Ripser (Ulrich Bauer, 2015) — 10-50× plus rapide.
  - API directe : ripser(X) → {'dgms': [H0, H1, ...]}
  - Aucune dépendance TensorFlow / compilateur complexe.
  - persim.wasserstein() = distance de Wasserstein exacte (pas d'approximation).

Format interne des diagrammes :
  Convention giotto-tda conservée pour compatibilité avec visualizations.py :
    np.ndarray (n, 3) avec colonnes [birth, death, dim]
  Conversion depuis ripser : _ripser_to_standard(result['dgms'])

Architecture mathématique :
┌─────────────────────────────────────────────────────────────┐
│  COUCHE 1 : Prétraitement                                   │
│  Log-rendements → Z-score glissant (W, sans look-ahead)     │
├─────────────────────────────────────────────────────────────┤
│  COUCHE 2 : Reconstruction de l'Espace des Phases           │
│  Takens Embedding (d, τ) numpy-only + PCA multi-actifs      │
├─────────────────────────────────────────────────────────────┤
│  COUCHE 3 : Topologie Persistante (ripser)                  │
│  Filtration VR → Diagrammes H₀, H₁                         │
│  Seuil dynamique IQR → filtrage du bruit blanc              │
├─────────────────────────────────────────────────────────────┤
│  COUCHE 4 : Descripteurs (persim + numpy)                   │
│  • Wasserstein exact W₁(Dₜ, Dₜ₋₁)  via persim              │
│  • Entropie de Persistance  E = -Σ pᵢ log₂ pᵢ              │
│  • Amplitude H₁ maximale                                    │
├─────────────────────────────────────────────────────────────┤
│  COUCHE 5 : TSS Composite                                   │
│  TSS = 0.45·W₁ + 0.30·Amp + 0.25·(1-E)                     │
├─────────────────────────────────────────────────────────────┤
│  COUCHE 6 : Analyse de Sensibilité                          │
│  W ∈ {W-5, W, W+5}, τ ∈ {1, 2, 3} → Enveloppe TSS ± σ     │
├─────────────────────────────────────────────────────────────┤
│  COUCHE 7 : Multi-Échelles (v2.3)                           │
│  TSS_fast(W=20) ‖ TSS_slow(W=100) [parallèle joblib]        │
│  ΔTSS = TSS_fast − TSS_slow  → Divergence Topologique      │
│  P-TSS = norm(ΔTSS⁺ × (1−TSS_slow)) → "Fracture" prédictive│
│  Early Warning : ΔTSS > μ(ΔTSS) + k·σ(ΔTSS)               │
└─────────────────────────────────────────────────────────────┘

Références :
  - Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips persistence.
  - Chintakunta et al. (2015). An entropy-based persistence barcode.
  - Gidea, M. & Katz, Y. (2018). Topological data analysis of financial time series.
  - Takens, F. (1981). Detecting strange attractors in turbulence.
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ripser import ripser
from persim import wasserstein as persim_wasserstein
from sklearn.decomposition import PCA


# ═══════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TDADescriptors:
    """Conteneur pour les descripteurs topologiques d'une fenêtre.

    Attributes:
        wasserstein: Distance de Wasserstein exacte H₁ vs fenêtre précédente.
        amplitude:   Persistance maximale H₁ après filtrage dynamique.
        entropy:     Entropie de persistance H₁ (bits, base 2).
        n_h1:        Nombre de cycles H₁ stables (> seuil IQR).
        threshold:   Seuil de persistance dynamique utilisé (IQR).
    """
    wasserstein: float = 0.0
    amplitude:   float = 0.0
    entropy:     float = 0.0
    n_h1:        int   = 0
    threshold:   float = 0.0


@dataclass
class SensitivityResult:
    """Résultat de l'analyse de sensibilité du TSS.

    Attributes:
        tss_matrix: Dict {(W, τ): pd.Series} — TSS pour chaque combinaison.
        tss_mean:   pd.Series — TSS moyen (consensus des configurations).
        tss_std:    pd.Series — Écart-type (mesure de robustesse).
        tss_upper:  pd.Series — Enveloppe haute (mean + std).
        tss_lower:  pd.Series — Enveloppe basse (mean - std), clippée à 0.
    """
    tss_matrix: Dict = field(default_factory=dict)
    tss_mean:   pd.Series = field(default_factory=pd.Series)
    tss_std:    pd.Series = field(default_factory=pd.Series)
    tss_upper:  pd.Series = field(default_factory=pd.Series)
    tss_lower:  pd.Series = field(default_factory=pd.Series)


# ═══════════════════════════════════════════════════════════════════════════
# 1. PRÉTRAITEMENT
# ═══════════════════════════════════════════════════════════════════════════

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcule les log-rendements à partir d'un DataFrame de prix.

    Formule : r_t = ln(P_t / P_{t-1})

    Les log-rendements sont additifs dans le temps et symétriques,
    préférables aux rendements simples pour l'agrégation multi-actifs.

    Args:
        prices: DataFrame (T × N) de prix ajustés, index=DatetimeIndex.

    Returns:
        DataFrame (T-1 × N) de log-rendements, NaN supprimés.
    """
    return np.log(prices / prices.shift(1)).dropna()


def rolling_zscore(
    returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Standardisation Z-score glissante sans look-ahead bias.

    À chaque date t, μ et σ sont estimés sur [t-W+1, t] exclusivement.
    La standardisation uniformise les magnitudes entre actifs avant
    l'embedding, empêchant la volatilité absolue de dominer la géométrie.

    Args:
        returns: DataFrame (T × N) de log-rendements.
        window:  Taille de fenêtre W (jours de trading). Défaut = 60.

    Returns:
        DataFrame standardisé sans les W premières lignes (NaN supprimés).

    Raises:
        ValueError: Si W > len(returns) — fenêtre mathématiquement invalide.
    """
    if window > len(returns):
        raise ValueError(
            f"Fenêtre W={window} > longueur données ({len(returns)} jours). "
            "Réduisez W ou élargissez la plage de dates."
        )
    roll_mean = returns.rolling(window=window).mean()
    roll_std  = returns.rolling(window=window).std().replace(0.0, np.nan)
    return ((returns - roll_mean) / roll_std).dropna()


# ═══════════════════════════════════════════════════════════════════════════
# 2. TAKENS EMBEDDING (implémentation numpy pure)
# ═══════════════════════════════════════════════════════════════════════════

def _takens_single(x: np.ndarray, d: int, tau: int) -> np.ndarray:
    """Embedding de Takens pour une série temporelle scalaire.

    Construit la matrice de vecteurs retardés :
        Φ(t) = [x(t), x(t+τ), x(t+2τ), ..., x(t+(d-1)τ)]

    Args:
        x:   Série temporelle 1D de longueur T.
        d:   Dimension d'embedding.
        tau: Délai τ entre coordonnées.

    Returns:
        Array (T - (d-1)*tau, d) des vecteurs retardés.
    """
    n = len(x) - (d - 1) * tau
    if n <= 0:
        raise ValueError(
            f"Données insuffisantes pour Takens(d={d}, τ={tau}). "
            f"Longueur requise : {(d-1)*tau + 1}, disponible : {len(x)}."
        )
    return np.column_stack([x[i * tau: i * tau + n] for i in range(d)])


def embed_timeseries(
    z_returns: pd.DataFrame,
    dimension: int = 3,
    delay: int = 1,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Reconstruction de l'espace des phases par embedding de Takens multi-actifs.

    Algorithme :
      1. Embedding de Takens par actif → (T', d) pour chaque colonne.
      2. Concaténation horizontale → (T', d×N).
      3. PCA → projection (T', dimension) préservant la variance maximale.

    La PCA élimine la redondance inter-actifs et fournit un nuage de points
    de dimension fixe pour la filtration VR, quel que soit le nombre d'actifs.

    Args:
        z_returns: DataFrame (T × N) de rendements Z-scorés.
        dimension: Dimension d de l'embedding. Défaut = 3.
        delay:     Délai τ en jours entre coordonnées. Défaut = 1.

    Returns:
        Tuple (point_cloud, dates) :
          - point_cloud : np.ndarray float32 (T', min(d, d×N)).
          - dates : DatetimeIndex de longueur T'.

    Raises:
        ValueError: Si les données sont trop courtes pour l'embedding.
    """
    embedded_list = []

    for col in z_returns.columns:
        series = z_returns[col].values.astype(np.float64)
        try:
            emb = _takens_single(series, d=dimension, tau=delay)
            embedded_list.append(emb)
        except ValueError as e:
            warnings.warn(f"Actif {col} ignoré dans l'embedding : {e}", RuntimeWarning)

    if not embedded_list:
        raise ValueError("Aucun actif valide pour l'embedding de Takens.")

    min_len  = min(e.shape[0] for e in embedded_list)
    combined = np.hstack([e[:min_len] for e in embedded_list])  # (T', d×N)

    # PCA : réduction en `dimension` composantes
    n_components = min(dimension, combined.shape[1], combined.shape[0] - 1)
    pca = PCA(n_components=n_components)
    point_cloud = pca.fit_transform(combined).astype(np.float32)

    # Compléter à 3 colonnes minimum pour la visualisation 3D
    if point_cloud.shape[1] < 3:
        pad = np.zeros(
            (point_cloud.shape[0], 3 - point_cloud.shape[1]), dtype=np.float32
        )
        point_cloud = np.hstack([point_cloud, pad])

    # Recalculer l'index de dates correspondant
    skip = (dimension - 1) * delay          # Points perdus par l'embedding
    offset = len(z_returns) - min_len
    dates = z_returns.index[offset:]

    return point_cloud, dates


# ═══════════════════════════════════════════════════════════════════════════
# 3. CONVERSION FORMAT RIPSER → FORMAT INTERNE
# ═══════════════════════════════════════════════════════════════════════════

def _ripser_to_standard(dgms: list) -> np.ndarray:
    """Convertit la sortie ripser en format interne (n, 3) [birth, death, dim].

    ripser retourne : {'dgms': [H0_arr, H1_arr, ...]}
      - H0_arr : (n, 2) np.ndarray [birth, death], death peut être +inf.
      - H1_arr : (m, 2) np.ndarray [birth, death], valeurs finies.

    Le format interne utilisé par le reste du code :
      - np.ndarray (n+m, 3) avec colonnes [birth, death, dim].

    Args:
        dgms: Liste de diagrammes par dimension (sortie de ripser['dgms']).

    Returns:
        Array (N, 3) consolidé. Renvoie array vide (0, 3) si dgms est vide.
    """
    parts = []
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            continue
        arr = dgm.astype(np.float32)
        # Remplacer +inf par une grande valeur finie (H0 : composante principale)
        arr[~np.isfinite(arr)] = 1e6
        dim_col = np.full((len(arr), 1), dim, dtype=np.float32)
        parts.append(np.hstack([arr, dim_col]))

    return np.vstack(parts) if parts else np.zeros((0, 3), dtype=np.float32)


def _extract_h1_diagram(std_diagram: np.ndarray) -> np.ndarray:
    """Extrait les points H₁ (dim==1) du diagramme standard.

    Args:
        std_diagram: Array (n, 3) [birth, death, dim].

    Returns:
        Array (m, 2) [birth, death] pour dim==1 uniquement.
    """
    h1_mask = std_diagram[:, 2] == 1
    return std_diagram[h1_mask][:, :2]  # (m, 2)


# ═══════════════════════════════════════════════════════════════════════════
# 4. HOMOLOGIE PERSISTANTE (fenêtres glissantes, backend ripser)
# ═══════════════════════════════════════════════════════════════════════════

def compute_persistence_diagrams(
    point_cloud: np.ndarray,
    window_size: int = 60,
    step: int = 5,
    homology_dims: Tuple[int, ...] = (0, 1),
) -> Tuple[List[np.ndarray], List[int]]:
    """Calcule les diagrammes de persistance par filtration Vietoris-Rips (ripser).

    Pour chaque fenêtre temporelle [t, t+W] :
      1. Extraire le sous-nuage de W points.
      2. Appeler ripser() → diagrammes H₀, H₁.
      3. Convertir au format interne (n, 3).

    H₀ : composantes connexes (naissances à ε=0, morts quand elles fusionnent).
    H₁ : cycles 1D (boucles dans l'espace des corrélations).
    Un pic de persistance H₁ = synchronisation d'actifs = signature de stress.

    Args:
        point_cloud:    Array (T, d) du nuage embeddi.
        window_size:    Nombre de points par fenêtre. Défaut = 60.
        step:           Pas entre fenêtres consécutives. Défaut = 5.
        homology_dims:  Dimensions à calculer (0 et/ou 1). Défaut = (0, 1).

    Returns:
        Tuple (diagrams_list, window_centers) :
          - diagrams_list  : liste de np.ndarray (n, 3) [birth, death, dim].
          - window_centers : liste d'indices du milieu de chaque fenêtre.
    """
    maxdim = max(homology_dims)
    diagrams_list  = []
    window_centers = []
    T = len(point_cloud)

    for start in range(0, T - window_size + 1, step):
        end    = start + window_size
        window = point_cloud[start:end].astype(np.float64)

        # ripser : distance_matrix=False → calcule les distances L2 en interne
        result = ripser(window, maxdim=maxdim)
        std_diag = _ripser_to_standard(result["dgms"])

        diagrams_list.append(std_diag)
        window_centers.append((start + end) // 2)

    return diagrams_list, window_centers


# ═══════════════════════════════════════════════════════════════════════════
# 5. SEUIL DE PERSISTANCE DYNAMIQUE
# ═══════════════════════════════════════════════════════════════════════════

def dynamic_persistence_threshold(
    diagram: np.ndarray,
    dim: int = 1,
    method: str = "iqr",
    k: float = 1.5,
) -> float:
    """Calcule un seuil de persistance adaptatif pour filtrer le bruit blanc.

    Le bruit topologique se manifeste par de nombreuses features de faible
    persistance (birth ≈ death), artefacts de la discrétisation du nuage.

    Méthodes disponibles :
      - "iqr"    : θ = Q₁ + k·IQR  (Tukey, robuste aux outliers, k=1.5).
      - "median" : θ = médiane(persistances).
      - "pct90"  : θ = percentile 90 (conserver les 10% les plus stables).

    Args:
        diagram: Array (n, 3) [birth, death, dim].
        dim:     Dimension homologique cible. Défaut = 1 (H₁).
        method:  Méthode de seuillage. Défaut = 'iqr'.
        k:       Coefficient IQR. Défaut = 1.5.

    Returns:
        Seuil θ ≥ 0. Retourne 0.0 si diagramme vide.

    Raises:
        ValueError: Si method invalide.
    """
    pts  = diagram[diagram[:, 2] == dim]
    pers = pts[:, 1] - pts[:, 0]
    pers = pers[np.isfinite(pers) & (pers > 0)]

    if len(pers) == 0:
        return 0.0

    if method == "iqr":
        q1, q3 = np.percentile(pers, [25, 75])
        return float(q1 + k * (q3 - q1))
    elif method == "median":
        return float(np.median(pers))
    elif method == "pct90":
        return float(np.percentile(pers, 90))
    else:
        raise ValueError(f"method doit être 'iqr', 'median' ou 'pct90'. Reçu: {method!r}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. DESCRIPTEURS TOPOLOGIQUES
# ═══════════════════════════════════════════════════════════════════════════

def compute_persistence_entropy(
    diagram: np.ndarray,
    dim: int = 1,
    threshold: float = 0.0,
) -> float:
    """Entropie de Persistance — mesure de la diversité topologique.

    Définition (Chintakunta et al., 2015) :
      lᵢ = dᵢ - bᵢ  (durée de vie de la feature i)
      L  = Σ lᵢ
      pᵢ = lᵢ / L
      E  = -Σ pᵢ log₂(pᵢ)

    Interprétation financière :
      - E élevée → distribution diverse → marché hétérogène, résilient.
      - E faible  → 1 feature domine  → marché monolithique, fragile.
      La chute soudaine d'entropie précède les krachs.

    Args:
        diagram:   Array (n, 3) [birth, death, dim].
        dim:       Dimension homologique. Défaut = 1.
        threshold: Seuil de filtrage. Défaut = 0.0 (aucun).

    Returns:
        Entropie ≥ 0 (bits, base 2). Retourne 0.0 si < 2 features.
    """
    pts  = diagram[diagram[:, 2] == dim]
    pers = pts[:, 1] - pts[:, 0]
    pers = pers[np.isfinite(pers) & (pers > 0)]

    if threshold > 0:
        pers = pers[pers > threshold]

    if len(pers) < 2:
        return 0.0

    L = pers.sum()
    if L < 1e-12:
        return 0.0

    p = pers / L
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def wasserstein_distance_h1(
    diag_a: np.ndarray,
    diag_b: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """Distance de Wasserstein exacte W₁ entre deux diagrammes H₁.

    Utilise persim.wasserstein() qui implémente l'algorithme d'assignation
    optimale (Hungarian / Auction). Contrairement à l'approximation 1D
    précédente, cette implémentation est exacte pour les diagrammes 2D.

    Convention persim : les diagrammes sont passés sous forme (n, 2)
    avec [birth, death]. Les points à l'infini sont automatiquement gérés.

    Args:
        diag_a:    Diagramme A au format interne (n, 3).
        diag_b:    Diagramme B au format interne (m, 3).
        threshold: Seuil de filtrage des features de bruit.

    Returns:
        W₁ ≥ 0.0. Retourne 0.0 si l'un des diagrammes H₁ est vide.
    """
    # Extraire H1 en (n, 2)
    h1_a = _extract_h1_diagram(diag_a)
    h1_b = _extract_h1_diagram(diag_b)

    # Filtrage par seuil de persistance
    if threshold > 0:
        pers_a = h1_a[:, 1] - h1_a[:, 0]
        pers_b = h1_b[:, 1] - h1_b[:, 0]
        h1_a   = h1_a[pers_a > threshold]
        h1_b   = h1_b[pers_b > threshold]

    if len(h1_a) == 0 and len(h1_b) == 0:
        return 0.0
    if len(h1_a) == 0 or len(h1_b) == 0:
        # Distance vers le diagramme vide = somme des demi-persistances
        nonempty = h1_a if len(h1_a) > 0 else h1_b
        pers = nonempty[:, 1] - nonempty[:, 0]
        return float(np.sum(pers / 2.0))

    # persim_wasserstein attend des tableaux float64 (n, 2)
    try:
        d = persim_wasserstein(
            h1_a.astype(np.float64),
            h1_b.astype(np.float64),
            matching=False,
        )
        return float(d)
    except Exception:
        # Fallback sur l'approximation 1D si persim échoue
        pers_a = np.sort(h1_a[:, 1] - h1_a[:, 0])[::-1]
        pers_b = np.sort(h1_b[:, 1] - h1_b[:, 0])[::-1]
        n = max(len(pers_a), len(pers_b))
        a_p = np.pad(pers_a, (0, n - len(pers_a)))
        b_p = np.pad(pers_b, (0, n - len(pers_b)))
        return float(np.sum(np.abs(a_p - b_p)))


def compute_descriptors_per_window(
    diagrams_list: List[np.ndarray],
    threshold_method: str = "iqr",
) -> List[TDADescriptors]:
    """Calcule l'ensemble des descripteurs topologiques pour chaque fenêtre.

    Pour chaque diagramme :
      1. Seuil dynamique θ (IQR/median/pct90).
      2. Amplitude H₁ = max(persistance filtrée).
      3. Entropie H₁ (bits).
      4. Wasserstein vs fenêtre précédente (persim exact).

    Args:
        diagrams_list:    Liste de diagrammes (n, 3).
        threshold_method: Méthode de seuillage. Défaut = 'iqr'.

    Returns:
        Liste de TDADescriptors, même longueur que diagrams_list.
    """
    results = []

    for i, diag in enumerate(diagrams_list):
        theta = dynamic_persistence_threshold(diag, dim=1, method=threshold_method)

        # Amplitude H₁
        h1_pts   = _extract_h1_diagram(diag)
        pers_h1  = h1_pts[:, 1] - h1_pts[:, 0] if len(h1_pts) > 0 else np.array([])
        pers_h1  = pers_h1[np.isfinite(pers_h1)]
        pers_sig = pers_h1[pers_h1 > theta] if theta > 0 and len(pers_h1) > 0 else pers_h1

        amplitude = float(np.max(pers_sig))   if len(pers_sig) > 0 else 0.0
        n_h1      = int(len(pers_sig))

        # Entropie
        entropy = compute_persistence_entropy(diag, dim=1, threshold=theta)

        # Wasserstein vs fenêtre précédente
        if i == 0:
            wass = 0.0
        else:
            wass = wasserstein_distance_h1(
                diagrams_list[i - 1], diag, threshold=theta
            )

        results.append(TDADescriptors(
            wasserstein=wass,
            amplitude=amplitude,
            entropy=entropy,
            n_h1=n_h1,
            threshold=theta,
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 7. TSS COMPOSITE v2.1
# ═══════════════════════════════════════════════════════════════════════════

def compute_tss(
    descriptors: List[TDADescriptors],
    window_centers: List[int],
    weights: Tuple[float, float, float] = (0.45, 0.30, 0.25),
    smoothing_span: int = 3,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Calcule le Topological Stress Score (TSS) composite.

    Formule :
        TSS = α·W₁_norm + β·Amp_norm + γ·(1 − E_norm)

    Composantes :
      - W₁_norm      : Vitesse de changement topologique (précurseur).
      - Amp_norm     : Complexité topologique instantanée.
      - (1 − E_norm) : Collapse d'entropie → marché monolithique → stress.

    Lissage EWM anti-bruit, re-normalisation finale ∈ [0, 1].

    Args:
        descriptors:    Liste de TDADescriptors par fenêtre.
        window_centers: Indices des centres de fenêtres.
        weights:        (α, β, γ) pour Wasserstein/Amplitude/Entropie.
        smoothing_span: Span EWM. Défaut = 3.

    Returns:
        Tuple (tss_series, components_df) :
          - tss_series    : pd.Series TSS ∈ [0, 1], index = window_centers.
          - components_df : DataFrame des composantes pour visualisation.

    Raises:
        ValueError: Si < 2 fenêtres.
    """
    if len(descriptors) < 2:
        raise ValueError("Au moins 2 fenêtres nécessaires pour le TSS.")

    α, β, γ = weights

    wass_arr = np.array([d.wasserstein for d in descriptors])
    amp_arr  = np.array([d.amplitude   for d in descriptors])
    ent_arr  = np.array([d.entropy     for d in descriptors])

    def _safe_norm(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return np.zeros_like(arr) if mx <= mn else (arr - mn) / (mx - mn)

    w_norm = _safe_norm(wass_arr)
    a_norm = _safe_norm(amp_arr)
    e_norm = _safe_norm(ent_arr)

    tss_raw = α * w_norm + β * a_norm + γ * (1.0 - e_norm)

    tss_s  = pd.Series(tss_raw, index=window_centers)
    tss_sm = tss_s.ewm(span=smoothing_span, adjust=False).mean()
    tss_f  = pd.Series(_safe_norm(tss_sm.values), index=window_centers)

    components_df = pd.DataFrame({
        "wasserstein_norm":  w_norm,
        "amplitude_norm":    a_norm,
        "entropy_raw":       ent_arr,
        "entropy_norm":      e_norm,
        "entropy_collapse":  1.0 - e_norm,
        "tss_raw":           tss_raw,
        "tss_final":         tss_f.values,
        "n_h1":              [d.n_h1      for d in descriptors],
        "threshold":         [d.threshold for d in descriptors],
    }, index=window_centers)

    return tss_f, components_df


# ═══════════════════════════════════════════════════════════════════════════
# 8. ANALYSE DE SENSIBILITÉ
# ═══════════════════════════════════════════════════════════════════════════

def sensitivity_analysis(
    prices: pd.DataFrame,
    base_window: int = 60,
    base_delay: int = 1,
    window_delta: int = 5,
    delays: Tuple[int, ...] = (1, 2, 3),
    embed_dim: int = 3,
    step: int = 5,
) -> SensitivityResult:
    """Analyse de robustesse du TSS face aux variations d'hyperparamètres.

    Teste toutes les combinaisons (W, τ) ∈ {W-Δ, W, W+Δ} × {τ₁, τ₂, τ₃}
    et calcule l'enveloppe statistique mean ± std du TSS.

    Un signal robuste présente un std faible → confiance élevée.

    Args:
        prices:       DataFrame de prix ajustés.
        base_window:  Fenêtre de référence W₀. Défaut = 60.
        base_delay:   Délai de référence τ₀. Défaut = 1.
        window_delta: Δ de variation. Défaut = 5.
        delays:       Délais à tester. Défaut = (1, 2, 3).
        embed_dim:    Dimension d'embedding (fixée). Défaut = 3.
        step:         Pas entre fenêtres. Défaut = 5.

    Returns:
        SensitivityResult avec tss_matrix, mean, std, upper, lower.
    """
    windows = sorted(set([
        max(20, base_window - window_delta),
        base_window,
        base_window + window_delta,
    ]))

    tss_dict: Dict[Tuple[int, int], pd.Series] = {}

    for W in windows:
        for tau in delays:
            try:
                tss, _, _, _, _ = run_tda_pipeline(
                    prices, window=W, step=step,
                    embed_dim=embed_dim, embed_delay=tau,
                )
                tss_dict[(W, tau)] = tss
            except Exception as e:
                warnings.warn(
                    f"Sensibilité : échec W={W}, τ={tau} — {e}", RuntimeWarning
                )

    if not tss_dict:
        return SensitivityResult()

    min_len = min(len(s) for s in tss_dict.values())
    mat = np.vstack([s.values[:min_len] for s in tss_dict.values()])

    ref_key = (base_window, base_delay)
    ref_idx = tss_dict.get(ref_key, next(iter(tss_dict.values()))).index[:min_len]

    mean_tss = pd.Series(mat.mean(axis=0), index=ref_idx)
    std_tss  = pd.Series(mat.std(axis=0),  index=ref_idx)

    return SensitivityResult(
        tss_matrix=tss_dict,
        tss_mean=mean_tss,
        tss_std=std_tss,
        tss_upper=mean_tss + std_tss,
        tss_lower=(mean_tss - std_tss).clip(lower=0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 9. PIPELINES COMPLETS
# ═══════════════════════════════════════════════════════════════════════════

def run_tda_pipeline(
    prices: pd.DataFrame,
    window: int = 60,
    step: int = 5,
    embed_dim: int = 3,
    embed_delay: int = 1,
    threshold_method: str = "iqr",
    tss_weights: Tuple[float, float, float] = (0.45, 0.30, 0.25),
) -> Tuple[pd.Series, np.ndarray, pd.DatetimeIndex, List[np.ndarray], List[int]]:
    """Pipeline TDA complet — interface compacte.

    Prix → Log-rendements → Z-score(W) → Takens(d,τ)
         → ripser (VR) → Descripteurs(θ_IQR) → TSS composite

    Args:
        prices:           DataFrame (T × N) de prix ajustés.
        window:           Fenêtre Z-score et homologie.
        step:             Pas entre fenêtres TDA.
        embed_dim:        Dimension d de Takens.
        embed_delay:      Délai τ de Takens.
        threshold_method: Méthode de seuillage IQR.
        tss_weights:      (α, β, γ) pour la composition du TSS.

    Returns:
        (tss, point_cloud, dates, diagrams_list, window_centers)
    """
    returns = compute_log_returns(prices)
    z = rolling_zscore(returns, window=window)
    point_cloud, dates = embed_timeseries(z, dimension=embed_dim, delay=embed_delay)
    diagrams_list, window_centers = compute_persistence_diagrams(
        point_cloud, window_size=window, step=step
    )
    descriptors = compute_descriptors_per_window(diagrams_list, threshold_method)
    tss, _ = compute_tss(descriptors, window_centers, weights=tss_weights)
    return tss, point_cloud, dates, diagrams_list, window_centers


def run_tda_pipeline_full(
    prices: pd.DataFrame,
    window: int = 60,
    step: int = 5,
    embed_dim: int = 3,
    embed_delay: int = 1,
    threshold_method: str = "iqr",
    tss_weights: Tuple[float, float, float] = (0.45, 0.30, 0.25),
) -> Dict:
    """Pipeline TDA complet — interface étendue (tous les artefacts).

    Utilisé par le dashboard Pro pour accéder aux composantes du TSS,
    aux descripteurs bruts et aux diagrammes de persistance.

    Returns:
        Dict : tss, point_cloud, dates, diagrams_list, window_centers,
               descriptors, components_df, z_returns.
    """
    returns = compute_log_returns(prices)
    z = rolling_zscore(returns, window=window)
    point_cloud, dates = embed_timeseries(z, dimension=embed_dim, delay=embed_delay)
    diagrams_list, window_centers = compute_persistence_diagrams(
        point_cloud, window_size=window, step=step
    )
    descriptors = compute_descriptors_per_window(diagrams_list, threshold_method)
    tss, components_df = compute_tss(descriptors, window_centers, weights=tss_weights)

    return {
        "tss":            tss,
        "point_cloud":    point_cloud,
        "dates":          dates,
        "diagrams_list":  diagrams_list,
        "window_centers": window_centers,
        "descriptors":    descriptors,
        "components_df":  components_df,
        "z_returns":      z,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 10. ANALYSE DE STABILITÉ DIMENSIONNELLE
# ═══════════════════════════════════════════════════════════════════════════

def _run_single_dim(
    args: Tuple,
) -> Dict:
    """Worker isolé pour un calcul TSS à dimension d fixée.

    Conçu pour être appelé via joblib.Parallel. Reçoit tous les paramètres
    sous forme de tuple pour compatibilité avec le backend "loky" (pickle).

    Args:
        args: Tuple (prices_values, prices_index, prices_columns,
                     dim, window, step, embed_delay, tss_weights)

    Returns:
        Dict avec clés : dim, tss_mean, tss_std, tss_min, tss_max,
                         tss_series (pd.Series), error (str ou None).
    """
    import time
    (prices_values, prices_index, prices_columns,
     dim, window, step, embed_delay, tss_weights) = args

    prices = pd.DataFrame(prices_values, index=prices_index, columns=prices_columns)
    t_start = time.perf_counter()

    try:
        tss, _, _, _, _ = run_tda_pipeline(
            prices,
            window=window,
            step=step,
            embed_dim=dim,
            embed_delay=embed_delay,
            tss_weights=tss_weights,
        )
        elapsed = time.perf_counter() - t_start
        return {
            "dim":      dim,
            "tss_mean": float(tss.mean()),
            "tss_std":  float(tss.std()),
            "tss_min":  float(tss.min()),
            "tss_max":  float(tss.max()),
            "tss_series": tss,
            "elapsed":  elapsed,
            "error":    None,
        }
    except Exception as e:
        return {
            "dim":      dim,
            "tss_mean": np.nan,
            "tss_std":  np.nan,
            "tss_min":  np.nan,
            "tss_max":  np.nan,
            "tss_series": pd.Series(dtype=float),
            "elapsed":  time.perf_counter() - t_start,
            "error":    str(e),
        }


def run_sensitivity_analysis(
    prices: pd.DataFrame,
    dim_range: Optional[List[int]] = None,
    window: int = 60,
    step: int = 5,
    embed_delay: int = 1,
    tss_weights: Tuple[float, float, float] = (0.45, 0.30, 0.25),
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Analyse de stabilité dimensionnelle du TSS.

    Évalue la robustesse du signal TSS face à la variation de la dimension
    d'embedding de Takens d ∈ dim_range. Si le TSS moyen et son écart-type
    restent stables entre les dimensions, le signal est une réalité
    topologique et non un artefact du choix de d.

    Parallélisme :
      Utilise joblib.Parallel avec le backend "loky" (multi-processus)
      pour tirer parti des cœurs CPU. Ripser libère le GIL (C extension),
      mais loky est préféré car il évite les conflits de ressources Python.
      Fallback séquentiel si joblib est indisponible.

    Complexité : O(n·W³) pour Vietoris-Rips en dimension d.
      - d=2 : rapide  (~5-15s pour 5 ans)
      - d=3 : nominal (~10-30s)
      - d=4 : modéré  (~20-60s)
      - d=5 : lent    (~40-120s) — barre de progression recommandée

    Args:
        prices:      DataFrame (T × N) de prix ajustés.
        dim_range:   Liste des dimensions à tester. Défaut = [2, 3, 4, 5].
        window:      Fenêtre Z-score et homologie (jours). Défaut = 60.
        step:        Pas entre fenêtres TDA. Défaut = 5.
        embed_delay: Délai τ de Takens. Défaut = 1.
        tss_weights: (α, β, γ) pondération du TSS. Défaut = (0.45, 0.30, 0.25).
        n_jobs:      Nombre de workers parallèles. -1 = tous les cœurs.
                     1 = séquentiel (utile pour le debug).

    Returns:
        pd.DataFrame avec colonnes :
          - "Dimension d"    : valeur de d testée (int).
          - "TSS Moyen"      : moyenne du TSS sur la période.
          - "Écart-type"     : std du TSS (dispersion intra-dimension).
          - "TSS Min"        : minimum observé.
          - "TSS Max"        : maximum observé.
          - "Temps (s)"      : temps de calcul pour cette dimension.
          - "Statut"         : "✅ OK" ou "❌ Erreur: <message>".

    Raises:
        ValueError: Si dim_range est vide ou prices trop court.
    """
    if dim_range is None:
        dim_range = [2, 3, 4, 5]

    if not dim_range:
        raise ValueError("dim_range ne peut pas être vide.")

    # Sérialiser prices en primitives pour la compatibilité pickle (loky)
    args_list = [
        (
            prices.values,          # np.ndarray — picklable
            prices.index,           # DatetimeIndex
            list(prices.columns),   # list of str
            int(d),
            window,
            step,
            embed_delay,
            tss_weights,
        )
        for d in dim_range
    ]

    results: List[Dict] = []

    # Tentative de parallélisme via joblib
    try:
        from joblib import Parallel, delayed

        raw = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_run_single_dim)(args) for args in args_list
        )
        results = list(raw)

    except ImportError:
        warnings.warn(
            "joblib non disponible — exécution séquentielle.",
            RuntimeWarning,
        )
        results = [_run_single_dim(args) for args in args_list]

    except Exception as e:
        warnings.warn(
            f"joblib.Parallel a échoué ({e}) — fallback séquentiel.",
            RuntimeWarning,
        )
        results = [_run_single_dim(args) for args in args_list]

    # Construire le DataFrame de résultats
    rows = []
    for r in sorted(results, key=lambda x: x["dim"]):
        rows.append({
            "Dimension d":  r["dim"],
            "TSS Moyen":    round(r["tss_mean"], 4) if not np.isnan(r["tss_mean"]) else np.nan,
            "Écart-type":   round(r["tss_std"],  4) if not np.isnan(r["tss_std"])  else np.nan,
            "TSS Min":      round(r["tss_min"],  4) if not np.isnan(r["tss_min"])  else np.nan,
            "TSS Max":      round(r["tss_max"],  4) if not np.isnan(r["tss_max"])  else np.nan,
            "Temps (s)":    round(r["elapsed"],  1),
            "Statut":       "✅ OK" if r["error"] is None else f"❌ {r['error'][:50]}",
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 11. MULTI-SCALE TDA — Détection Prédictive des Krachs (v2.3)
# ═══════════════════════════════════════════════════════════════════════════

# Paramètres par défaut — surchargeables via run_multiscale_pipeline()
MULTISCALE_CONFIG: Dict[str, object] = {
    "W_FAST":          20,    # Fenêtre courte : micro-instabilités (jours)
    "W_SLOW":         100,    # Fenêtre longue : structure de fond (jours)
    "STEP_FAST":        5,    # Pas de glissement fenêtres fast
    "STEP_SLOW":       10,    # Pas de glissement fenêtres slow
    "EW_ACCEL_WINDOW": 20,    # Fenêtre rolling pour détection d'accélération ΔTSS
    "EW_ACCEL_K":     2.0,    # Nb de σ au-dessus de la moyenne → Early Warning
}


@dataclass
class MultiscaleTDAResult:
    """Résultat du pipeline TDA Multi-Échelles.

    Fusionne deux pipelines indépendants (fast/slow) pour construire
    la Divergence Topologique et le score prédictif P-TSS.

    Attributes:
        tss_fast:      TSS(W_fast) aligné sur les dates de prix ∈ [0, 1].
        tss_slow:      TSS(W_slow) aligné sur les dates de prix ∈ [0, 1].
        delta_tss:     ΔTSS = TSS_fast − TSS_slow ∈ [-1, 1].
                       Positif → fracture locale croissante.
        ptss:          Predictive-TSS = norm(ΔTSS⁺ × (1−TSS_slow)) ∈ [0, 1].
                       Haut uniquement quand ΔTSS > 0 ET TSS_slow encore bas.
        early_warning: True si ΔTSS > μ(ΔTSS) + k·σ(ΔTSS) → accélération.
        common_dates:  DatetimeIndex des prix (index commun aligné).
        config:        Dict des hyperparamètres utilisés.
        point_cloud_fast / _slow : nuages de points embeddi par échelle.
        dates_fast / _slow        : DatetimeIndex du nuage par échelle.
        diagrams_fast / _slow     : listes de diagrammes de persistance.
        window_centers_fast / _slow : indices des centres de fenêtres.
        descriptors_fast / _slow  : descripteurs TDADescriptors par fenêtre.
        components_fast / _slow   : DataFrames des composantes normalisées.
        z_returns_fast / _slow    : rendements Z-scorés par échelle.
    """
    tss_fast:             pd.Series
    tss_slow:             pd.Series
    delta_tss:            pd.Series
    ptss:                 pd.Series
    early_warning:        pd.Series
    common_dates:         pd.DatetimeIndex
    point_cloud_fast:     np.ndarray
    dates_fast:           pd.DatetimeIndex
    diagrams_fast:        List[np.ndarray]
    window_centers_fast:  List[int]
    descriptors_fast:     List[TDADescriptors]
    components_fast:      pd.DataFrame
    point_cloud_slow:     np.ndarray
    dates_slow:           pd.DatetimeIndex
    diagrams_slow:        List[np.ndarray]
    window_centers_slow:  List[int]
    descriptors_slow:     List[TDADescriptors]
    components_slow:      pd.DataFrame
    z_returns_fast:       pd.DataFrame
    z_returns_slow:       pd.DataFrame
    config:               Dict = field(default_factory=dict)


def _run_scale_worker(args: Tuple) -> Dict:
    """Worker picklable pour l'exécution parallèle d'une échelle TDA.

    Conçu pour joblib.Parallel (backend "loky" — multi-processus).
    Reçoit les données de prix sous forme de primitives numpy/list
    pour garantir la compatibilité pickle.

    Args:
        args: Tuple (prices_values, prices_index, prices_columns,
                     window, step, embed_dim, embed_delay,
                     threshold_method, tss_weights)

    Returns:
        Dict : tss, point_cloud, dates, diagrams_list,
               window_centers, descriptors, components_df, z_returns.

    Raises:
        Aucune exception propagée — erreurs retournées dans le dict.
    """
    (prices_values, prices_index, prices_columns,
     window, step, embed_dim, embed_delay,
     threshold_method, tss_weights) = args

    prices = pd.DataFrame(prices_values, index=prices_index, columns=prices_columns)
    returns = compute_log_returns(prices)
    z = rolling_zscore(returns, window=window)
    point_cloud, dates = embed_timeseries(z, dimension=embed_dim, delay=embed_delay)
    diagrams_list, window_centers = compute_persistence_diagrams(
        point_cloud, window_size=window, step=step
    )
    descriptors = compute_descriptors_per_window(diagrams_list, threshold_method)
    tss, components_df = compute_tss(descriptors, window_centers, weights=tss_weights)

    return {
        "tss":             tss,
        "point_cloud":     point_cloud,
        "dates":           dates,
        "diagrams_list":   diagrams_list,
        "window_centers":  window_centers,
        "descriptors":     descriptors,
        "components_df":   components_df,
        "z_returns":       z,
    }


def _align_tss_to_prices(
    tss: pd.Series,
    point_cloud_dates: pd.DatetimeIndex,
    window_centers: List[int],
    price_index: pd.DatetimeIndex,
) -> pd.Series:
    """Aligne le TSS (index entier → point_cloud) sur les dates de prix.

    Mappe chaque centre de fenêtre (entier) sur sa date dans le nuage
    embeddi, puis interpole temporellement sur l'index des prix complet.

    Args:
        tss:               pd.Series TSS avec index entier (window centers).
        point_cloud_dates: DatetimeIndex du nuage embeddi.
        window_centers:    Positions des centres de fenêtres dans le nuage.
        price_index:       DatetimeIndex cible (index des prix).

    Returns:
        pd.Series TSS daté, aligné sur price_index.
        NaN en début de période (avant la première fenêtre) → 0.0.
    """
    centers_dates = [
        point_cloud_dates[min(wc, len(point_cloud_dates) - 1)]
        for wc in window_centers
    ]
    tss_dated = pd.Series(tss.values, index=pd.DatetimeIndex(centers_dates))
    combined_idx = price_index.union(tss_dated.index).sort_values()
    tss_reindexed = tss_dated.reindex(combined_idx).interpolate(
        method="time", limit_direction="forward"
    )
    return tss_reindexed.reindex(price_index).ffill().fillna(0.0)


def compute_ptss(
    tss_fast_aligned: pd.Series,
    tss_slow_aligned: pd.Series,
    ew_accel_window: int = 20,
    ew_accel_k: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calcule la Divergence Topologique, le P-TSS et l'Early Warning.

    ΔTSS = TSS_fast − TSS_slow
      Positif → instabilité locale croît plus vite que la structure globale.
      Négatif → les deux échelles convergent (détente ou stress systémique).

    P-TSS = normalize(clip(ΔTSS, 0, ∞) × (1 − TSS_slow))
      Haut uniquement si ΔTSS > 0 ET TSS_slow encore bas.
      → "Fracture topologique" : micro-stress non encore propagé à grande
         échelle. C'est la fenêtre d'anticipation du krach (Lead Time > 0).

    Early Warning = ΔTSS > rolling_mean(ΔTSS) + k·rolling_std(ΔTSS)
      Détecte les accélérations anormales de la divergence topologique
      par rapport à son propre historique récent.

    Args:
        tss_fast_aligned: TSS(W_fast) aligné sur les dates de prix.
        tss_slow_aligned: TSS(W_slow) aligné sur les dates de prix.
        ew_accel_window:  Fenêtre rolling pour l'EW. Défaut = 20.
        ew_accel_k:       Coefficient σ pour le seuil EW. Défaut = 2.0.

    Returns:
        Tuple (delta_tss, ptss, early_warning) :
          - delta_tss    : pd.Series ∈ [-1, 1], même index que les entrées.
          - ptss         : pd.Series ∈ [0, 1] — Predictive-TSS normalisé.
          - early_warning: pd.Series[bool] — True si accélération détectée.
    """
    # Divergence Topologique
    delta_tss = tss_fast_aligned - tss_slow_aligned

    # Fracture topologique : ΔTSS positif amplifié par la stabilité apparente
    fracture = delta_tss.clip(lower=0.0) * (1.0 - tss_slow_aligned.clip(0.0, 1.0))
    mn, mx = float(fracture.min()), float(fracture.max())
    if mx > mn:
        ptss = (fracture - mn) / (mx - mn)
    else:
        ptss = pd.Series(0.0, index=fracture.index)

    # Early Warning : accélération de ΔTSS au-dessus de sa propre moyenne
    min_periods = max(3, ew_accel_window // 4)
    roll_mean = delta_tss.rolling(window=ew_accel_window, min_periods=min_periods).mean()
    roll_std  = delta_tss.rolling(window=ew_accel_window, min_periods=min_periods).std()
    ew_threshold = roll_mean + ew_accel_k * roll_std.fillna(0.0)
    early_warning = (delta_tss > ew_threshold).fillna(False)

    return delta_tss, ptss, early_warning


def run_multiscale_pipeline(
    prices: pd.DataFrame,
    w_fast: int = 20,
    w_slow: int = 100,
    step_fast: int = 5,
    step_slow: int = 10,
    embed_dim: int = 3,
    embed_delay: int = 1,
    threshold_method: str = "iqr",
    tss_weights: Tuple[float, float, float] = (0.45, 0.30, 0.25),
    ew_accel_window: int = 20,
    ew_accel_k: float = 2.0,
    n_jobs: int = 2,
) -> MultiscaleTDAResult:
    """Pipeline TDA Multi-Échelles — cœur du système de détection prédictive.

    Exécute deux pipelines TDA indépendants en parallèle (joblib.loky) :
      • Échelle rapide W_fast : capture les micro-instabilités locales.
      • Échelle lente  W_slow : représente la structure topologique de fond.

    La "fracture topologique" — ΔTSS positif avec TSS_slow encore bas —
    est le précurseur des krachs. Le P-TSS normalise cette divergence en
    un score [0, 1] utilisable comme déclencheur de stratégie.

    Parallélisme :
      joblib.Parallel(n_jobs=2, backend="loky") lance les deux échelles
      simultanément. Ripser libère le GIL (C extension), mais loky évite
      les conflits de ressources Python. Fallback séquentiel si joblib absent.

    Args:
        prices:           DataFrame (T × N) de prix ajustés (DatetimeIndex).
        w_fast:           Fenêtre courte W_fast en jours. Défaut = 20.
        w_slow:           Fenêtre longue W_slow en jours. Défaut = 100.
        step_fast:        Pas glissant pour l'échelle fast. Défaut = 5.
        step_slow:        Pas glissant pour l'échelle slow. Défaut = 10.
        embed_dim:        Dimension d de Takens (partagée). Défaut = 3.
        embed_delay:      Délai τ de Takens (partagé). Défaut = 1.
        threshold_method: Méthode de seuillage IQR. Défaut = 'iqr'.
        tss_weights:      (α, β, γ) pondération TSS. Défaut = (0.45,0.30,0.25).
        ew_accel_window:  Fenêtre rolling Early Warning. Défaut = 20.
        ew_accel_k:       Coefficient σ EW. Défaut = 2.0.
        n_jobs:           Workers parallèles (≤ 2 utile ici). Défaut = 2.

    Returns:
        MultiscaleTDAResult avec tous les signaux et artefacts.

    Raises:
        ValueError: Si w_slow ≤ w_fast, ou données trop courtes pour w_slow.
    """
    if w_slow <= w_fast:
        raise ValueError(
            f"W_slow ({w_slow}) doit être strictement > W_fast ({w_fast})."
        )
    min_required = w_slow + (embed_dim - 1) * embed_delay + 2
    if len(prices) < min_required:
        raise ValueError(
            f"Données trop courtes pour W_slow={w_slow} : "
            f"{len(prices)} jours disponibles, {min_required} requis."
        )

    # Sérialiser pour pickle (loky multi-processus)
    pv  = prices.values
    pi  = prices.index
    pc  = list(prices.columns)

    args_fast = (pv, pi, pc, w_fast, step_fast, embed_dim, embed_delay, threshold_method, tss_weights)
    args_slow = (pv, pi, pc, w_slow, step_slow, embed_dim, embed_delay, threshold_method, tss_weights)

    # Exécution parallèle des deux échelles
    try:
        from joblib import Parallel, delayed
        raw = Parallel(n_jobs=min(n_jobs, 2), backend="loky", verbose=0)(
            delayed(_run_scale_worker)(args) for args in [args_fast, args_slow]
        )
        res_fast, res_slow = raw[0], raw[1]
    except ImportError:
        warnings.warn("joblib non disponible — exécution séquentielle.", RuntimeWarning)
        res_fast = _run_scale_worker(args_fast)
        res_slow = _run_scale_worker(args_slow)
    except Exception as exc:
        warnings.warn(
            f"joblib.Parallel multiscale échoué ({exc}) — fallback séquentiel.",
            RuntimeWarning,
        )
        res_fast = _run_scale_worker(args_fast)
        res_slow = _run_scale_worker(args_slow)

    # Aligner les deux TSS sur l'index des prix
    price_index = prices.index
    tss_fast_aligned = _align_tss_to_prices(
        res_fast["tss"], res_fast["dates"], res_fast["window_centers"], price_index,
    )
    tss_slow_aligned = _align_tss_to_prices(
        res_slow["tss"], res_slow["dates"], res_slow["window_centers"], price_index,
    )

    # Divergence Topologique, P-TSS, Early Warning
    delta_tss, ptss, early_warning = compute_ptss(
        tss_fast_aligned, tss_slow_aligned,
        ew_accel_window=ew_accel_window,
        ew_accel_k=ew_accel_k,
    )

    config = {
        "W_FAST": w_fast, "W_SLOW": w_slow,
        "STEP_FAST": step_fast, "STEP_SLOW": step_slow,
        "EW_ACCEL_WINDOW": ew_accel_window, "EW_ACCEL_K": ew_accel_k,
    }

    return MultiscaleTDAResult(
        tss_fast=tss_fast_aligned,
        tss_slow=tss_slow_aligned,
        delta_tss=delta_tss,
        ptss=ptss,
        early_warning=early_warning,
        common_dates=price_index,
        point_cloud_fast=res_fast["point_cloud"],
        dates_fast=res_fast["dates"],
        diagrams_fast=res_fast["diagrams_list"],
        window_centers_fast=res_fast["window_centers"],
        descriptors_fast=res_fast["descriptors"],
        components_fast=res_fast["components_df"],
        point_cloud_slow=res_slow["point_cloud"],
        dates_slow=res_slow["dates"],
        diagrams_slow=res_slow["diagrams_list"],
        window_centers_slow=res_slow["window_centers"],
        descriptors_slow=res_slow["descriptors"],
        components_slow=res_slow["components_df"],
        z_returns_fast=res_fast["z_returns"],
        z_returns_slow=res_slow["z_returns"],
        config=config,
    )
