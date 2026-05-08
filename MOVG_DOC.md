# MOVG  Magnitude-Ordered Visibility Graph

A Python package for time series preprocessing via hierarchical graph construction and path signatures. MOVG transforms a univariate time series into a directed, magnitude-ordered graph, decomposes it into chains, and emits a fixed-length feature vector suitable for downstream regression or classification models. Each prediction made by a model trained on MOVG features can be traced back to a finite set of indices in the raw input signal.

---

## Table of Contents

1. [What MOVG does](#1-what-movg-does)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [The MOVG class  public API](#4-the-movg-class--public-api)
5. [Pipeline: how the graph is built](#5-pipeline-how-the-graph-is-built)
6. [The 20 reduced features](#6-the-20-reduced-features)
7. [Backtracing: from prediction to signal](#7-backtracing-from-prediction-to-signal)
8. [Integration with ML models](#8-integration-with-ml-models)
9. [Developer reference: file structure](#9-developer-reference-file-structure)
10. [Performance and known limitations](#10-performance-and-known-limitations)

---

## 1. What MOVG does

A standard visibility graph treats a time series as an unordered set of nodes connected by visibility relationships. MOVG adds two structural constraints:

- **Chronological ordering**  every edge is implicitly directed from the earlier index to the later index, eliminating cycles by construction.
- **Magnitude ordering**  at each chronological step, edges are accepted only if they preserve a magnitude-based hierarchy, producing a tree-like structure on top of the visibility graph.

The result is a graph in which every node has a well-defined position in time and in a magnitude hierarchy. This admits a polynomial-time chain decomposition and supports path signatures from rough path theory, which collapse a chain of arbitrary length into a fixed-size feature vector. The features extracted from this representation are stable across time series of different lengths and amplitudes, and the chain structure provides a deterministic mapping from a feature value back to specific indices in the raw signal.

MOVG was developed for predictive maintenance on bearing degradation data (NASA IMS Set 2) but is domain-general. It has been validated against FFT and wavelet preprocessing on financial, meteorological, and industrial time series.

---

## 2. Installation

The package requires Python 3.9 or later.

```bash
git clone <repo-url>
cd movg
pip install -e .
```

Optional dependency groups:

```bash
# Path signatures via the esig library (otherwise approximated)
pip install -e ".[signatures]"

# Excel and CSV export support
pip install -e ".[export]"

# Everything
pip install -e ".[all]"
```

Core dependencies are installed automatically: `numpy`, `networkx`, `numba`, `psutil`, `matplotlib`. `esig` is optional; if it is not installed, the package falls back to a closed-form approximation of the depth-3 path signature.

To verify the install:

```python
from movg import MOVG
m = MOVG([1.0, 5.0, 3.0, 8.0, 2.0, 7.0, 4.0, 6.0])
m.build()
print(m.get_depth())
```

---

## 3. Quick start

```python
import numpy as np
from movg import MOVG

# Build a synthetic series
t = np.linspace(0, 4 * np.pi, 200)
y = np.sin(t) + 0.3 * np.cos(5 * t) + 0.05 * np.random.randn(200)

# Build the magnitude-ordered visibility graph
m = MOVG(y)
m.build()

# Extract the 20-feature vector for ML
features = m.extract_features()
print(features.shape)        # (20,)
print(features.tolist())

# Inspect hierarchy
print("Tree depth:", m.get_depth())
print("Roots:", m.get_roots())
print("Children of node 0:", m.get_children(0))
```

`extract_features()` runs the entire pipeline (visibility check → hierarchy → chain decomposition → signature computation → feature collation) and returns a flat 20-element vector. This is the function you will call inside any ML preprocessing loop.

---

## 4. The MOVG class  public API

The `MOVG` class is the only object most users need to interact with. It is exported from the top-level package:

```python
from movg import MOVG
```

### Constructor

```python
MOVG(series: array_like)
```

Accepts a 1-D NumPy array, list of floats, or pandas Series. The series is stored internally as a NumPy array of float64.

### Pipeline methods

| Method | Returns | Description |
|---|---|---|
| `build()` | `self` | Construct the standard visibility graph and the chronological hierarchy in sequence. Must be called before any inspection or feature extraction. |
| `decompose(node)` | `dict` | Run chain decomposition starting from `node`. Returns the chain set, sub-paths, and per-chain signatures. Useful for backtracing a single prediction. |
| `extract_features(sample_every=1, fast_mode=False)` | `np.ndarray` shape `(20,)` | Run the full pipeline and return the 20-feature vector. `sample_every` thins the node sample for large series; `fast_mode` skips signature computation on very short chains. |

### Hierarchy inspection

| Method | Returns | Description |
|---|---|---|
| `get_depth()` | `int` | Maximum depth of the chronological hierarchy (root = level 0). |
| `get_roots()` | `list[int]` | Node indices that are roots in the hierarchy (no parent). |
| `get_parent(node)` | `int` or `None` | Parent index of `node`, or `None` if `node` is a root. |
| `get_children(node)` | `list[int]` | Child indices of `node`. |
| `get_level(node)` | `int` | Hierarchy level of `node` (depth from root). |

### Module-level functions

For users who want fine-grained control of individual pipeline stages:

```python
from movg import (
    build_standard_visibility_graph,
    build_chronological_graph,
    start_graph_decomposition,
    calculate_path_signatures,
    calculate_simple_signatures,
)
```

These are the same functions that the `MOVG` class wraps internally and have stable signatures across versions.

---

## 5. Pipeline: how the graph is built

The end-to-end pipeline runs in five stages.

### Stage 1  Visibility check

For every pair of indices $(i, j)$ with $i < j$, check whether all intermediate points $k \in (i, j)$ lie below the straight line connecting $(i, y_i)$ and $(j, y_j)$. If they do, an edge is admitted between $i$ and $j$. This is the standard Lacasa visibility criterion.

The implementation uses a Numba-compiled batched routine (`visibility_utils.check_visibility_batch_fast`) for the inner loop and an intermediate lookup table to avoid repeated re-evaluation during later stages. Visibility-check cost is $\mathcal{O}(n^2)$ in the worst case for a series of length $n$ and dominates end-to-end runtime for large $n$.

### Stage 2  Chronological hierarchy

The visibility graph is converted into a directed, hierarchically ordered structure by accepting an edge $(i, j)$ only if it respects both the time ordering ($i < j$) and a magnitude-based admission rule. The resulting structure is a forest of trees rooted at the local magnitude maxima of the visibility graph.

This stage produces three lookup tables: `parent_map` (node → parent), `children_map` (node → children), and `level_map` (node → depth from root). All three are exposed via the public API.

### Stage 3  Chain decomposition

A chain in MOVG is a path in the hierarchy that is monotonic in magnitude. The decomposition routine `start_graph_decomposition` finds all maximal monotonic chains starting from a chosen node and enumerates all sub-chains by progressive truncation. Sub-chain enumeration is bounded by the chain length and runs in $\mathcal{O}(k \cdot L^2)$ for $k$ chains of average length $L$.

### Stage 4  Signature computation

For each chain, the path signature is computed as a depth-3 truncation in two dimensions (time and amplitude). This produces a 15-component tensor, of which three scalar projections are retained as features:

- **`sig[2]`**  first-order term, equivalent to net amplitude change along the chain.
- **`sig[4]`**  second-order time-amplitude cross term, equivalent to a Lévy-area-like signed quantity.
- **`sig[8]`**  third-order time-weighted acceleration term.

The lift from one to two dimensions is essential. A signature computed on the amplitude alone collapses to powers of the net increment, which discards all information about the path's geometry; the lift to $(t, y)$ recovers it.

If `esig` is installed, signatures are computed via `roughpy.LieIncrementStream`. If not, the package falls back to a closed-form approximation of the same three projections.

### Stage 5  Feature collation

The chain-level signatures are aggregated alongside structural descriptors of the graph (degree statistics, spectral properties, edge geometry, level-wise amplitude statistics, hierarchy composition ratios) and the result is reduced via the offline-curated 20-feature selection described in Section 6.

---

## 6. The 20 reduced features

The full MOVG extraction pipeline produces 155 candidate structural features across 14 categories (root statistics, leaf statistics, internal-node statistics, level-wise statistics, depth distributions, branching factors, chain-level signature projections, edge geometry, degree distributions, spectral graph properties, clustering and centrality, hierarchy composition, fraction-based ratios, and aggregate counts). The 20-feature reduction was obtained by Pearson correlation filtering, variance-inflation-factor screening, and domain-driven curation.

Each row of `movg_features_reduced.csv` corresponds to one analysis window (e.g.\ one IMS bearing file or one lookback window in a sliding-window setup) and contains exactly these 20 columns:

| # | Feature | Group | Meaning |
|---|---|---|---|
| 1 | `root_amp_cv` | Root statistics | Coefficient of variation of amplitudes across all root nodes (hierarchy-level-0 maxima). High values indicate inhomogeneous peak structure. |
| 2 | `internal_amp_min` | Internal statistics | Minimum amplitude across all internal (non-root, non-leaf) nodes. Sensitive to the lowest non-trivial structural element. |
| 3 | `leaf_amp_range` | Leaf statistics | Peak-to-peak amplitude range across all leaf nodes (terminal nodes in the hierarchy). |
| 4 | `subtree_sig8_std` | Signature aggregates | Standard deviation of the third-order signature projection (`sig[8]`) computed over all sub-chains in the hierarchy. Captures heterogeneity of time-weighted acceleration. |
| 5 | `level2_std_amp` | Level-wise statistics | Standard deviation of node amplitudes restricted to hierarchy level 2. Captures structure at a fixed depth from the root. |
| 6 | `edge_len_min` | Edge geometry | Minimum edge length in the visibility graph (Euclidean distance between connected node coordinates in the $(t,y)$ plane). |
| 7 | `sig_10` | Signature components | Tenth component of the depth-3 2-D path signature on the global path. Third-order $tyy$ component. |
| 8 | `leaf_spacing_cv` | Leaf statistics | Coefficient of variation of inter-leaf time gaps. Captures regularity of terminal events. |
| 9 | `amp_ratio_internal_leaf` | Composition | Ratio of mean internal-node amplitude to mean leaf amplitude. A scale-free indicator of how the hierarchy compresses amplitude with depth. |
| 10 | `diff_deg_entropy` | Degree distribution | Shannon entropy of the node-degree distribution computed from successive-window differences. Captures change in degree heterogeneity. |
| 11 | `spectral_second` | Spectral | Second eigenvalue of the graph Laplacian. The Fiedler value, indicative of algebraic connectivity. |
| 12 | `sig_12` | Signature components | Twelfth component of the depth-3 2-D path signature on the global path. Third-order $yty$ component. |
| 13 | `edge_angle_max` | Edge geometry | Maximum edge slope in the visibility graph, expressed as an angle. Sensitive to extreme local gradients. |
| 14 | `frac_high_branch` | Branching | Fraction of internal nodes whose out-degree exceeds the branching-factor median. A robust measure of branching concentration. |
| 15 | `diff_clustering` | Clustering | Difference of the average clustering coefficient between successive analysis windows. Captures change in local connectivity. |
| 16 | `closeness_mean` | Centrality | Mean closeness centrality over all nodes. Reflects how reachable a typical node is from the rest. |
| 17 | `deg_skew` | Degree distribution | Skewness of the node-degree distribution. |
| 18 | `edge_angle_std` | Edge geometry | Standard deviation of edge slopes (in angle). Captures geometric heterogeneity of the visibility relation. |
| 19 | `deg_entropy_norm` | Degree distribution | Shannon entropy of the node-degree distribution, normalised by $\log_2$ of the node count. Bounded in $[0, 1]$. |
| 20 | `ratio_leaves` | Composition | Ratio of leaf-node count to total node count. Captures the "bushiness" of the hierarchy. |

The reduction preserves the mean absolute Pearson correlation with the regression target within tolerance compared with the full 155-feature set, while removing the highly collinear features that inflated certain models' apparent performance during early experiments.

### Feature provenance

| Stage | Count |
|---|---|
| Full structural extraction | 155 |
| After Pearson \|r\| ≥ 0.30 filter | ~75 |
| After VIF ≥ 5 removal | ~50 |
| After domain curation across 14 categories | **20** |

The full 155-feature column list is available in `movg_features.csv` for users who want to run their own reduction. The file `movg_features_reduced.csv` ships the 20 columns above.

---

## 7. Backtracing: from prediction to signal

The defining design property of MOVG is that any prediction made by a downstream model can be mapped back to a finite set of time indices in the input signal. The mapping is deterministic given a fixed feature pipeline and proceeds in three steps.

1. **Prediction → contributing features.** Apply SHAP (or any attribution method) to the trained model on a chosen test window. The output is a ranked list of features, each annotated with its contribution to the prediction.
2. **Contributing features → contributing chains.** Each of the 20 features in Section 6 is a deterministic function of a specific subset of MOVG chains (or of a global graph property derivable from the chains). The package exposes this mapping via `MOVG.decompose(node)`, which returns the chains and their constituent node indices.
3. **Contributing chains → time indices.** Each chain is a sequence of node indices, and each node index corresponds directly to an index in the raw input signal. The output is therefore a list of $(t_i, y_i)$ pairs that the model effectively used.

Programmatically:

```python
m = MOVG(window)
m.build()
result = m.decompose(node_of_interest)

print(result["chains"])           # list of node-index sequences
print(result["signatures"])       # per-chain signature dict
print(result["time_indices"])     # raw-signal indices for each chain
```

The decomposition output can be exported to JSON (`movg.export.export_node_json_with_signatures`) for offline inspection or for embedding in a dashboard.

---

## 8. Integration with ML models

MOVG features are scalar, fixed-length, and do not require any model-specific preprocessing beyond the standard z-scoring used for tree-based and kernel-based regressors. The package has been validated with the following downstream models:

- **Tree-based**  XGBoost (`tree_method='hist'`), LightGBM, RandomForest. These models gain consistently and modestly from the MOVG representation.
- **Kernel-based**  Support Vector Regression (RBF kernel), Gaussian Process Regression. These models gain most from the structured feature space, since they cannot independently discover hierarchical structure from a raw window.
- **Linear**  Ridge regression. Performs surprisingly well on MOVG features once highly collinear features are removed.
- **Distance-based**  KNN performs poorly on MOVG features, since the reduced 20-feature space is not isotropic in unweighted Euclidean distance and KNN cannot down-weight features automatically.

A typical training loop:

```python
import numpy as np
from movg import MOVG
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Slide a 96-sample lookback window across the training series
X, y = [], []
for i in range(96, len(series) - 1):
    window = series[i - 96 : i]
    m = MOVG(window)
    m.build()
    X.append(m.extract_features())
    y.append(series[i + 1])
X = np.vstack(X)
y = np.array(y)

# Z-score and train
scaler = StandardScaler().fit(X)
model = XGBRegressor(tree_method="hist", n_estimators=500).fit(scaler.transform(X), y)
```

Walk-forward cross-validation with periodic retraining is recommended for non-stationary series; see the `examples/` directory.

---

## 9. Developer reference: file structure

The package is organised as follows:

```
movg/
├── __init__.py            # Public API exports: MOVG class + module functions
├── core.py                # MagnitudeOrderedVisibilityGraph class (aliased as MOVG)
├── visibility.py          # Standard visibility graph construction
├── visibility_utils.py    # Numba-compiled visibility kernels and lookup tables
├── hierarchy.py           # Chronological + magnitude hierarchy construction
├── decomposition.py       # Chain decomposition, sub-chain enumeration
├── signatures.py          # Path signature computation (esig + closed-form fallback)
├── export.py              # JSON / CSV export, decomposition log generation
├── viz.py                 # Plotting helpers (matplotlib)
└── unified_movg_pipeline.py  # End-to-end pipeline wrapper used by the test suite

pyproject.toml             # Package metadata, dependencies, optional extras
README.md                  # Top-level overview (this file is a longer companion)
tests/                     # 45 unit tests covering all public methods
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `core.py` | Defines the `MagnitudeOrderedVisibilityGraph` class. Holds the input series, the visibility graph, the hierarchy, and the lookup tables. All public-API methods live here as thin wrappers around the worker modules. |
| `visibility.py` | Implements `build_standard_visibility_graph(series) -> nx.Graph`. Uses the Lacasa visibility criterion. |
| `visibility_utils.py` | Numba JIT-compiled inner loops: `check_single_visibility_fast`, `check_visibility_batch_fast`, `find_visible_neighbors_fast`. Also exposes the visibility lookup table used to avoid repeated re-evaluation. |
| `hierarchy.py` | Implements `build_chronological_graph(visibility_graph, series) -> dict` returning the parent / children / level maps. |
| `decomposition.py` | Implements `start_graph_decomposition(node, ...) -> dict` returning chains, sub-chains, and per-chain signatures. The decomposition is non-recursive and uses an explicit stack. |
| `signatures.py` | Implements `calculate_path_signatures(path) -> dict` with two backends: `roughpy.LieIncrementStream` if installed, otherwise a closed-form depth-3 2-D approximation. |
| `export.py` | JSON / CSV / Excel export of decomposition results and feature vectors. Handles NumPy → native-Python type conversion to avoid `int64` JSON errors. |
| `viz.py` | Matplotlib helpers for plotting the visibility graph, the hierarchy, and the chain decomposition. Not used in the production pipeline. |

### Extension points

The package was built with the following extensions in mind:

- **Multivariate input.** The hierarchy module is the only stage that depends on a single magnitude ordering. A multivariate extension via multiplex visibility graphs would replace `hierarchy.py` and leave the rest of the pipeline unchanged.
- **GPU visibility check.** `visibility_utils.py` is the runtime bottleneck. A CUDA-backed reimplementation using a dense pairwise comparison kernel is the principal performance target.
- **Alternative signature transforms.** `signatures.py` exposes a single function. Replacing the depth-3 2-D signature with a logsignature, a higher depth, or a different lift only requires changing this module.

---

## 10. Performance and known limitations

### Computational cost

For a series of length $n$:

| Stage | Time | Space |
|---|---|---|
| Visibility check | $\mathcal{O}(n^2)$ worst | $\mathcal{O}(n^2)$ |
| Hierarchy construction | $\mathcal{O}(n^2)$ | $\mathcal{O}(n)$ |
| Chain decomposition | $\mathcal{O}(k \cdot n)$ | $\mathcal{O}(k \cdot L)$ |
| Sub-chain enumeration | $\mathcal{O}(k \cdot L^2)$ | $\mathcal{O}(k \cdot L^2)$ |
| Signature computation | $\mathcal{O}(L \cdot 2^m)$ | $\mathcal{O}(2^m)$ |

where $k$ is the number of maximal chains, $L$ is the average chain length, and $m$ is the signature depth (default 3). The horizontal-visibility variant of the algorithm admits an $\mathcal{O}(n \log n)$ construction but discards the slope-based ordering on which the magnitude hierarchy depends, so it cannot be used as a drop-in replacement.

### Practical guidance

For series of length up to $\sim 1{,}000$ samples, end-to-end runtime is sub-second on modern CPUs. For series of length $\sim 5{,}000$, runtime is a few seconds. For series of length $\sim 15{,}000$, the windowed pipeline (chunks of 500 samples with overlap) is recommended and has been validated to produce identical algorithmic output. For longer series, a sliding-window approach (e.g.\ 96-sample lookback) is the standard pattern and is what all dissertation results use.

### Known limitations

1. **Univariate only.** The hierarchy assumes a single magnitude ordering. Multivariate inputs must be reduced to a single channel before processing.
2. **CPU only.** The current implementation runs on `NetworkX`, which has no GPU support. The visibility-check stage is the principal bottleneck.
3. **Signature-library version sensitivity.** The optional `esig` dependency has a known bug on Python 3.13 Windows (row/index mismatch); the package detects this and falls back to the closed-form approximation. The two paths produce the same three signature projections to numerical tolerance.
4. **No native streaming mode.** The `MOVG` class operates on a complete window. Online / streaming extensions exist as a separate `dynamic_*` module set; see the developer documentation in that module for details.

---

## License

MIT. See `LICENSE`.

## Citing

If you use MOVG in academic work, please cite the dissertation:

> Sahas Talasila. *Magnitude-Ordered Visibility Graphs for Interpretable Time Series Preprocessing*. BEng dissertation, Newcastle University, 2026.
