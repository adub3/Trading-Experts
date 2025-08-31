# diagnostics_selfcontained.py
import numpy as np
import pandas as pd
import itertools
from typing import Dict, List, Tuple, Optional, Callable, Sequence
import matplotlib.pyplot as plt  # plotting is optional

def _safe_zscore(df: pd.DataFrame) -> pd.DataFrame:
    std = df.std(ddof=0).replace(0, np.nan)
    z = (df - df.mean()) / std
    return z.fillna(0.0)

def _build_pred_df_from_fn(
    experts: Sequence[object],
    prices: pd.Series | pd.DataFrame,
    predict_fn: Callable[[Sequence[object], pd.Series | pd.DataFrame, int], Dict[str, float]],
    start: int = 0,
    end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calls predict_fn(experts, prices, t) for t in [start, end) and builds a T×N DataFrame of signals.
    predict_fn must return a dict: {expert_name: signal in [-1,1] or R}.
    """
    if end is None:
        end = len(prices)
    records = []
    index = []
    for t in range(start, end):
        preds = predict_fn(experts, prices, t)  # {name: signal}
        if not isinstance(preds, dict) or not preds:
            continue
        records.append(preds)
        # index handling: try to use prices.index if available, else t
        if hasattr(prices, "index"):
            idx_val = prices.index[t]
        else:
            idx_val = t
        index.append(idx_val)
    if not records:
        raise ValueError("No predictions were produced by predict_fn over the given range.")
    pred_df = pd.DataFrame.from_records(records, index=index).sort_index()
    # Ensure consistent column ordering (by name)
    pred_df = pred_df.reindex(sorted(pred_df.columns), axis=1)
    return pred_df

def diagnose_expert_redundancy_selfcontained(
    pred_df: Optional[pd.DataFrame] = None,
    *,
    experts: Optional[Sequence[object]] = None,
    prices: Optional[pd.Series | pd.DataFrame] = None,
    predict_fn: Optional[Callable[[Sequence[object], pd.Series | pd.DataFrame, int], Dict[str, float]]] = None,
    build_start: int = 0,
    build_end: Optional[int] = None,
    corr_method: str = "pearson",
    cluster_threshold: float = 0.75,
    top_k_per_cluster: int = 1,
    prefer: Optional[pd.Series] = None,  # e.g., per-expert Sharpe; higher is better
    standardize: bool = True,
    plot: bool = True,
) -> Dict[str, object]:
    """
    Self-contained redundancy diagnostic.
    Use either:
      - Mode A: provide pred_df (T×N DataFrame of signals)
      - Mode B: provide (experts, prices, predict_fn) and it will build pred_df internally

    Returns a dict with:
      corr, avg_abs_corr, clusters, kept, dropped, summary, signals (the DataFrame used)
    """
    if pred_df is None:
        if experts is None or prices is None or predict_fn is None:
            raise ValueError("Provide either pred_df OR (experts, prices, predict_fn).")
        pred_df = _build_pred_df_from_fn(experts, prices, predict_fn, build_start, build_end)

    # Clean/sanitize
    df = pred_df.copy()
    df = df.loc[:, df.columns.notnull()]
    df = df.dropna(axis=1, how="all").fillna(0.0)

    Z = _safe_zscore(df) if standardize else df.copy()

    # Correlation
    corr = Z.corr(method=corr_method)

    # Average absolute correlation (excluding diagonal)
    n = len(corr)
    if n == 0:
        raise ValueError("No experts (columns) found in prediction DataFrame.")
    mask = ~np.eye(n, dtype=bool)
    avg_abs_corr = (corr.abs().where(mask).sum() / mask.sum(axis=0))
    avg_abs_corr.name = "avg_abs_corr"

    # Build clusters via simple union-find using |corr| >= threshold
    names = list(corr.columns)
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in itertools.combinations(names, 2):
        if abs(corr.loc[i, j]) >= cluster_threshold:
            union(i, j)

    clusters: Dict[str, List[str]] = {}
    for n_ in names:
        r = find(n_)
        clusters.setdefault(r, []).append(n_)

    # Pick representatives within each cluster
    kept: List[str] = []
    dropped: List[str] = []
    for root, members in clusters.items():
        if len(members) <= top_k_per_cluster:
            kept.extend(members)
            continue

        if prefer is not None:
            pref = prefer.reindex(members).fillna(-np.inf)
            # tiebreaker: less redundant first (lower avg abs corr), then name
            red = -avg_abs_corr.reindex(members).fillna(0.0)
            order = (
                pd.DataFrame({"pref": pref, "red": red})
                .sort_values(["pref", "red",], ascending=[False, False])
                .index
            )
        else:
            # pick least redundant first
            order = pd.Index(members)[np.argsort(avg_abs_corr.reindex(members).values)]

        keep_these = list(order[:top_k_per_cluster])
        drop_these = [m for m in members if m not in keep_these]
        kept.extend(keep_these)
        dropped.extend(drop_these)

    kept = list(dict.fromkeys(kept))
    dropped = [d for d in names if d not in kept]

    # Summary table
    summary = pd.DataFrame({
        "avg_abs_corr": avg_abs_corr.reindex(names).values,
        "cluster_id": [find(nm) for nm in names],
        "kept": [nm in kept for nm in names],
    }, index=names).sort_values(["kept", "avg_abs_corr"], ascending=[False, True])

    # Optional heatmap
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr.values, interpolation="nearest", aspect="auto")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.set_title("Expert Signal Correlation")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()

    return {
        "signals": df,           # signals actually used
        "corr": corr,
        "avg_abs_corr": avg_abs_corr,
        "clusters": clusters,
        "kept": kept,
        "dropped": dropped,
        "summary": summary,
    }
