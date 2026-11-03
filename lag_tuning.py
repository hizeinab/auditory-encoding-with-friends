"""Choose LAGS from the TRAINING episodes only (leave-one-episode-out). Never look at the test episode.

Two tools:
  lag_profile_cv   one model per single lag -> how well does the sound d TRs ago predict the brain now?
                   (shows WHERE the hemodynamic response sits in your data)
  compare_lag_sets full models with several candidate lag sets -> which set predicts best on held-out
                   training episodes, and what is the simplest set that is statistically as good?

Inputs are "raw folds": a list of folds (one per episode); each fold is a list of clips
(X_layer (T, C), Y (T, V), n_tokens (T,)), exactly what data.load_subject returns for one layer.

FAIRNESS (changed from the notebook version): every candidate is scored on the SAME rows, and none of them sees a
zero-padded lag block. That needs the number of dropped leading rows (`drop_first`) to be at least the longest lag of
ANY candidate. It is therefore computed here from the candidates (not taken from the current config.LAGS, which
tuning is supposed to decide), and returned in every result row for the record.
"""
import numpy as np

from data import lag_features, row_mask
from ridge import pearson_multi, ridge_predict


def _design(fold, lags, drop_first):
    parts = [(lag_features(X, lags)[row_mask(n, drop_first)], Y[row_mask(n, drop_first)]) for X, Y, n in fold]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _fold_r(raw_folds, lags, alphas, backend, drop_first):
    """Held-out r for every fold: (n_folds, n_alpha, V). Train on the other episodes, predict the held-out one."""
    designs = [_design(f, lags, drop_first) for f in raw_folds]
    out = []
    for k in range(len(designs)):
        Xtr = np.concatenate([d[0] for j, d in enumerate(designs) if j != k])
        Ytr = np.concatenate([d[1] for j, d in enumerate(designs) if j != k])
        out.append(pearson_multi(ridge_predict(Xtr, Ytr, designs[k][0], alphas, backend), designs[k][1]))
    return np.stack(out)


def lag_profile_cv(raw_folds, lags=range(-2, 13), alpha=1e4, backend="auto", drop_first=None):
    """{lag: r per parcel (V,)}, averaged over held-out training episodes.
    d > 0 uses sound from d TRs ago (causal). Because neighbouring TRs of the features are similar, d <= 0 is NOT
    expected to be zero: read the PEAK position and the asymmetry (r at +d clearly above r at -d), not "lag 0 ~ 0".
    drop_first=None -> the longest lag, so every lag is scored on the same rows with no zero-padded block."""
    lags = list(lags)
    k = max(max(lags), 0) if drop_first is None else drop_first
    return {d: _fold_r(raw_folds, [d], [alpha], backend, k)[:, 0].mean(0) for d in lags}


def compare_lag_sets(raw_folds, lag_sets, alphas, top_frac=0.10, backend="auto", drop_first=None):
    """Score candidate lag sets by leave-one-episode-out CV on the training episodes.

    lag_sets : {name: list of lags}
    drop_first=None -> the longest lag among ALL candidates (same rows for every set, no zero-padded block).
    Per parcel, alpha is picked from the fold-averaged r (the same rule for every set, so the comparison is fair;
    the scores are slightly optimistic in absolute terms, not in their ranking).
    Score = mean held-out r over the `top_frac` most responsive parcels (the same parcels for every set).
    Returns (rows, chosen_name): rows has mean_r, se (over folds), per_fold, n_lags, drop_first;
    chosen = the smallest set whose mean is within 1 SE of the best (1-SE rule: prefer the simpler model).
    Ties in size are broken by the smaller maximum lag (a convention, not evidence).
    """
    k = max(max(v) for v in lag_sets.values()) if drop_first is None else drop_first
    R = {name: _fold_r(raw_folds, list(lags), alphas, backend, k) for name, lags in lag_sets.items()}
    V = next(iter(R.values())).shape[2]
    best_mean = np.max(np.stack([r.mean(0).max(0) for r in R.values()]), 0)
    resp = np.argsort(best_mean)[-max(1, int(top_frac * V)):]
    rows = []
    for name, r in R.items():
        a = r.mean(0).argmax(0)
        per_fold = r[:, a, np.arange(V)][:, resp].mean(1)
        rows.append({"name": name, "lags": list(lag_sets[name]), "n_lags": len(lag_sets[name]),
                     "mean_r": float(per_fold.mean()),
                     "se": float(per_fold.std(ddof=1) / np.sqrt(len(per_fold))) if len(per_fold) > 1 else 0.0,
                     "per_fold": per_fold, "drop_first": int(k)})
    best = max(rows, key=lambda x: x["mean_r"])
    ok = [x for x in rows if x["mean_r"] >= best["mean_r"] - best["se"]]
    choice = min(ok, key=lambda x: (x["n_lags"], max(x["lags"])))
    return rows, choice["name"]
