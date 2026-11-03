"""Step 3b: ridge regression with many alphas solved at once via SVD.

Works with torch (GPU if available, much faster) or plain numpy (CPU fallback; also used by tests).
Math: standardise with TRAIN statistics, X = U S V^T, weights = V diag(s / (s^2 + alpha)) U^T Y,
so one SVD gives the solution for every alpha.
"""
import numpy as np

try:
    import torch
except ImportError:        # numpy-only environment
    torch = None


def _standardize(Xtr, Ytr, Xte):
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-6
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-6
    return (Xtr - xm) / xs, (Ytr - ym) / ys, (Xte - xm) / xs


def _ridge_numpy(Xtr, Ytr, Xte, alphas):
    U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
    UtY, XteV = U.T @ Ytr, Xte @ Vt.T
    return np.stack([XteV @ ((S / (S ** 2 + a))[:, None] * UtY) for a in alphas])


def _ridge_torch(Xtr, Ytr, Xte, alphas):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr, Xte = (torch.as_tensor(a, device=dev) for a in (Xtr, Ytr, Xte))
    U, S, Vt = torch.linalg.svd(Xtr, full_matrices=False)
    UtY, XteV = U.T @ Ytr, Xte @ Vt.T
    out = [(XteV @ ((S / (S ** 2 + a))[:, None] * UtY)).cpu().numpy() for a in alphas]
    del U, S, Vt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return np.stack(out)


def ridge_predict(Xtr, Ytr, Xte, alphas, backend="auto"):
    """Fit on train, predict test for every alpha. -> (n_alpha, n_test, n_targets) float32 (standardised units)."""
    Xtr, Ytr, Xte = (np.asarray(a, dtype=np.float32) for a in (Xtr, Ytr, Xte))
    Xtr, Ytr, Xte = _standardize(Xtr, Ytr, Xte)
    use_torch = torch is not None and backend in ("auto", "torch")
    if backend == "torch" and torch is None:
        raise ImportError("torch backend requested but torch is not installed")
    P = (_ridge_torch if use_torch else _ridge_numpy)(Xtr, Ytr, Xte, alphas)
    return P.astype(np.float32)


def pearson_multi(P, Y):
    """Column-wise Pearson r. P (A, M, V), Y (M, V) -> (A, V)."""
    Pc = P - P.mean(1, keepdims=True)
    Yc = Y - Y.mean(0)
    num = np.einsum("amv,mv->av", Pc, Yc)
    den = np.linalg.norm(Pc, axis=1) * np.linalg.norm(Yc, axis=0)[None] + 1e-8
    return num / den


def cv_then_test(train_folds, test, alphas, backend="auto"):
    """train_folds: list of folds; each fold is a list of (X, Y) clips (e.g. one fold per episode).
    test: (X, Y). Returns {'r_test', 'best_alpha', 'cv_r'}, each of shape (n_targets,).

    1) leave-one-fold-out CV on the training folds -> mean r per (alpha, target)
    2) best alpha per target
    3) refit on all training data, evaluate ONCE on the test clip
    """
    V = train_folds[0][0][1].shape[1]
    cv_r = np.zeros((len(alphas), V))
    cv_folds = np.zeros((len(train_folds), len(alphas), V), dtype=np.float32)   # every fold's held-out r
    for k in range(len(train_folds)):
        tr = [c for j, f in enumerate(train_folds) if j != k for c in f]
        Xtr, Ytr = np.concatenate([c[0] for c in tr]), np.concatenate([c[1] for c in tr])
        Xva, Yva = (np.concatenate([c[i] for c in train_folds[k]]) for i in (0, 1))
        cv_folds[k] = pearson_multi(ridge_predict(Xtr, Ytr, Xva, alphas, backend), Yva)
        cv_r += cv_folds[k] / len(train_folds)

    best = cv_r.argmax(0)
    Xall = np.concatenate([c[0] for f in train_folds for c in f])
    Yall = np.concatenate([c[1] for f in train_folds for c in f])
    r_all = pearson_multi(ridge_predict(Xall, Yall, test[0], alphas, backend), test[1])
    return {"r_test": r_all[best, np.arange(V)],
            "best_alpha": np.asarray(alphas)[best],
            "cv_r": cv_r.max(0),
            "cv_r_folds": cv_folds[:, best, np.arange(V)]}      # (K, V): per-fold held-out r at the chosen alpha