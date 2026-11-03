"""Step 3a: load fMRI, align with BEATs features, build lagged design matrices."""
import numpy as np

from config import (ATLAS_DIRNAME, ATLAS_GLOB, ATLAS_SHAPE, D_MODEL,
                    DROP_FIRST_TRS, DROP_LAST_TRS, FEAT_DIR, FMRI_DIR,
                    MIN_TOKENS_PER_TR, N_LAYERS, N_PARCELS, FMRI_GLOB)


def find_clips(season: int, episode: int, feat_dir=FEAT_DIR):
    """Parts of an episode (a, b, ...) that have features, e.g. ['friends_s02e01a', 'friends_s02e01b']."""
    stems = sorted(p.stem for p in feat_dir.glob(f"friends_s{season:02d}e{episode:02d}[a-z].npz"))
    if not stems:
        raise FileNotFoundError(f"no feature files for s{season:02d}e{episode:02d} in {feat_dir}")
    return stems


def fmri_path(subject: str):
    hits = sorted((FMRI_DIR / subject / "func").glob(FMRI_GLOB))
    if not hits:
        raise FileNotFoundError(f"no Friends Schaefer-1000 h5 for {subject} in {FMRI_DIR}")
    if len(hits) > 1:
        raise RuntimeError(f"multiple Friends Schaefer-1000 h5 files for {subject}; refusing to choose arbitrarily: {hits}")
    return hits[0]


def atlas_path(subject: str):
    """Find the unique subject-specific Schaefer-1000 NIfTI atlas file."""
    folder = FMRI_DIR / subject / ATLAS_DIRNAME
    hits = sorted(folder.glob(ATLAS_GLOB)) if folder.exists() else []
    if not hits:
        raise FileNotFoundError(
            f"No subject atlas matching {ATLAS_GLOB!r} for {subject} in {folder}. "
            "Run notebook 01's atlas-only acquisition step."
        )
    if len(hits) > 1:
        raise RuntimeError(
            f"Multiple atlas files found for {subject}; refusing to choose arbitrarily: {hits}. "
            "Narrow config.ATLAS_GLOB."
        )
    return hits[0]


def check_atlas(parcel_vol: np.ndarray, n_parcels: int = N_PARCELS, expected_shape=None) -> bool:
    """Validate a 3D integer-valued parcel-label volume (zero is background).

    ``expected_shape`` is optional so synthetic volumes can be tested. The
    project-specific loader below checks the dataset-documented atlas shape.
    """
    volume = np.asarray(parcel_vol)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D atlas volume, got shape {volume.shape}")
    if expected_shape is not None and tuple(volume.shape) != tuple(expected_shape):
        raise ValueError(f"Expected atlas shape {tuple(expected_shape)}, got {volume.shape}")
    if not np.issubdtype(volume.dtype, np.integer):
        if not np.isfinite(volume).all() or not np.allclose(volume, np.rint(volume)):
            raise ValueError("Atlas contains non-finite or non-integer parcel labels")
    values = np.unique(volume)
    if values.size == 0 or np.min(values) < 0:
        raise ValueError("Atlas labels must be non-negative; zero is reserved for background")
    positive = values[values > 0]
    if positive.size == 0:
        raise ValueError("Atlas contains no positive parcel labels")
    if positive.min() < 1 or positive.max() > int(n_parcels):
        raise ValueError(
            f"Atlas labels must be between 1 and {n_parcels}; observed range "
            f"{positive.min()}–{positive.max()}"
        )
    return True


def load_atlas(subject: str):
    """Load and validate one subject's atlas.

    Returns ``(parcel_volume, affine, path)``. Parcel IDs remain integer-valued;
    the affine is the NIfTI voxel-to-world transform and should be retained for
    all spatial visualizations or NIfTI exports.
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "Loading NIfTI atlases requires nibabel. Install it with `pip install nibabel`."
        ) from exc

    path = atlas_path(subject)
    image = nib.load(str(path))
    raw_volume = np.asanyarray(image.dataobj)
    check_atlas(raw_volume, N_PARCELS, expected_shape=ATLAS_SHAPE)
    volume = np.rint(raw_volume).astype(np.int16, copy=False)
    return volume, np.asarray(image.affine, dtype=float), path


def _find_key(h5, task: str):
    matches = [k for k in h5.keys() if task in k]
    if len(matches) > 1:
        exact = [k for k in matches if k.endswith(task)]
        matches = exact if len(exact) == 1 else matches
    if len(matches) != 1:
        raise KeyError(f"{task}: matched {matches}; example keys: {list(h5.keys())[:5]}")
    return matches[0]


def load_clip(h5, stem: str, feat_dir=FEAT_DIR):
    """-> X (n_layers, T, C) float32, Y (T, V) float32, n_tokens (T,), all cut to the same T."""
    Y = h5[_find_key(h5, stem.replace("friends_", ""))][:].astype(np.float32)
    with np.load(feat_dir / f"{stem}.npz") as z:
        X, n_tok = z["X"].astype(np.float32), z["n_tokens"].astype(np.int64)
    if Y.ndim != 2 or Y.shape[1] != N_PARCELS:
        raise ValueError(f"{stem}: expected fMRI data shaped (TRs, {N_PARCELS}), got {Y.shape}")
    if X.ndim != 3 or X.shape[0] != N_LAYERS or X.shape[2] != D_MODEL:
        raise ValueError(f"{stem}: expected BEATs features shaped ({N_LAYERS}, TRs, {D_MODEL}), got {X.shape}")
    if n_tok.ndim != 1 or len(n_tok) != X.shape[1]:
        raise ValueError(f"{stem}: n_tokens has shape {n_tok.shape}, but the feature time axis has {X.shape[1]} TRs")
    T = min(Y.shape[0], X.shape[1])
    if Y.shape[0] != X.shape[1]:
        print(f"  [info] {stem}: fMRI {Y.shape[0]} TRs vs audio {X.shape[1]} TRs -> using {T}")
    return X[:, :T], Y[:T], n_tok[:T]


def lag_features(X: np.ndarray, lags) -> np.ndarray:
    """X (T, C) -> (T, C * len(lags)).  Row t holds the features of TR t-d for each lag d.
    d > 0: the past (the brain responds LATER than the sound); d < 0: the future (diagnostics only).
    Missing rows are zeros. Always applied per clip, so lags never cross clip boundaries."""
    T = X.shape[0]
    out = []
    for d in lags:
        Z = np.zeros_like(X)
        if d == 0:
            Z = X.copy()
        elif 0 < d < T:
            Z[d:] = X[:-d]
        elif -T < d < 0:
            Z[:d] = X[-d:]
        out.append(Z)
    return np.concatenate(out, axis=1)


def row_mask(n_tokens: np.ndarray, drop_first=None) -> np.ndarray:
    """Rows we trust: enough BEATs tokens, and not in the optional edge-drop zones.
    drop_first: how many leading rows to drop; None -> config.DROP_FIRST_TRS (= max of the frozen LAGS).
    Tuning passes it explicitly (the longest candidate lag) so every candidate set is scored on the SAME rows
    and none of them sees a zero-padded lag block."""
    k = DROP_FIRST_TRS if drop_first is None else drop_first
    m = n_tokens >= MIN_TOKENS_PER_TR
    if k:
        m[:k] = False
    if DROP_LAST_TRS:
        m[len(m) - DROP_LAST_TRS:] = False
    return m


def build_design(X_layer: np.ndarray, Y: np.ndarray, n_tokens: np.ndarray, lags, drop_first=None):
    """Lag first (so lags see the real past), then drop untrusted rows. -> (X_lagged, Y)."""
    keep = row_mask(n_tokens, drop_first)
    return lag_features(X_layer, lags)[keep], Y[keep]


def load_subject(subject: str, stems, feat_dir=FEAT_DIR):
    """dict stem -> (X (L,T,C), Y (T,V), n_tokens (T,))."""
    import h5py                      # imported here so the rest of the module works without it
    data = {}
    with h5py.File(fmri_path(subject), "r") as h5:
        for s in stems:
            data[s] = load_clip(h5, s, feat_dir)
    return data