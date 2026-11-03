"""Central configuration for the BEATs-to-fMRI encoding project.

All filesystem locations can be overridden with environment variables.  In Colab,
Drive is mounted by the notebooks (not as an import side effect here).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path


def _path_from_env(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


IN_COLAB = (os.environ.get("BEATS_BRAIN_IN_COLAB") == "1") or ("google.colab" in sys.modules)
PROJECT_DIR = Path(__file__).resolve().parent

if IN_COLAB:
    DRIVE_ROOT = _path_from_env(
        "BEATS_BRAIN_DRIVE_ROOT", "/content/drive/MyDrive/BEATs_algonauts_project"
    )
    ALGONAUTS_ROOT = _path_from_env("ALGONAUTS_ROOT", "/content/data")
    DRIVE_DATA = _path_from_env("ALGONAUTS_DRIVE_DATA", "/content/drive/MyDrive/algonauts_2025")
    BEATS_REPO = _path_from_env("BEATS_REPO", "/content/unilm/beats")
    BEATS_CKPT = _path_from_env(
        "BEATS_CKPT", DRIVE_ROOT / "checkpoints" / "BEATs_iter3_plus_AS2M.pt"
    )
    WORK_DIR = _path_from_env("BEATS_BRAIN_WORK_DIR", DRIVE_ROOT / "outputs")
else:
    ALGONAUTS_ROOT = _path_from_env("ALGONAUTS_ROOT", PROJECT_DIR / "data" / "algonauts_2025.competitors")
    _drive_data = os.environ.get("ALGONAUTS_DRIVE_DATA")
    DRIVE_DATA = Path(_drive_data).expanduser() if _drive_data else None
    BEATS_REPO = _path_from_env("BEATS_REPO", PROJECT_DIR / "external" / "unilm" / "beats")
    BEATS_CKPT = _path_from_env(
        "BEATS_CKPT", PROJECT_DIR / "checkpoints" / "BEATs_iter3_plus_AS2M.pt"
    )
    WORK_DIR = _path_from_env("BEATS_BRAIN_WORK_DIR", PROJECT_DIR / "outputs")

# ---- Algonauts dataset selection ---------------------------------------------
ALGONAUTS_REPO_URL = "https://github.com/courtois-neuromod/algonauts_2025.competitors.git"
FMRI_GLOB = "*task-friends*Schaefer*1000*.h5"

# Subject-specific Schaefer-1000 atlas volumes supplied with the Algonauts data.
# The documentation places them in fmri/<subject>/atlas/. The pattern supports
# both compressed and uncompressed NIfTI filenames; exactly one file is expected
# per subject. Narrow this pattern if the directory contains multiple NIfTI files.
ATLAS_DIRNAME = "atlas"
ATLAS_GLOB = "*.nii.gz*"
ATLAS_SHAPE = (97, 115, 97)

# ---- facts about the audio, BEATs and fMRI (do not change casually) ------------
SR = 16_000
FBANK_SHIFT_SEC = 0.010
PATCH_FRAMES = 16
N_MELS = 128
PATCH_BINS = 16
PATCH_SEC = PATCH_FRAMES * FBANK_SHIFT_SEC
N_FREQ_PATCHES = N_MELS // PATCH_BINS
N_LAYERS = 12
D_MODEL = 768
N_PARCELS = 1000
TR = 1.49

# ---- audio windowing and pooling choices --------------------------------------
SEGMENT_SEC = 10.0
MARGIN_SEC = 2.0
HOP_SEC = SEGMENT_SEC - 2 * MARGIN_SEC
if HOP_SEC <= 0:
    raise ValueError("SEGMENT_SEC must be greater than twice MARGIN_SEC")
TOKEN_EPS = 1e-6
FEATURE_TAG = f"W{SEGMENT_SEC:g}_m{MARGIN_SEC:g}"
MIN_TOKENS_PER_TR = 5

# ---- experiment split and ridge model -----------------------------------------
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]
SEASON = 2
TRAIN_EPS = [1, 2, 3, 4]
TEST_EP = 5
ALPHAS = [10 ** x for x in range(2, 8)] + [3 * 10 ** x for x in range(2, 7)]

# ---- derived paths -------------------------------------------------------------
STIM_DIR = ALGONAUTS_ROOT / "stimuli" / "movies" / "friends"
FMRI_DIR = ALGONAUTS_ROOT / "fmri"
WAV_DIR = WORK_DIR / "wav"
FEAT_DIR = WORK_DIR / "features" / f"beats_{FEATURE_TAG}"
RAND_FEAT_DIR = WORK_DIR / "features" / f"beats_random_{FEATURE_TAG}"
RESULT_DIR = WORK_DIR / "results"

# ---- lag parameters: tune on training episodes, then freeze -------------------
DEFAULT_LAGS = list(range(2, 9))  # provisional only; tune_lags.py selects the final set
TUNED_PATH = RESULT_DIR / "tuned_hparams.json"


def load_tuned_lags(path: str | Path | None = None):
    """Return frozen lags from the tuning record, or None if no valid record exists."""
    path = Path(path or TUNED_PATH)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
        lags = record.get("LAGS")
        if not isinstance(lags, list) or not lags:
            return None
        lags = [int(x) for x in lags]
        if any(x < 0 for x in lags) or sorted(set(lags)) != lags:
            raise ValueError(f"Invalid LAGS in {path}: {lags}")
        return lags
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Could not read tuned lag configuration from {path}") from exc


_tuned_lags = load_tuned_lags()
LAGS_FROZEN = _tuned_lags is not None
LAGS = list(_tuned_lags) if LAGS_FROZEN else list(DEFAULT_LAGS)
DROP_FIRST_TRS = max(LAGS)
DROP_LAST_TRS = 0


def n_trs(duration_sec: float) -> int:
    """Number of TRs represented by a clip of ``duration_sec`` seconds."""
    return math.ceil(round(duration_sec / TR, 9))


def n_windows(duration_sec: float) -> int:
    """Number of overlapping BEATs windows needed to cover a clip."""
    if duration_sec <= SEGMENT_SEC:
        return 1
    return math.ceil((duration_sec - SEGMENT_SEC) / HOP_SEC) + 1
