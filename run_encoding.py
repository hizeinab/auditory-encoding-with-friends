"""STAGE 3 - layer-wise encoding. Train on S2 E1-E4 (CV inside, alpha per parcel), test on S2 E5 ONCE.

PREREQUISITE: tune_lags.py has frozen LAGS (RESULT_DIR/tuned_hparams.json). This script refuses to run without it,
because an untuned LAGS in a final fit is the easiest way to contaminate a result. For development only:
--allow-untuned runs with config.DEFAULT_LAGS and prints a warning (never use those results as findings).

Usage:  python run_encoding.py                    # trained BEATs, all subjects
        python run_encoding.py --subjects sub-01  # one subject first
        python run_encoding.py --random           # random-init control (extract its features first); SAME frozen LAGS
"""
import argparse

import numpy as np

import config
from config import (ALPHAS, DROP_FIRST_TRS, FEAT_DIR, LAGS, N_LAYERS, RAND_FEAT_DIR, RESULT_DIR, SEASON,
                    SUBJECTS, TEST_EP, TRAIN_EPS)
from data import build_design, find_clips, load_subject
from ridge import cv_then_test


def require_frozen_lags(allow_untuned=False):
    if config.LAGS_FROZEN:
        return
    msg = ("LAGS are not frozen: run tune_lags.py on the training episodes first (it writes tuned_hparams.json), "
           "then rerun this script.")
    if not allow_untuned:
        raise RuntimeError(msg)
    print("[WARNING] " + msg + " Continuing with DEFAULT_LAGS because of --allow-untuned: do not report these results.")


def run(subject: str, feat_dir, tag: str, allow_untuned=False):
    require_frozen_lags(allow_untuned)
    if DROP_FIRST_TRS < max(LAGS):
        raise ValueError(f"DROP_FIRST_TRS={DROP_FIRST_TRS} < max(LAGS)={max(LAGS)}: rows with zero-padded lag blocks would be used")
    if TEST_EP in TRAIN_EPS:
        raise ValueError(f"test episode {TEST_EP} is also a training episode")
    train_stems = [find_clips(SEASON, e, feat_dir) for e in TRAIN_EPS]      # one CV fold per episode
    test_stems = find_clips(SEASON, TEST_EP, feat_dir)
    if set(test_stems) & {s for f in train_stems for s in f}:
        raise ValueError("a test clip is also in the training folds")
    data = load_subject(subject, [s for f in train_stems for s in f] + test_stems, feat_dir)

    r_l, a_l, cv_l, cvf_l = [], [], [], []
    for layer in range(N_LAYERS):
        def prep(stems):
            return [build_design(data[s][0][layer], data[s][1], data[s][2], LAGS, DROP_FIRST_TRS) for s in stems]

        folds = [prep(f) for f in train_stems]
        parts = prep(test_stems)
        test = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
        out = cv_then_test(folds, test, ALPHAS)
        r_l.append(out["r_test"]); a_l.append(out["best_alpha"]); cv_l.append(out["cv_r"]); cvf_l.append(out["cv_r_folds"])
        r = out["r_test"]
        print(f"{subject} layer {layer + 1:2d}: held-out mean r={np.mean(r):.4f}  "
              f"median r={np.median(r):.4f}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RESULT_DIR / f"{subject}_{tag}.npz", r_test=np.stack(r_l), best_alpha=np.stack(a_l), cv_r=np.stack(cv_l),
             cv_r_folds=np.stack(cvf_l), lags=np.array(LAGS), lags_frozen=np.array(config.LAGS_FROZEN))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--allow-untuned", action="store_true", help="development only; never report these results")
    args = ap.parse_args()
    feat_dir, tag = (RAND_FEAT_DIR, "random") if args.random else (FEAT_DIR, "trained")
    for sub in args.subjects:
        run(sub, feat_dir, tag, args.allow_untuned)
