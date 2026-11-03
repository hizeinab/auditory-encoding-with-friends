"""STAGE 2 - choose LAGS from the TRAINING episodes only, then freeze the choice in tuned_hparams.json.

Order of the whole project (do not reorder):
    1. extract BEATs features                      (independent of LAGS)
    2. tune_lags.py   <- this file                 (episodes 1-4 only; the test episode is never even loaded)
    3. run_encoding.py                             (reads the frozen LAGS, trains on E1-E4, tests on E5 ONCE)
    4. hierarchy analysis and brain maps

Run:  python tune_lags.py                    # all subjects, layers 3, 6 and 12
      python tune_lags.py --subjects sub-01  # quick look
The result is written to RESULT_DIR/tuned_hparams.json; config.py reads it at import time (LAGS_FROZEN = True).
Do NOT rerun this after you have looked at test-episode results: that would turn the test episode into tuning data.
"""
import argparse
import datetime
import json
from pathlib import Path

from lag_tuning import compare_lag_sets

CANDIDATES = {                       # WRITE THE CANDIDATE LIST DOWN BEFORE LOOKING AT RESULTS (docs/PROTOCOL.md)
    "1-5": [1, 2, 3, 4, 5], "2-6": [2, 3, 4, 5, 6], "3-7": [3, 4, 5, 6, 7], "1-8": list(range(1, 9)),
    "2-8": list(range(2, 9)), "3-5": [3, 4, 5], "1-3": [1, 2, 3], "4": [4],
}
LAYERS = (2, 5, 11)                  # 0-based -> layers 3, 6, 12 (early, middle, late). LAGS must be ONE set for all layers.


def within_1se(rows):
    """Names of the sets whose mean is within one SE of the best (the same rule as compare_lag_sets)."""
    best = max(rows, key=lambda r: r["mean_r"])
    return [r["name"] for r in rows if r["mean_r"] >= best["mean_r"] - best["se"]]


def tune_from_folds(combos, candidates, alphas, backend="auto"):
    """combos: {(subject, layer): raw_folds}, raw folds as in lag_tuning (TRAINING episodes only).
    Returns {(subject, layer): {'rows': [...], 'chosen': name, 'within_1se': [names]}}.
    compare_lag_sets scores every candidate on the same rows (drop_first = longest candidate lag)."""
    out = {}
    for key, folds in combos.items():
        rows, chosen = compare_lag_sets(folds, candidates, alphas, backend=backend)
        out[key] = {"rows": rows, "chosen": chosen, "within_1se": within_1se(rows)}
    return out


def consensus(results, candidates):
    """The simplest set that is within 1 SE of the best in EVERY (subject, layer) combination; None if no set is.
    A design choice (LAGS must be shared by all layers and subjects), not a theorem. Single-combination choices
    wobble between tied sets; this rule gives a stable answer."""
    common = set(candidates)
    for res in results.values():
        common &= set(res["within_1se"])
    if not common:
        return None
    return min(common, key=lambda n: (len(candidates[n]), max(candidates[n])))


def to_record(results, candidates, alphas, final, extra):
    """JSON-serialisable provenance record: what was tuned, on what, and what was chosen."""
    first = next(iter(results.values()))["rows"][0]
    return {
        "LAGS": candidates[final] if final else None, "chosen_set": final,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "candidates": candidates, "alphas": [float(a) for a in alphas], "drop_first_used_in_tuning": first["drop_first"],
        "per_combo": {f"{s}|layer{ly + 1}": {
            "chosen_by_1se": r["chosen"], "within_1se": r["within_1se"],
            "scores": {x["name"]: {"mean_r": round(x["mean_r"], 4), "se": round(x["se"], 4)} for x in r["rows"]}}
            for (s, ly), r in results.items()},
        **extra,
    }


def main(subjects=None, layers=LAYERS, out_path=None):
    import data as data_mod
    from config import ALPHAS, FEAT_DIR, TUNED_PATH, SEASON, SUBJECTS, TRAIN_EPS
    subjects = subjects or SUBJECTS
    combos = {}
    for sub in subjects:
        train_stems = [data_mod.find_clips(SEASON, e, FEAT_DIR) for e in TRAIN_EPS]
        d = data_mod.load_subject(sub, [s for f in train_stems for s in f], FEAT_DIR)   # test clips are never loaded
        for ly in layers:
            combos[(sub, ly)] = [[(d[s][0][ly], d[s][1], d[s][2]) for s in f] for f in train_stems]
    results = tune_from_folds(combos, CANDIDATES, ALPHAS)
    final = consensus(results, CANDIDATES)
    for (s, ly), r in results.items():
        print(f"{s} layer {ly + 1:2d}: 1-SE choice = {r['chosen']:5s} | tied with the best: {r['within_1se']}")
    print("\nCONSENSUS (simplest set tied with the best everywhere):", final)
    if final is None:
        print("No single set is tied everywhere: read the table, decide by hand WITH a written reason, and put the lags "
              "into the JSON yourself (docs/PROTOCOL.md, 'deviations').")
    rec = to_record(results, CANDIDATES, ALPHAS, final,
                    {"train_episodes": list(TRAIN_EPS), "test_episode_loaded": False, "subjects": list(subjects)})
    path = Path(out_path or TUNED_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2))
    print("written:", path, "\nRestart the kernel / reimport config so LAGS_FROZEN picks it up.")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--layers", nargs="+", type=int, default=list(LAYERS), help="0-based layer indices")
    a = ap.parse_args()
    main(a.subjects, a.layers)
