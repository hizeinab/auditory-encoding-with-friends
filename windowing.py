"""Windowing + pooling logic. Pure numpy: no torch, no BEATs, no data -> fully unit-testable.

Picture (W = 10 s, m = 2 s, hop H = 6 s), '=' kept, 'x' margin discarded:
    w0  ========xx
    w1        xx======xx
    w2              xx======xx          every second of the clip is kept exactly once
The model side is injected as a function `forward`, so tests can swap in a fake model.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

# BEATs fbank geometry at 16 kHz (see guide, section 3): 25 ms frames every 10 ms, 16 frames per patch.
FRAME_LEN, FRAME_SHIFT, PATCH_FRAMES = 400, 160, 16


def n_fbank_frames(n_samples: int) -> int:
    """Kaldi framing without edge padding: a frame needs all of its 400 samples."""
    return 0 if n_samples < FRAME_LEN else 1 + (n_samples - FRAME_LEN) // FRAME_SHIFT


def n_time_patches(n_samples: int) -> int:
    """Time-patches per window = floor(frames / 16)."""
    return n_fbank_frames(n_samples) // PATCH_FRAMES


def n_trs(n_samples: int, sr: int, tr: float) -> int:
    return math.ceil(round(n_samples / sr / tr, 9))


def window_starts(n_samples: int, sr: int, seg_sec: float, margin_sec: float):
    """Start sample of every window. Integer sample arithmetic -> no float drift.
    Keep adding the hop until the latest window reaches the end of the clip."""
    seg = int(round(seg_sec * sr))
    hop = int(round((seg_sec - 2 * margin_sec) * sr))
    if hop <= 0:
        raise ValueError("need seg_sec > 2 * margin_sec")
    starts = [0]
    while starts[-1] + seg < n_samples:
        starts.append(starts[-1] + hop)
    return starts, seg, hop


def kept_tokens(n_samples, sr, seg_sec, margin_sec, patch_sec, eps=1e-6):
    """For each window: indices and absolute centre times (s) of the tokens we keep.

    keep  <=>  a_k <= c < b_k  and  c < D,   a_k = 0 (first) else s_k + m,   b_k = inf (last) else s_k + W - m
    `eps` shifts both thresholds so a token exactly on a seam goes to the later window, once.
    """
    starts, seg, _ = window_starts(n_samples, sr, seg_sec, margin_sec)
    K, dur, n_tok = len(starts), n_samples / sr, n_time_patches(seg)
    plan = []
    for k, s in enumerate(starts):
        s_sec = s / sr
        c = s_sec + (np.arange(n_tok) + 0.5) * patch_sec
        lo = 0.0 if k == 0 else s_sec + margin_sec
        hi = np.inf if k == K - 1 else s_sec + seg_sec - margin_sec
        keep = (c >= lo - eps) & (c < hi - eps) & (c < dur - eps)
        plan.append({"k": k, "start": s, "start_sec": s_sec,
                     "idx": np.nonzero(keep)[0], "centers": c[keep]})
    return plan, starts, seg


def count_tokens_per_tr(n_samples, sr, tr, seg_sec, margin_sec, patch_sec, eps=1e-6):
    """How many kept tokens fall in each TR (needs no model)."""
    plan, _, _ = kept_tokens(n_samples, sr, seg_sec, margin_sec, patch_sec, eps)
    n = n_trs(n_samples, sr, tr)
    cnt = np.zeros(n, dtype=np.int64)
    for w in plan:
        r = np.clip(np.floor(w["centers"] / tr).astype(int), 0, n - 1)
        np.add.at(cnt, r, 1)
    return cnt


def pool_clip(forward: Callable[[np.ndarray], np.ndarray], wav: np.ndarray, sr: int, tr: float,
              seg_sec: float, margin_sec: float, patch_sec: float, n_layers: int, d_model: int,
              batch_size: int = 8, eps: float = 1e-6):
    """Cut `wav` into windows, run `forward`, keep trustworthy tokens, average them per TR.

    forward(batch) : (B, seg_samples) float32 -> (B, n_layers, T', d_model)   [one vector per 0.16 s]
    returns        : X (n_layers, n_TR, d_model) float32,  n_tokens (n_TR,) tokens averaged per TR
    """
    wav = np.asarray(wav, dtype=np.float32)
    plan, starts, seg = kept_tokens(len(wav), sr, seg_sec, margin_sec, patch_sec, eps)
    n_tok = n_time_patches(seg)
    ntr = n_trs(len(wav), sr, tr)
    acc = np.zeros((n_layers, ntr, d_model), dtype=np.float64)
    cnt = np.zeros(ntr, dtype=np.int64)

    for b0 in range(0, len(starts), batch_size):
        ks = list(range(b0, min(len(starts), b0 + batch_size)))
        batch = np.zeros((len(ks), seg), dtype=np.float32)       # zero-padding of the last window happens here
        for i, k in enumerate(ks):
            chunk = wav[starts[k]: starts[k] + seg]
            batch[i, : len(chunk)] = chunk
        feats = np.asarray(forward(batch))
        if feats.shape[:2] != (len(ks), n_layers) or feats.shape[2] != n_tok or feats.shape[3] != d_model:
            raise RuntimeError(f"forward() returned {feats.shape}, expected "
                               f"({len(ks)}, {n_layers}, {n_tok}, {d_model})")
        for i, k in enumerate(ks):
            w = plan[k]
            f = feats[i][:, w["idx"]].astype(np.float64)          # (L, n_keep, C)
            r = np.clip(np.floor(w["centers"] / tr).astype(int), 0, ntr - 1)
            for rr in np.unique(r):                               # at most ~4 TRs per window
                sel = r == rr
                acc[:, rr] += f[:, sel].sum(axis=1)
                cnt[rr] += sel.sum()

    if (cnt == 0).any():
        print(f"[warn] {(cnt == 0).sum()} TR(s) received no tokens")
    return (acc / np.maximum(cnt, 1)[None, :, None]).astype(np.float32), cnt