"""Step 2: per-layer BEATs features, pooled into fMRI TR bins.

Output per clip (.npz):  X        float16 (N_LAYERS, n_TR, D_MODEL)
                         n_tokens int     (n_TR,)   how many BEATs tokens were averaged in each TR

Inside BEATs:  wav (16 kHz) -> 128-mel fbank (100 frames/s) -> 16x16 patches -> tokens ordered
(time_patch, freq_patch) with freq fastest -> 12 transformer layers (768-d).
We hook each layer, average the 8 frequency patches (one vector per 160 ms), and hand the result
to windowing.pool_clip, which keeps the trustworthy tokens and averages them per TR.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

from config import (BEATS_CKPT, BEATS_REPO, D_MODEL, FEAT_DIR, MARGIN_SEC, N_FREQ_PATCHES,
                    N_LAYERS, PATCH_SEC, RAND_FEAT_DIR, SEGMENT_SEC, SR, TOKEN_EPS, TR, WAV_DIR,
                    SEASON, TRAIN_EPS, TEST_EP)
from windowing import n_time_patches, pool_clip
from audio_extract import wanted_clips


def _import_beats():
    """BEATs.py lives in the microsoft/unilm repo (folder `beats`), not on pip."""
    if str(BEATS_REPO) not in sys.path:
        sys.path.insert(0, str(BEATS_REPO))
    from BEATs import BEATs, BEATsConfig
    return BEATs, BEATsConfig


class BeatsExtractor:
    def __init__(self, ckpt_path=BEATS_CKPT, device=None, random_init=False, seed=0):
        BEATs, BEATsConfig = _import_beats()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # weights_only=False: the checkpoint stores a plain config dict next to the weights
        # (newer torch defaults to weights_only=True). Only do this for files you trust.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        torch.manual_seed(seed)
        self.model = BEATs(BEATsConfig(ckpt["cfg"]))      # a freshly built BEATs is randomly initialised
        if not random_init:                               # control: skip loading -> same architecture, no training
            self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(self.device)
        self.random_init = random_init

        layers = self.model.encoder.layers
        if len(layers) != N_LAYERS:
            raise RuntimeError(f"expected {N_LAYERS} layers, found {len(layers)}")
        self._buf = {}
        self._handles = [layer.register_forward_hook(self._make_hook(i)) for i, layer in enumerate(layers)]

    def _make_hook(self, i):
        def hook(_mod, _inp, out):
            x = out[0] if isinstance(out, tuple) else out
            self._buf[i] = x.detach()
        return hook

    @staticmethod
    def _to_bnc(x: torch.Tensor, batch: int) -> torch.Tensor:
        """BEATs layers are time-first (N_tok, B, C); make that explicit instead of assuming it."""
        if x.shape[0] == batch and x.shape[1] != batch:
            return x                                   # already (B, N, C)
        if x.shape[1] == batch:
            return x.transpose(0, 1)                   # (N, B, C) -> (B, N, C)
        raise RuntimeError(f"cannot find batch axis {batch} in hooked shape {tuple(x.shape)}")

    @torch.no_grad()
    def _forward(self, wavs: np.ndarray) -> np.ndarray:
        """(B, samples) -> (B, N_LAYERS, T', D_MODEL), frequency patches averaged."""
        x = torch.from_numpy(np.asarray(wavs, dtype=np.float32)).to(self.device)
        B = x.shape[0]
        self._buf.clear()
        self.model.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))
        layers = []
        for i in range(N_LAYERS):
            h = self._to_bnc(self._buf[i], B)
            _, N, C = h.shape
            if C != D_MODEL or N % N_FREQ_PATCHES:
                raise RuntimeError(f"layer {i}: unexpected token tensor {tuple(h.shape)}")
            h = h.reshape(B, N // N_FREQ_PATCHES, N_FREQ_PATCHES, C).mean(2)   # token index = t * 8 + f
            layers.append(h)
        out = torch.stack(layers, 1)
        expected = n_time_patches(x.shape[1])
        if out.shape[2] != expected:                   # guards the 998-frames / 62-patches assumption
            raise RuntimeError(f"BEATs produced {out.shape[2]} time-patches, formula says {expected}")
        return out.float().cpu().numpy()

    def clip_features(self, wav_path: Path, batch_size: int = 8):
        wav, sr = torchaudio.load(str(wav_path))
        if sr != SR:
            raise ValueError(f"expected {SR} Hz, got {sr}")
        return pool_clip(self._forward, wav.mean(0).numpy(), sr=SR, tr=TR, seg_sec=SEGMENT_SEC,
                         margin_sec=MARGIN_SEC, patch_sec=PATCH_SEC, n_layers=N_LAYERS,
                         d_model=D_MODEL, batch_size=batch_size, eps=TOKEN_EPS)


def extract_all(random_init: bool = False, batch_size: int = 8) -> None:
    """Extract features for the configured training and test episodes only.

    Existing feature files are retained, so rerunning this function safely skips completed clips.
    """
    out_dir = RAND_FEAT_DIR if random_init else FEAT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = wanted_clips(SEASON, list(TRAIN_EPS) + [TEST_EP])
    wavs = [WAV_DIR / f"{clip.stem}.wav" for clip in clips]
    missing = [p for p in wavs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing WAV files. Run audio_extract.py first:\n" + "\n".join(map(str, missing[:10])))
    ext = BeatsExtractor(random_init=random_init)
    for wav in wavs:
        out = out_dir / f"{wav.stem}.npz"
        if out.exists():
            continue
        X, cnt = ext.clip_features(wav, batch_size)
        np.savez(out, X=X.astype(np.float16), n_tokens=cnt)
        print(f"saved {out.name}: X{X.shape}, tokens/TR min={cnt.min()} median={int(np.median(cnt))}")


if __name__ == "__main__":
    extract_all(random_init=False)