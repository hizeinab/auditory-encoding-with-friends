"""Step 1: pull the audio track out of the Friends clips we need (requires ffmpeg on PATH)."""
import subprocess
import wave
from pathlib import Path

from config import SEASON, SR, STIM_DIR, TEST_EP, TRAIN_EPS, WAV_DIR


def mkv_to_wav(mkv: Path, wav: Path, sr: int = SR) -> None:
    """Mono, 16 kHz, 16-bit PCM. ffmpeg downmixes multichannel audio by averaging channels."""
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mkv),
           "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(wav)]
    subprocess.run(cmd, check=True)


def wav_info(wav: Path):
    """(sample_rate, n_samples, duration_s) read from the wav header."""
    with wave.open(str(wav)) as w:
        return w.getframerate(), w.getnframes(), w.getnframes() / w.getframerate()


def wanted_clips(season=SEASON, episodes=None):
    """All parts (a, b, ...) of the requested episodes, discovered from the files on disk."""
    episodes = episodes or (list(TRAIN_EPS) + [TEST_EP])
    clips = []
    for e in episodes:
        hits = sorted(STIM_DIR.rglob(f"friends_s{season:02d}e{e:02d}[a-z].mkv"))
        if not hits:
            raise FileNotFoundError(f"no mkv for season {season} episode {e} under {STIM_DIR}")
        clips += hits
    return clips


def extract_all(season=SEASON, episodes=None) -> None:
    clips = wanted_clips(season, episodes)
    print(f"Found {len(clips)} video clips for season {season}, episodes {episodes or (list(TRAIN_EPS) + [TEST_EP])}.")
    for mkv in clips:
        wav = WAV_DIR / f"{mkv.stem}.wav"
        if not wav.exists():
            mkv_to_wav(mkv, wav)
        sr, n, dur = wav_info(wav)
        assert sr == SR, f"{wav.name}: {sr} Hz"
        print(f"{wav.name}: {dur:7.1f} s  ({n} samples)  -> ~{dur / 1.49:.0f} TRs")


if __name__ == "__main__":
    extract_all()