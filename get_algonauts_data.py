"""STEP 1 (run once) - get ONLY the Algonauts 2025 files this project needs, and keep a permanent copy on Drive.

What is needed, taken from config.py (nothing is hard-coded here):
    fMRI      one Friends .h5 per subject in config.SUBJECTS            (same glob as data.fmri_path: config.FMRI_GLOB)
    stimuli   every part (a, b, c ...) of the clips of season config.SEASON, episodes config.TRAIN_EPS + config.TEST_EP

The dataset is distributed with DataLad. `datalad install -r` clones only the git METADATA (small); the large files are
fetched by `datalad get`, which is restricted to the paths chosen by select_files(). Files that were not fetched are broken
symlinks: `path.exists()` is False for them, which is how "downloaded or not" is decided below.

Three actions (CLI or functions; notebook 01 uses the functions):
    fetch    install the metadata, list what will be downloaded, download it          (fetch_all)
    backup   copy the downloaded files to Google Drive, keeping the folder structure  (backup_to_drive)
    restore  copy them back from Drive to the fast local disk at the start of a session (restore_from_drive)

    python get_algonauts_data.py list            # dry run: what WOULD be downloaded (needs the metadata clone)
    python get_algonauts_data.py fetch
    python get_algonauts_data.py backup
    python get_algonauts_data.py restore

Only the fMRI time series and movie clips required by the encoding analysis are selected. Atlas files are not
retrieved by this workflow; spatial mapping is handled separately and is outside the current pipeline.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import config


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------
def episodes_needed():
    """Every episode used for training or testing, e.g. [1, 2, 3, 4, 5]."""
    return sorted(set(config.TRAIN_EPS) | {config.TEST_EP})


def _run(cmd, cwd=None):
    """Run a command, stop on failure. Tests pass a recorder instead, so nothing is executed."""
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=None if cwd is None else str(cwd), check=True)


# ----------------------------------------------------------------------------------------------------------------------
# 1. tools and the metadata clone
# ----------------------------------------------------------------------------------------------------------------------
def ensure_tools(run=_run):
    """Install DataLad and git-annex in Google Colab."""
    run([
        sys.executable, "-m", "pip", "-q",
        "install", "datalad", "datalad-installer"
    ])

    run([
        "datalad-installer",
        "--sudo", "ok",
        "--log-level", "DEBUG",
        "git-annex",
        "-m", "datalad/git-annex:release"
    ])


def ensure_git_identity(name=None, email=None, run=_run, get=None):
    """
    Configure Git identity if it has not already been configured.
    """
    get = get or (
        lambda key: subprocess.run(
            ["git", "config", "--global", "--get", key],
            capture_output=True,
            text=True
        ).stdout.strip()
    )

    if not get("user.name"):
        run([
            "git", "config", "--global",
            "user.name", name or "algonauts-user"
        ])

    if not get("user.email"):
        run([
            "git", "config", "--global",
            "user.email", email or "algonauts-user@example.com"
        ])


def install_dataset(parent_dir, url=None, run=_run):
    """`datalad install -r`: metadata only. Must NOT run inside Google Drive (slow, and symlinks do not work there)."""
    parent_dir = Path(parent_dir)
    if "/drive/" in str(parent_dir.resolve()) + "/":
        raise ValueError(f"{parent_dir} is inside Google Drive: install the dataset on the local disk (e.g. /content)")
    parent_dir.mkdir(parents=True, exist_ok=True)
    repo = parent_dir / "algonauts_2025.competitors"
    if (repo / ".git").exists():
        print(f"Using existing DataLad installation: {repo}")
        return repo
    run(["datalad", "install", "-r", "-s", url or config.ALGONAUTS_REPO_URL], cwd=parent_dir)
    return repo


# ----------------------------------------------------------------------------------------------------------------------
# 2. choose the files (pure function: easy to test, no network)
# ----------------------------------------------------------------------------------------------------------------------
def select_files(repo, subjects=None, season=None, episodes=None):
    """Return the required fMRI and clip paths plus warnings, all inside `repo`."""
    repo = Path(repo)
    subjects = list(subjects or config.SUBJECTS)
    season = config.SEASON if season is None else season
    episodes = list(episodes or episodes_needed())
    out = {"fmri": [], "clips": [], "warnings": []}

    for sub in subjects:                                         # A. one Friends Schaefer-1000 fMRI file per subject
        hits = sorted((repo / "fmri" / sub / "func").glob(config.FMRI_GLOB))
        if not hits:
            out["warnings"].append(f"no fMRI file matching {config.FMRI_GLOB} for {sub}")
        out["fmri"].extend(hits)

    stim_dir = repo / "stimuli" / "movies" / "friends" / f"s{season}"      # B. every part of every needed episode
    pattern = re.compile(rf"s0?-?{season}e0?({'|'.join(str(e) for e in episodes)})[a-z]\.mkv$", re.IGNORECASE)
    out["clips"] = sorted(p for p in stim_dir.glob("*.mkv") if pattern.search(p.name))
    found = {int(re.search(rf"e0?(\d+)[a-z]\.mkv$", p.name, re.IGNORECASE).group(1)) for p in out["clips"]}
    for ep in episodes:
        if ep not in found:
            out["warnings"].append(f"no clip found for season {season}, episode {ep} in {stim_dir}")

    return out


def all_targets(selection):
    return selection["fmri"] + selection["clips"]


def describe(selection, repo):
    """Human-readable plan: print this and CHECK it before downloading."""
    repo = Path(repo)
    lines = [f"fMRI files: {len(selection['fmri'])} | video clips: {len(selection['clips'])}"]
    lines += [f"  - {p.relative_to(repo)}" for p in all_targets(selection)]
    lines += [f"WARNING: {w}" for w in selection["warnings"]]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------------
# 3. download, check, back up, restore
# ----------------------------------------------------------------------------------------------------------------------
def download(repo, selection, jobs=8, run=_run):
    """`datalad get` for exactly the selected paths (nothing else is downloaded)."""
    rel = [str(p.relative_to(repo)) for p in all_targets(selection)]
    if not rel:
        raise ValueError("nothing selected: check the warnings from select_files()")
    run(["datalad", "get", f"-J{jobs}", *rel], cwd=repo)
    return rel


def fetch_report(selection):
    """(present, total): a file is present if it has real content (a not-fetched annex file is a broken symlink)."""
    targets = all_targets(selection)
    return sum(1 for p in targets if p.exists()), len(targets)


def backup_to_drive(selection, repo, dest_root):
    """Copy the downloaded files to `dest_root`, keeping fmri/... and stimuli/... Skips files already there with the same size.
    Returns (copied, skipped, missing)."""
    repo, dest_root = Path(repo), Path(dest_root)
    copied = skipped = missing = 0
    for src in all_targets(selection):
        if not src.exists():
            missing += 1
            continue
        dst = dest_root / src.relative_to(repo)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(src, dst)                     # follows the symlink: copies real content
        copied += 1
    return copied, skipped, missing


def restore_from_drive(src_root, dest_root):
    """Copy the backup from Drive to the fast local disk (replaces `!cp -r`). Skips same-size files, so it can be rerun."""
    src_root, dest_root = Path(src_root), Path(dest_root)
    if not src_root.exists():
        raise FileNotFoundError(f"{src_root} does not exist: run notebook 01 (fetch + backup) first, or fix config.DRIVE_DATA")
    copied = skipped = 0
    for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
        dst = dest_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


def fetch_all(parent_dir="/content", skip_tools=False, git_name=None, git_email=None, jobs=8, dry_run=False, run=_run):
    """Install metadata, print the plan, download. dry_run=True stops after printing the plan."""
    if not skip_tools:
        ensure_tools(run)
    ensure_git_identity(git_name, git_email, run)
    repo = install_dataset(parent_dir, run=run)
    selection = select_files(repo)
    print(describe(selection, repo))
    if selection["warnings"]:
        raise FileNotFoundError(
            "The required file selection is incomplete. Resolve the warnings above before downloading."
        )
    if dry_run:
        return repo, selection
    download(repo, selection, jobs, run)
    present, total = fetch_report(selection)
    print(f"\n{present}/{total} required files present (one fMRI file per subject plus the selected clips)")
    if present != total:
        raise RuntimeError(f"DataLad fetched only {present} of {total} selected files.")
    return repo, selection


# ----------------------------------------------------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="action", required=True)
    f = sub.add_parser("fetch", help="install metadata, list, download")
    f.add_argument("--parent-dir", default="/content")
    f.add_argument("--skip-tools", action="store_true")
    f.add_argument("--git-name"); f.add_argument("--git-email")
    f.add_argument("--jobs", type=int, default=8)
    f.add_argument("--dry-run", action="store_true")
    for name in ("list", "backup"):
        s = sub.add_parser(name)
        s.add_argument("--repo", default="/content/algonauts_2025.competitors")
        if name == "backup":
            s.add_argument("--dest", default=str(config.DRIVE_DATA) if config.DRIVE_DATA else None,
                           required=config.DRIVE_DATA is None)
    r = sub.add_parser("restore")
    r.add_argument("--src", default=str(config.DRIVE_DATA) if config.DRIVE_DATA else None,
                   required=config.DRIVE_DATA is None)
    r.add_argument("--dest", default=str(config.ALGONAUTS_ROOT))
    a = ap.parse_args(argv)

    if a.action == "fetch":
        fetch_all(a.parent_dir, a.skip_tools, a.git_name, a.git_email, a.jobs, a.dry_run)
    elif a.action == "list":
        print(describe(select_files(a.repo), a.repo))
    elif a.action == "backup":
        print("copied %d, skipped %d, missing %d" % backup_to_drive(select_files(a.repo), a.repo, a.dest))
    elif a.action == "restore":
        print("copied %d, skipped %d" % restore_from_drive(a.src, a.dest))


if __name__ == "__main__":
    main()
