"""Acquire, back up, and restore the Algonauts 2025 inputs used by this project.

Selectable categories:
    fmri   one Friends / Schaefer-1000 HDF5 file per configured subject
    clips  every available part of the configured Friends episodes
    atlas  one subject-specific Schaefer-1000 NIfTI volume per subject

DataLad installs repository metadata first; ``datalad get`` fetches only the
selected file paths. A category can be fetched separately, so an existing
fMRI/movie backup can be extended with atlas files without downloading the
large files again.

Examples (from the project directory):
    python get_algonauts_data.py list
    python get_algonauts_data.py fetch --categories fmri clips
    python get_algonauts_data.py fetch --categories atlas
    python get_algonauts_data.py backup --categories atlas
    python get_algonauts_data.py restore
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import config

CATEGORIES = ("fmri", "clips", "atlas")


def episodes_needed() -> list[int]:
    """All episodes used for training or held-out testing."""
    return sorted(set(config.TRAIN_EPS) | {config.TEST_EP})


def _normalise_categories(categories: Iterable[str] | str | None = None) -> tuple[str, ...]:
    if categories is None:
        requested = CATEGORIES
    elif isinstance(categories, str):
        requested = (categories,)
    else:
        requested = tuple(categories)
    unknown = sorted(set(requested) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown categories {unknown}; choose from {CATEGORIES}.")
    if not requested:
        raise ValueError("Select at least one data category.")
    # Stable ordering, and remove duplicates.
    return tuple(category for category in CATEGORIES if category in requested)


def _run(cmd, cwd=None):
    """Run a command with live stdout/stderr and fail on a non-zero status."""
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=None if cwd is None else str(cwd), check=True)


def _run_with_diagnostics(cmd, cwd=None):
    """Run an installation command and retain combined output on failure.

    Fetch commands use ``_run`` so DataLad progress can stream to the notebook.
    Installer commands are short, so capturing and replaying their output gives
    a useful diagnostic when Colab raises a CalledProcessError.
    """
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    completed = subprocess.run(
        cmd,
        cwd=None if cwd is None else str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode, cmd, output=completed.stdout
        )
    return completed


def _error_text(exc) -> str:
    """Extract useful installer output from a subprocess failure."""
    output = getattr(exc, "output", None) or getattr(exc, "stdout", None) or ""
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    output = str(output).strip()
    return output[-6000:] if output else str(exc)


def ensure_tools(run=_run):
    """Install and verify DataLad/git-annex in Colab, with fallback methods.

    The previous installer invocation relied on datalad-installer's default
    ``--sudo ask`` behavior. A notebook has no interactive terminal to answer
    that prompt. The installer is now called with ``--sudo ok`` and tries two
    documented binary sources before falling back to the system apt package.
    """
    # Keep test/caller injection possible. The default installation runner
    # captures output so a failed installer reports its actionable diagnostics.
    runner = _run_with_diagnostics if run is _run else run

    runner([
        sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
        "datalad", "datalad-installer",
    ])
    runner(["datalad", "--version"])

    # Reuse a working git-annex installation from an earlier notebook cell.
    if shutil.which("git-annex"):
        runner(["git-annex", "version"])
        print("Verified existing git-annex installation.")
        return

    installer = shutil.which("datalad-installer") or "datalad-installer"
    failures = []

    # Both sources are documented by datalad-installer and do not require a
    # GitHub OAuth token. Try the official release first, then DataLad packages.
    for method in ("datalad/git-annex:release", "datalad/packages"):
        command = [
            installer, "--sudo", "ok", "--log-level", "INFO",
            "git-annex", "-m", method,
        ]
        try:
            runner(command)
            runner(["git-annex", "version"])
            print(f"Verified git-annex installation using {method}.")
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            failures.append(f"{method}: {_error_text(exc)}")
            print(f"Installation method {method!r} failed; trying the next fallback.", flush=True)

    # Colab is a Debian/Ubuntu-based, root-managed environment. Direct apt-get
    # avoids an additional dependency on sudo being present in the container.
    apt_get = shutil.which("apt-get")
    is_root = hasattr(os, "geteuid") and os.geteuid() == 0
    if apt_get and is_root:
        try:
            runner([apt_get, "update", "-qq"])
            runner([apt_get, "install", "-y", "git-annex"])
            runner(["git-annex", "version"])
            print("Verified git-annex installation using the system apt package.")
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            failures.append(f"apt-get: {_error_text(exc)}")
    elif apt_get:
        command = [
            installer, "--sudo", "ok", "--log-level", "INFO",
            "git-annex", "-m", "apt",
        ]
        try:
            runner(command)
            runner(["git-annex", "version"])
            print("Verified git-annex installation using apt.")
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            failures.append(f"apt: {_error_text(exc)}")

    details = "\n\n".join(failures) if failures else "No installation method could be started."
    raise RuntimeError(
        "Could not install or verify git-annex. DataLad acquisition has NOT started. "
        "Review the installer diagnostics below.\n\n" + details
    )


def ensure_git_identity(name=None, email=None, run=_run, get=None):
    """Set a neutral Git identity only when one is not already configured."""
    get = get or (
        lambda key: subprocess.run(
            ["git", "config", "--global", "--get", key],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    )
    if not get("user.name"):
        run(["git", "config", "--global", "user.name", name or "algonauts-user"])
    if not get("user.email"):
        run(["git", "config", "--global", "user.email", email or "algonauts-user@example.com"])


def install_dataset(parent_dir="/content", url=None, run=_run) -> Path:
    """Install DataLad metadata locally (never inside Google Drive); reuse an existing clone."""
    parent_dir = Path(parent_dir)
    if "/drive/" in (str(parent_dir.resolve()) + "/"):
        raise ValueError(f"{parent_dir} is inside Google Drive; install the DataLad dataset on local disk.")
    parent_dir.mkdir(parents=True, exist_ok=True)
    repo = parent_dir / "algonauts_2025.competitors"
    if (repo / ".git").exists():
        print(f"Using existing DataLad installation: {repo}")
        return repo
    run(["datalad", "install", "-r", "-s", url or config.ALGONAUTS_REPO_URL], cwd=parent_dir)
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"DataLad install did not create the expected repository: {repo}")
    return repo


def _warning(selection: dict, category: str, message: str):
    selection["warnings_by_category"][category].append(message)
    selection["warnings"].append(f"[{category}] {message}")


def select_files(repo, subjects=None, season=None, episodes=None) -> dict:
    """Select fMRI, video, and atlas paths from a DataLad clone without downloading them.

    Returns category path lists plus category-specific warnings. Atlas lookup is
    deliberately restricted to ``fmri/<subject>/<ATLAS_DIRNAME>/``. Exactly one
    NIfTI candidate per subject is expected; ambiguous folders are never guessed.
    """
    repo = Path(repo)
    subjects = list(config.SUBJECTS if subjects is None else subjects)
    season = config.SEASON if season is None else int(season)
    episodes = list(episodes_needed() if episodes is None else episodes)
    selection = {
        "fmri": [],
        "clips": [],
        "atlas": [],
        "warnings": [],
        "warnings_by_category": {category: [] for category in CATEGORIES},
    }

    # A. One Friends/Schaefer-1000 HDF5 file per subject.
    for subject in subjects:
        folder = repo / "fmri" / subject / "func"
        hits = sorted(folder.glob(config.FMRI_GLOB)) if folder.exists() else []
        if len(hits) == 1:
            selection["fmri"].extend(hits)
        elif not hits:
            _warning(selection, "fmri", f"no file matching {config.FMRI_GLOB!r} for {subject} in {folder}")
        else:
            _warning(selection, "fmri", f"expected one fMRI file for {subject}, found {[p.name for p in hits]}")

    # B. Every available segment (a, b, c, ...) for the requested episodes.
    stim_dir = repo / "stimuli" / "movies" / "friends" / f"s{season}"
    episode_alt = "|".join(str(int(ep)) for ep in sorted(set(episodes)))
    clip_re = re.compile(rf"s0?-?{season}e0?({episode_alt})[a-z]\.mkv$", re.IGNORECASE)
    selection["clips"] = sorted(
        p for p in stim_dir.glob("*.mkv") if clip_re.search(p.name)
    ) if stim_dir.exists() else []
    found = {
        int(match.group(1))
        for path in selection["clips"]
        if (match := re.search(r"e0?(\d+)[a-z]\.mkv$", path.name, re.IGNORECASE))
    }
    for episode in episodes:
        if int(episode) not in found:
            _warning(selection, "clips", f"no video part found for Season {season}, Episode {episode} in {stim_dir}")

    # C. One subject-specific Schaefer-1000 NIfTI volume per subject.
    for subject in subjects:
        atlas_dir = repo / "fmri" / subject / config.ATLAS_DIRNAME
        hits = sorted(atlas_dir.glob(config.ATLAS_GLOB)) if atlas_dir.exists() else []
        if len(hits) == 1:
            selection["atlas"].extend(hits)
        elif not hits:
            _warning(
                selection, "atlas",
                f"no NIfTI atlas matching {config.ATLAS_GLOB!r} for {subject} in {atlas_dir}",
            )
        else:
            _warning(
                selection, "atlas",
                f"expected one atlas for {subject}, found {[p.name for p in hits]} in {atlas_dir}; "
                "narrow config.ATLAS_GLOB rather than choosing automatically",
            )

    return selection


def _targets_for_categories(selection: dict, categories=None) -> list[Path]:
    categories = _normalise_categories(categories)
    return [path for category in categories for path in selection[category]]


def all_targets(selection: dict, categories=None) -> list[Path]:
    """Return selected target files, all categories by default."""
    return _targets_for_categories(selection, categories)


def describe(selection: dict, repo, categories=None) -> str:
    """Create a reviewable fetch plan containing only requested categories and warnings."""
    repo = Path(repo)
    categories = _normalise_categories(categories)
    counts = " | ".join(f"{category}: {len(selection[category])}" for category in categories)
    lines = [f"Selected files — {counts}"]
    lines.extend(f"  - {path.relative_to(repo)}" for path in _targets_for_categories(selection, categories))
    warning_map = selection.get("warnings_by_category", {})
    for category in categories:
        lines.extend(f"WARNING [{category}]: {message}" for message in warning_map.get(category, []))
    return "\n".join(lines)


def validate_selection(selection: dict, categories=None, subjects=None) -> None:
    """Fail before a fetch if a selected category has missing or ambiguous required files."""
    categories = _normalise_categories(categories)
    subjects = list(config.SUBJECTS if subjects is None else subjects)
    warning_map = selection.get("warnings_by_category", {})
    issues = [
        f"{category}: {message}"
        for category in categories
        for message in warning_map.get(category, [])
    ]
    if "fmri" in categories and len(selection["fmri"]) != len(subjects):
        issues.append(f"fMRI: expected {len(subjects)} subject files, selected {len(selection['fmri'])}")
    if "atlas" in categories and len(selection["atlas"]) != len(subjects):
        issues.append(f"atlas: expected {len(subjects)} subject files, selected {len(selection['atlas'])}")
    if "clips" in categories and not selection["clips"]:
        issues.append("clips: no stimulus files were selected")
    if issues:
        raise FileNotFoundError("Required selection is incomplete:\n- " + "\n- ".join(issues))


def download(repo, selection, jobs=8, run=_run, categories=None) -> list[str]:
    """Run ``datalad get`` for exactly the selected category paths."""
    categories = _normalise_categories(categories)
    relpaths = [str(path.relative_to(repo)) for path in _targets_for_categories(selection, categories)]
    if not relpaths:
        raise ValueError(f"No paths selected for categories {categories}.")
    run(["datalad", "get", f"-J{int(jobs)}", *relpaths], cwd=repo)
    return relpaths


def fetch_report(selection, categories=None) -> tuple[int, int]:
    """Return (present, total); unfetched annex files are broken symlinks and do not exist."""
    targets = _targets_for_categories(selection, categories)
    return sum(path.exists() for path in targets), len(targets)


def backup_to_drive(selection, repo, dest_root, categories=None) -> tuple[int, int, int]:
    """Copy downloaded files to Drive, retaining their paths relative to the dataset root.

    Returns ``(copied, skipped_same_size, missing)``. Existing backup files are
    skipped when their sizes match, mirroring the project's previous behaviour.
    """
    repo, dest_root = Path(repo), Path(dest_root)
    categories = _normalise_categories(categories)
    copied = skipped = missing = 0
    for src in _targets_for_categories(selection, categories):
        if not src.exists():
            missing += 1
            continue
        dst = dest_root / src.relative_to(repo)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(src, dst)  # follows DataLad's symlink and copies the file content
        copied += 1
    return copied, skipped, missing


def restore_from_drive(src_root, dest_root) -> tuple[int, int]:
    """Restore all backed-up categories from Drive to local disk; safe to rerun."""
    src_root, dest_root = Path(src_root), Path(dest_root)
    if not src_root.exists():
        raise FileNotFoundError(
            f"{src_root} does not exist. Run notebook 01 (fetch + backup) or check config.DRIVE_DATA."
        )
    copied = skipped = 0
    for src in sorted(path for path in src_root.rglob("*") if path.is_file()):
        dst = dest_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


def fetch_all(parent_dir="/content", skip_tools=False, git_name=None, git_email=None,
              jobs=8, dry_run=False, run=_run, categories=None):
    """Install/reuse DataLad metadata, validate, and fetch the requested categories.

    By default all three categories are fetched. Use ``categories=("atlas",)``
    to add atlas volumes to an existing Drive backup without re-fetching fMRI or clips.
    """
    categories = _normalise_categories(categories)
    if not skip_tools:
        ensure_tools(run)
    ensure_git_identity(git_name, git_email, run)
    repo = install_dataset(parent_dir, run=run)
    selection = select_files(repo)
    print(describe(selection, repo, categories))
    validate_selection(selection, categories)
    if dry_run:
        return repo, selection

    download(repo, selection, jobs=jobs, run=run, categories=categories)
    present, total = fetch_report(selection, categories)
    print(f"\n{present}/{total} selected files present for categories {categories}.")
    if present != total:
        raise RuntimeError(f"DataLad fetched only {present}/{total} selected files for {categories}.")
    return repo, selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)

    fetch_parser = sub.add_parser("fetch", help="install/reuse DataLad metadata, then download selected categories")
    fetch_parser.add_argument("--parent-dir", default="/content")
    fetch_parser.add_argument("--skip-tools", action="store_true")
    fetch_parser.add_argument("--git-name")
    fetch_parser.add_argument("--git-email")
    fetch_parser.add_argument("--jobs", type=int, default=8)
    fetch_parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    fetch_parser.add_argument("--dry-run", action="store_true")

    list_parser = sub.add_parser("list", help="list candidate paths without downloading")
    list_parser.add_argument("--repo", default="/content/algonauts_2025.competitors")
    list_parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))

    backup_parser = sub.add_parser("backup", help="copy selected downloaded files to Drive")
    backup_parser.add_argument("--repo", default="/content/algonauts_2025.competitors")
    backup_parser.add_argument("--dest", default=str(config.DRIVE_DATA) if config.DRIVE_DATA else None,
                               required=config.DRIVE_DATA is None)
    backup_parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))

    restore_parser = sub.add_parser("restore", help="restore the complete Drive backup to local disk")
    restore_parser.add_argument("--src", default=str(config.DRIVE_DATA) if config.DRIVE_DATA else None,
                                required=config.DRIVE_DATA is None)
    restore_parser.add_argument("--dest", default=str(config.ALGONAUTS_ROOT))

    args = parser.parse_args(argv)
    if args.action == "fetch":
        fetch_all(args.parent_dir, args.skip_tools, args.git_name, args.git_email, args.jobs,
                  args.dry_run, categories=args.categories)
    elif args.action == "list":
        selection = select_files(args.repo)
        print(describe(selection, args.repo, categories=args.categories))
    elif args.action == "backup":
        selection = select_files(args.repo)
        print("copied %d, skipped %d, missing %d" %
              backup_to_drive(selection, args.repo, args.dest, categories=args.categories))
    elif args.action == "restore":
        print("copied %d, skipped %d" % restore_from_drive(args.src, args.dest))


if __name__ == "__main__":
    main()
