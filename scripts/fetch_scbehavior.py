"""Materialize the SCBehavior YOLO files stored through Git LFS.

The GitHub source archive contains small LFS pointer files. This script copies
the YOLO tree into data/raw and replaces every pointer with the corresponding
immutable media object from the recorded commit.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


POINTER_RE = re.compile(
    rb"version https://git-lfs.github.com/spec/v1\s+"
    rb"oid sha256:([0-9a-f]{64})\s+size (\d+)\s*"
)


def pointer_metadata(path: Path) -> tuple[str, int] | None:
    payload = path.read_bytes()
    match = POINTER_RE.fullmatch(payload)
    if not match:
        return None
    return match.group(1).decode("ascii"), int(match.group(2))


def download_one(path: Path, root: Path, commit: str, retries: int = 5) -> str:
    metadata = pointer_metadata(path)
    if metadata is None:
        return "regular"
    expected_sha, expected_size = metadata
    relative = "SCBehavior_YOLO/" + path.relative_to(root).as_posix()
    quoted = urllib.parse.quote(relative, safe="/")
    url = (
        "https://media.githubusercontent.com/media/CCNUZFW/SCBehavior/"
        f"{commit}/{quoted}"
    )
    temporary = path.with_suffix(path.suffix + ".download")

    for attempt in range(1, retries + 1):
        try:
            subprocess.run(
                [
                    "curl.exe",
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "180",
                    "-o",
                    str(temporary),
                    url,
                ],
                check=True,
            )
            size = temporary.stat().st_size
            if size != expected_size:
                raise ValueError(f"size {size} != {expected_size}")
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if digest != expected_sha:
                raise ValueError("SHA-256 mismatch")
            temporary.replace(path)
            return "downloaded"
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(attempt * 2)
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    source = args.archive_root / "SCBehavior_YOLO"
    if not source.is_dir():
        raise FileNotFoundError(source)

    if args.destination.exists():
        shutil.rmtree(args.destination)
    shutil.copytree(source, args.destination)

    files = [path for path in args.destination.rglob("*") if path.is_file()]
    pointers = [path for path in files if pointer_metadata(path) is not None]
    print(f"Files: {len(files)}; LFS pointers: {len(pointers)}")

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, path, args.destination, args.commit): path
            for path in pointers
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(pointers):
                print(f"Materialized {completed}/{len(pointers)}")

    remaining = [path for path in args.destination.rglob("*") if path.is_file() and pointer_metadata(path)]
    if remaining:
        raise RuntimeError(f"Unresolved LFS pointers: {len(remaining)}")
    print("SCBehavior YOLO materialization complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
