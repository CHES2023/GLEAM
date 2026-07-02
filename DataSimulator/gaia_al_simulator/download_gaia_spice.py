#!/usr/bin/env python3
"""
Download the Gaia SPICE kernels required by the Gaia-like AL simulator.

The files are downloaded from the official ESA SPICE Service.  Run this
script once before running simulate_gaia_al.py.  The local Gaia/ directory
will be created next to this script.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

BASE_URL = "https://spiftp.esac.esa.int/data/SPICE/GAIA/"
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR / "Gaia"
KERNEL_DIR = ROOT_DIR / "kernels"

META_KERNEL_REL = "mk/gaia_ops.tm"
META_KERNEL_URL = BASE_URL + "kernels/" + META_KERNEL_REL
META_KERNEL_PATH = KERNEL_DIR / META_KERNEL_REL

CHUNK_SIZE = 1024 * 1024


def format_size(num_bytes: int | None) -> str:
    if not num_bytes:
        return "unknown size"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def download_file(url: str, output_path: Path) -> None:
    """Download one file unless a non-empty local copy already exists."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[skip] {output_path.relative_to(SCRIPT_DIR)}")
        return

    tmp_path = output_path.with_name(output_path.name + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[down] {url}")
    request = Request(url, headers={"User-Agent": "Python Gaia SPICE downloader"})

    try:
        with urlopen(request, timeout=120) as response:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else None
            downloaded = 0
            last_report_mb = -1

            with tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)

                    current_report_mb = downloaded // (5 * CHUNK_SIZE)
                    if current_report_mb != last_report_mb:
                        last_report_mb = current_report_mb
                        if total:
                            percent = downloaded * 100.0 / total
                            print(
                                f"       {format_size(downloaded)} / {format_size(total)} "
                                f"({percent:5.1f}%)",
                                end="\r",
                            )
                        else:
                            print(f"       {format_size(downloaded)}", end="\r")

        print(" " * 80, end="\r")
        tmp_path.replace(output_path)
        print(f"[done] {output_path.relative_to(SCRIPT_DIR)} ({format_size(output_path.stat().st_size)})")

    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download {url}\nReason: {exc}") from exc


def parse_kernel_list(meta_kernel_text: str) -> list[str]:
    """Return paths such as 'spk/de432s.bsp' listed as '$KERNELS/...' in the meta-kernel."""
    kernel_files = re.findall(r"'\$KERNELS/([^']+)'", meta_kernel_text)
    # Preserve order while removing possible duplicates.
    return list(dict.fromkeys(kernel_files))


def patch_meta_kernel_path(meta_kernel_text: str) -> str:
    """Patch PATH_VALUES so gaia_ops.tm points to the local absolute kernel directory."""
    kernel_path = KERNEL_DIR.resolve().as_posix()
    replacement = f"PATH_VALUES = ( '{kernel_path}' )"
    patched, count = re.subn(
        r"PATH_VALUES\s*=\s*\(\s*'[^']*'\s*\)",
        replacement,
        meta_kernel_text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not find a PATH_VALUES entry in gaia_ops.tm.")
    return patched


def main() -> int:
    print("Gaia SPICE kernel downloader")
    print("Source: official ESA SPICE Service")
    print(f"Base URL: {BASE_URL}")
    print(f"Local directory: {ROOT_DIR}")
    print()

    try:
        # 1. Download the current operational meta-kernel.
        download_file(META_KERNEL_URL, META_KERNEL_PATH)

        # 2. Read it and find all kernel files listed in KERNELS_TO_LOAD.
        original_meta_kernel = META_KERNEL_PATH.read_text(encoding="utf-8", errors="ignore")
        kernel_files = parse_kernel_list(original_meta_kernel)
        if not kernel_files:
            raise RuntimeError(
                "No '$KERNELS/...' entries were found in gaia_ops.tm. "
                "Please check whether the ESA meta-kernel format has changed."
            )

        print()
        print(f"Found {len(kernel_files)} kernel files in gaia_ops.tm.")
        print("Downloading missing files...")
        print()

        # 3. Download each file listed by the official meta-kernel.
        for rel_path in kernel_files:
            download_file(BASE_URL + "kernels/" + rel_path, KERNEL_DIR / rel_path)

        # 4. Patch PATH_VALUES in the local copy of gaia_ops.tm.
        patched_meta_kernel = patch_meta_kernel_path(original_meta_kernel)
        META_KERNEL_PATH.write_text(patched_meta_kernel, encoding="utf-8")

        print()
        print("Done.")
        print(f"Gaia SPICE directory: {ROOT_DIR}")
        print(f"Local meta-kernel:     {META_KERNEL_PATH}")
        print()
        print("Expected next step:")
        print("  python simulate_gaia_al.py")
        return 0

    except Exception as exc:  # Keep this broad so the user sees a clean message.
        print()
        print("Download failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print(
            "\nPlease check your network connection and retry. Existing completed files "
            "will be skipped on the next run.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
