"""Patch torchvision native files into a PyInstaller-bundled exe.

Background
----------
PyInstaller's static analyzer does not bundle ``torchvision``'s native C
extension files (``_C_stable.pyd``, ``image_stable.pyd``) nor its
``jpeg8.dll`` / ``libpng16.dll`` / ``libsharpyuv.dll`` / ``libwebp.dll`` /
``zlib.dll`` runtime DLLs. As a result, ultralytics fails at runtime with::

    operator torchvision::nms does not exist

when the trained model tries to invoke the NMS op (used by validators,
metric plots, etc.).

Usage
-----
Run this *after* ``pyinstaller autolabel.spec`` succeeds. It copies the
missing files from the active venv into ``dist/AutoLabelDock/_internal/
torchvision/`` so the bundled exe can train / validate without rebuilding.

    python packaging/patch_torchvision.py

Or with a custom layout:

    python packaging/patch_torchvision.py --exe-dir dist/AutoLabelDock
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


# Files torchvision loads at import / first-NMS-call time. Listed in the
# order they are resolved; PyInstaller's excludes list keeps these out of
# the COLLECT pass because they live under a non-imported namespace.
# torchvision 0.21.0 uses _C.pyd / image.pyd (older versions used _C_stable.pyd / image_stable.pyd).
PATCH_FILES = (
    "_C.pyd",
    "image.pyd",
    "cudart64_12.dll",
    "jpeg8.dll",
    "libjpeg.dll",
    "libpng16.dll",
    "libsharpyuv.dll",
    "libwebp.dll",
    "nvjpeg64_12.dll",
    "zlib.dll",
)


def _find_venv_torchvision(repo_root: Path) -> Path | None:
    """Locate the active venv's torchvision install directory."""
    candidates = [
        repo_root / "venv" / "Lib" / "site-packages" / "torchvision",
        repo_root / ".venv" / "Lib" / "site-packages" / "torchvision",
    ]
    for path in candidates:
        if (path / "_C_stable.pyd").is_file():
            return path
    return None


def patch(exe_dir: Path, source: Path | None = None) -> int:
    """Copy missing torchvision native files into the exe bundle.

    Returns the number of files successfully patched.
    """
    target = exe_dir / "_internal" / "torchvision"
    if not target.is_dir():
        print(f"[ERROR] torchvision dir not found at {target}", file=sys.stderr)
        return 0

    if source is None:
        source = _find_venv_torchvision(Path(__file__).resolve().parent.parent)
        if source is None:
            print("[ERROR] could not locate venv torchvision install", file=sys.stderr)
            return 0

    patched = 0
    for name in PATCH_FILES:
        src_file = source / name
        dst_file = target / name
        if not src_file.is_file():
            print(f"  [SKIP] {name} not found in venv ({source})")
            continue
        shutil.copy2(src_file, dst_file)
        print(f"  [OK]   {name}  ({src_file.stat().st_size:,} bytes)")
        patched += 1
    return patched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--exe-dir",
        type=Path,
        default=Path("dist/AutoLabelDock"),
        help="PyInstaller COLLECT output directory (default: dist/AutoLabelDock).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="torchvision source dir; defaults to <repo>/venv/Lib/site-packages/torchvision.",
    )
    args = parser.parse_args(argv)

    print(f"Patching torchvision natives -> {args.exe_dir}")
    n = patch(args.exe_dir, args.source)
    print(f"Done: {n} file(s) patched.")
    return 0 if n == len(PATCH_FILES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
