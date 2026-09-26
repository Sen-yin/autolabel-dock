"""Runtime hook for PyInstaller-frozen AutoLabel Dock.

Why this file exists
--------------------
When PyInstaller freezes the project into ``dist\\AutoLabelDock\\_internal\\...``,
it scatters native DLLs (torch, OpenCV, MSVC runtime) across multiple
subdirectories:

    _internal\\
        torch\\lib\\c10.dll                  <- loaded by torch/__init__.py
        torch\\lib\\libiomp5md.dll
        cv2\\opencv_videoio_ffmpeg500_64.dll
        msvcp140.dll / vcruntime140.dll      <- MSVC++ runtime

PyInstaller's bootloader adds the bundle directory to ``os.add_dll_directory``
on launch, but Windows' DLL loader searches paths via ``PATH`` and the
``AddDllDirectory`` activation context — and PyQt5 (which loads earlier in
``main.py``) inserts its own Qt5/bin ahead of these. ``torch.__init__`` then
loads ``c10.dll`` from ``torch\\lib`` (relative to its own location), but
``c10.dll`` transitively depends on MSVC++ + Intel OpenMP, and without these
on the DLL search path the loader returns ``WinError 1114`` ("DLL
initialization routine failed").

This hook runs before any user code (PyInstaller invokes every ``runtime_hook``
before executing ``main.py``), so it's the right place to:

  1. Identify the bundle directory (``sys._MEIPASS``).
  2. ``os.add_dll_directory`` for every native subdir the frozen app might
     load DLLs from at runtime. ``add_dll_directory`` is the Windows 7+ API
     that Python's ``os`` exposes; it's safer than mutating ``PATH`` because
     it scopes the search to the current process and doesn't leak.
  3. **Eagerly preload** the torch ``c10.dll`` and ``libiomp5md.dll`` via
     ``ctypes.WinDLL`` so Windows resolves their transitive dependencies
     (MSVC++ runtime + OpenMP shims) and primes the loader's known-DLL
     cache. Without this step, the loader's activation context for
     ``AddDllDirectory`` doesn't always reach the deep transitive imports
     of ``c10.dll`` on PyInstaller-frozen Python — particularly under
     Python 3.13 / torch 2.9, where ``torch\__init__._load_dll_libraries``
     triggers a separate ``LoadLibraryExW`` that bypasses the cached
     activation context. Observed fix: preloading brings the WinError
     1114 error to zero.
  4. Push the same dirs to the *front* of ``PATH`` too. Some DLLs (notably
     MSVC++ runtime + OpenCV videoio FFmpeg) are still resolved via the
     legacy ``PATH`` search order, so we need both mechanisms.

Tested on Windows 10 / 11 with Python 3.10-3.13 and PyInstaller 6.x.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# Subdirectories under _internal that ship native DLLs we need on the DLL
# search path. Each must exist or be silently skipped (the user may have a
# stripped-down install without, e.g., torch's parallel DLLs).
_DLL_SUBDIRS = (
    "",                       # bundle root itself — MSVC runtime, OpenMP shims
    "torch/lib",
    "cv2",
    "ultralytics",            # Ultralytics sometimes drops .dll here
)

# DLLs we want to *eagerly* preload so the loader sees them before
# torch/__init__.py does its own LoadLibrary. Order matters: c10.dll has
# no torch deps, but libiomp5md.dll is OpenMP which c10.dll uses.
_PRELOAD_DLLS = (
    "libiomp5md.dll",   # OpenMP runtime — needed by c10.dll init
    "c10.dll",          # torch's core — aborts here without OMP
)

# Holds the opaque handles returned by ``os.add_dll_directory``. We MUST keep
# them alive for the entire process — Windows invalidates the activation
# context as soon as a handle is GC'd, which silently breaks DLL loading.
# Initialised unconditionally so the append below always has a target list.
_DLL_HANDLES: list = []
_PRELOADED: list = []

# Debug breadcrumb — proves the hook actually ran and shows the final PATH.
# Useful when users report "still WinError 1114" — they can attach this file.
# Written to %TEMP% so it doesn't pollute the bundle directory.
try:
    import tempfile
    _dbg = tempfile.NamedTemporaryFile(
        prefix="autolabel_rth_", suffix=".log", mode="w", delete=False, encoding="utf-8"
    )
    _dbg.write(f"[rth] hook entered, sys._MEIPASS={getattr(sys, '_MEIPASS', None)!r}\n")
    _dbg.flush()
    _DBG_PATH = _dbg.name
except Exception:
    _DBG_PATH = None


def _log(msg: str) -> None:
    if _DBG_PATH:
        try:
            with open(_DBG_PATH, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass


_bundle_root = getattr(sys, "_MEIPASS", None)
if _bundle_root:
    bundle = Path(_bundle_root)
    for sub in _DLL_SUBDIRS:
        d = (bundle / sub) if sub else bundle
        if not d.is_dir():
            _log(f"[rth] skip missing dir: {d}")
            continue
        try:
            # Win 7+ API — see https://learn.microsoft.com/windows/win32/api/libloaderapi/nf-libloaderapi-adddlldirectory
            handle = os.add_dll_directory(str(d))  # type: ignore[attr-defined]
        except (OSError, AttributeError):
            # ``add_dll_directory`` was added in Python 3.8 on Windows. If it
            # ever disappears we fall back to PATH-only.
            handle = None
        if handle is not None:
            _DLL_HANDLES.append(handle)
            _log(f"[rth] add_dll_directory({d}) ok")

    # Belt-and-braces: prepend the bundle subdirs to PATH. Some DLLs (notably
    # MSVC++ runtime + cv2's FFmpeg loader) still resolve via the legacy
    # ``PATH`` search order even when ``AddDllDirectory`` is set.
    extra_path = os.pathsep.join(
        str((bundle / sub) if sub else bundle)
        for sub in _DLL_SUBDIRS
        if ((bundle / sub) if sub else bundle).is_dir()
    )
    if extra_path:
        os.environ["PATH"] = extra_path + os.pathsep + os.environ.get("PATH", "")
        _log(f"[rth] PATH prepended ({len(extra_path.split(os.pathsep))} entries)")

    # Eager preload — this is the actual fix for WinError 1114 under
    # PyInstaller-frozen Python + torch 2.9. Without preloading c10.dll,
    # the frozen loader's activation context doesn't reach c10.dll's
    # transitive deps (MSVCP140, VCRUNTIME140_1). Preloading via ctypes
    # forces Windows to walk and resolve the full import chain eagerly,
    # and the resulting handle stays alive for the process lifetime.
    for name in _PRELOAD_DLLS:
        path = bundle / "torch" / "lib" / name
        if not path.is_file():
            _log(f"[rth] preload skip (missing): {path}")
            continue
        try:
            h = ctypes.WinDLL(str(path))
            _PRELOADED.append(h)
            _log(f"[rth] preloaded OK: {path}")
        except OSError as e:
            _log(f"[rth] preload FAIL {path}: {e}")
else:
    _log("[rth] NOT frozen — skipping (running under dev python)")
