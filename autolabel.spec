# -*- coding: utf-8 -*-
"""PyInstaller spec for AutoLabel Dock (onedir mode)."""
from pathlib import Path
import sys

block_cipher = None

# ROOT is the repo root (cwd when PyInstaller runs the spec).
ROOT = Path.cwd()

# -------------------------------------------------------------------
# Collect all packages that PyInstaller's static scanner might miss.
# -------------------------------------------------------------------
hiddenimports = [
    # PyQt5 submodules used at runtime
    "PyQt5.QtPrintSupport",
    # torch lazy-loads _C from torch/utils/tensorboard (tensorboard itself is optional)
    "torch.utils.tensorboard",
    # PIL lazy-imports _tkinter_finder under some configurations
    "PIL._tkinter_finder",
]

# -------------------------------------------------------------------
# Collect DLLs that OpenCV's FFmpeg / CUDA code needs.
# -------------------------------------------------------------------
from PyInstaller.utils.hooks import collect_dynamic_libs

cv2_binaries = collect_dynamic_libs("cv2")

# -------------------------------------------------------------------
# Other data / binary extras
# -------------------------------------------------------------------
datas = [
    # App resources (icons, screenshots, etc.)
    (str(ROOT / "resources"), "resources"),
    # Model stubs (bundled yolov8 weights distributed with the repo)
    (str(ROOT / "yolov8n.pt"), "."),
    (str(ROOT / "yolov8s.pt"), "."),
]

binaries = cv2_binaries[:]

# -------------------------------------------------------------------
# Custom runtime hook (manages DLL search path before main.py runs)
# -------------------------------------------------------------------
runtime_tmpdir = ROOT / "packaging"
rthook_path = str(runtime_tmpdir / "pyi_rth_dllpath.py")

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(runtime_tmpdir)],
    hooksconfig={},
    runtime_hooks=[rthook_path],
    excludes=[
        # None of these are used by this app and only bloat the bundle.
        "tkinter",          # we use PyQt5, not tkinter
        "test",             # pytest/unittest test suites
        "distutils.tests",  # setuptools distutils tests
        "lib2to3.tests",   # 2to3 grammar fixer tests
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoLabelDock",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,            # UPX on Windows with PyQt5 + torch often causes issues
    console=False,        # Headless GUI app — no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "resources" / "icons" / "app.ico"),
    version=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AutoLabelDock",
)
