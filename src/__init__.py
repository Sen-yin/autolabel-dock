"""AutoLabel Dock — application package.

Eager-imports ``torch`` / ``ultralytics`` before any PyQt5 module can grab
the PATH or pre-load MSVC runtime DLLs. See ``src._bootstrap`` for the
rationale (Windows PyQt5 + torch DLL conflict producing WinError 1114).
"""
from src import _bootstrap  # noqa: F401  (must be the first import)
