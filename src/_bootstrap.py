"""Bootstrap ordering for Windows PyQt5 + torch DLL conflicts.

Import this module BEFORE any ``from PyQt5.*`` or ``from src.app import ...``
statement. It eager-loads ``torch`` and ``ultralytics`` so that the heavy DLLs
(``c10.dll``, OpenMP runtime, etc.) are resident before PyQt5's Qt5Core /
Qt5Gui DLLs are ever loaded — once loaded, the DLL conflict that produces
``WinError 1114: DLL initialization routine failed`` cannot happen, because
PyQt5 is imported *after* torch.

Safe to use on every platform; on non-Windows the imports are simply harmless
(no DLL probe involved).

Call sites:
    - ``main.py`` — first import in the entry-point module.
    - Any other entry-point or test that builds a ``MainWindow`` directly.

This module is also re-imported lazily by ``src/__init__.py`` as a safety
net for IDE debug launches that bypass ``main.py``.
"""
from __future__ import annotations

# Order matters: torch BEFORE ultralytics (ultralytics re-exports torch).
import torch  # noqa: F401  (eager side-effect import for Windows DLL ordering)
import ultralytics  # noqa: F401  (same reason)
