"""AutoLabel Dock — entry point."""
import sys

# IMPORTANT: torch (and transitively ultralytics) MUST be imported *before*
# PyQt5's QApplication is constructed. PyQt5 inserts the Qt5 DLL directory
# at the front of PATH and pre-loads MSVC++ runtime DLLs that conflict with
# torch's c10.dll loader, causing "WinError 1114: DLL initialization routine
# failed" the moment torch is imported later (e.g. lazy ``import torch``
# inside _detect_available_devices when opening a project). Eagerly loading
# torch here once means all later imports hit ``sys.modules`` and skip the
# broken DLL probe entirely. This is a Windows-only quirk; harmless on
# Linux/macOS.
import torch  # noqa: F401  (eager side-effect import — see note above)
import ultralytics as _ultralytics  # noqa: F401  (same reason: lazy-loaded by YOLO later; renamed to silence unused-import warnings)

from PyQt5.QtWidgets import QApplication

from src.app import MainWindow
from src.ui.theme import apply_theme
from src.utils.logging_config import setup_logging
from src.ui.app_icon import load_app_icon


def main() -> int:
    setup_logging()
    app = QApplication(sys.argv)
    # Set the icon on the application (not just the window) so Windows uses
    # it for the taskbar and Alt-Tab. The window-level call in MainWindow
    # still owns the title-bar icon.
    app.setWindowIcon(load_app_icon())
    apply_theme(app)
    window = MainWindow()
    window.setWindowIcon(load_app_icon())
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
