"""In-process script runner for PyInstaller-frozen environments.

When the app is frozen (``sys.frozen == True``) there is no ``python.exe`` in
the bundle: ``sys.executable`` points to the GUI launcher (e.g.
``AutoLabelDock.exe``), which cannot be used as a script interpreter because
its ``stdout`` / ``stderr`` are not connected to QProcess pipes.

Running the script in-process solves three problems at once:

1. No need to locate a Python executable (there isn't one in a console=False
   frozen bundle).
2. All bundled packages (``ultralytics``, ``torch``, ``cv2`` …) are already
   importable — no extra environment setup needed.
3. Output is captured via ``sys.stdout`` / ``sys.stderr`` redirection, so the
   panel's ``output`` signal fires correctly.

The price is that the script shares the main process: a bare ``sys.exit()``
call would terminate the app. We guard against this by replacing ``sys.exit``
with a function that raises ``_ScriptExit`` (a subclass of ``SystemExit``)
so the ``try/except`` in the worker thread catches it safely.
"""
from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal

__all__ = ["InProcessScriptRunner"]


class _ScriptExit(SystemExit):  # noqa: N818
    """Raised instead of calling ``sys.exit`` so the worker thread can catch it."""


def _script_exit(code: int = 0) -> None:
    raise _ScriptExit(code)


class InProcessScriptRunner(QObject):
    """Execute a Python script in-process (no subprocess) with captured I/O.

    Emits the same signals as ``ScriptRunner`` so it can be dropped in as a
    drop-in replacement in the UI panel. Signals are emitted from the worker
    thread; Qt will marshal them to the main thread automatically.
    """

    output = pyqtSignal(str)           # decoded stdout/stderr chunks
    failed_to_start = pyqtSignal()    # never emitted — kept for API compat
    process_error = pyqtSignal(str)   # never emitted — kept for API compat
    finished = pyqtSignal(int, bool)  # exit_code, normal_exit

    def __init__(self, program: str | None = None, parent=None):
        super().__init__(parent)
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def run(
        self,
        script_text: str,
        working_dir: Path | str,
        *,
        extra_env: dict | None = None,
    ) -> bool:
        """Run ``script_text`` in a background thread.

        ``extra_env`` is applied to ``os.environ`` for the duration of the
        script run and then reverted, so the main application environment
        stays clean.
        """
        if self._running:
            return False

        self._running = True

        def worker():
            # Snapshot and apply environment
            orig_env = {}
            if extra_env:
                for k, v in extra_env.items():
                    orig_env[k] = os.environ.get(k)
                    os.environ[k] = v

            # Snapshot and apply cwd
            orig_cwd = os.getcwd()
            cwd = Path(working_dir)
            if cwd.is_dir():
                try:
                    os.chdir(cwd)
                except OSError:
                    pass

            # Replace sys.exit so bare calls don't kill the whole app
            orig_exit = sys.exit
            sys.exit = _script_exit  # type: ignore[assignment]

            # Redirect stdout/stderr so print() output arrives as signals
            orig_stdout = sys.stdout
            orig_stderr = sys.stderr
            captured_out = io.StringIO()
            captured_err = io.StringIO()
            sys.stdout = captured_out
            sys.stderr = captured_err

            try:
                # ``compile`` with ``optimize`` removes ``assert`` / ``__debug__``
                # blocks, matching what ``python -OO`` would do.
                compiled = compile(script_text, "<script>", "exec", optimize=1)
                try:
                    exec(compiled, {"__name__": "__main__", "__file__": ""})
                except _ScriptExit as exc:
                    exit_code = exc.code if isinstance(exc.code, int) else 1
                except KeyboardInterrupt:
                    self.output.emit("[系统] 脚本被用户中断\n")
                    exit_code = 1
                except BaseException as exc:
                    import traceback
                    tb = traceback.format_exc()
                    self.output.emit(tb)
                    exit_code = 1
                else:
                    exit_code = 0

            finally:
                # Restore I/O, exit hook, cwd, and environment
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
                sys.exit = orig_exit  # type: ignore[assignment]

                # Emit any remaining captured output
                out_text = captured_out.getvalue()
                err_text = captured_err.getvalue()
                if out_text:
                    self.output.emit(out_text)
                if err_text:
                    self.output.emit(err_text)

                if cwd.is_dir():
                    try:
                        os.chdir(orig_cwd)
                    except OSError:
                        pass
                for k, v in orig_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

            self.finished.emit(exit_code, True)
            self._running = False

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """In-process runner cannot forcefully terminate; interrupt is a no-op."""
        pass

    def cleanup_temp(self) -> None:
        """No temp files are created; kept for ScriptRunner API compatibility."""
        pass
