"""Script runner controller — owns the tempfile + QProcess lifecycle.

Two execution modes:

* **Subprocess (dev):** writes the script to a tempfile and runs it with
  ``sys.executable -u script.py`` via ``QProcess``. Output streams back through
  ``readyReadStandardOutput/StandardError`` signals.

* **In-process (frozen/PyInstaller):** the ``sys.executable`` of a
  ``console=False`` PyInstaller bundle is the GUI launcher (e.g.
  ``AutoLabelDock.exe``), not a Python interpreter, and its stdout/stderr are
  not connected to ``QProcess`` pipes — all script output would be silently
  lost.  In this case we delegate to ``InProcessScriptRunner`` which executes
  the script in a background thread inside the same process, capturing output
  via ``sys.stdout``/``sys.stderr`` redirection.  The runner exposes the same
  signal API so callers (e.g. the tool panel) are unchanged.

Contract pinned by the panel's close choreography (see the task PRD §6):

- ``run()`` keeps the synchronous start check (``waitForStarted(1000)``):
  on failure it releases the process, cleans the temp script, emits
  ``failed_to_start`` and returns False. The ``FailedToStart`` code is
  suppressed in the ``errorOccurred`` handler so the failure is not
  double-reported through ``process_error``.
- ``stop()`` is synchronous: terminate → ``waitForFinished(1500)`` → kill →
  ``waitForFinished(1000)``. When it returns the process has died and the
  ``finished`` signal has already been delivered.
- ``finished(exit_code, normal_exit)`` is emitted only after the runner has
  released its QProcess and cleaned the temp script, so ``is_running`` is
  False inside any ``finished`` slot.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

from PyQt5.QtCore import QObject, QProcess, pyqtSignal

from src.controllers.script_tools_inprocess import InProcessScriptRunner

logger = logging.getLogger(__name__)

_ERROR_NAMES = {
    QProcess.FailedToStart: "启动失败",
    QProcess.Crashed: "进程崩溃",
    QProcess.Timedout: "超时",
    QProcess.WriteError: "写入错误",
    QProcess.ReadError: "读取错误",
    QProcess.UnknownError: "未知错误",
}

START_TIMEOUT_MS = 1000
TERMINATE_TIMEOUT_MS = 1500
KILL_TIMEOUT_MS = 1000


def _should_use_inprocess() -> bool:
    """Return True when we must use the in-process runner.

    In a PyInstaller ``console=False`` frozen bundle, ``sys.executable`` is the
    GUI launcher (e.g. ``AutoLabelDock.exe``), not a Python interpreter. Starting
    a subprocess with that executable produces no output whatsoever because Qt's
    QProcess cannot capture the GUI launcher's stdout/stderr pipes.

    Detection: ``sys.frozen`` tells us we are in a PyInstaller bundle. The
    additional suffix check rules out the rare case where someone runs the
    frozen interpreter directly (``.py``/``.pyw`` suffix).
    """
    if not getattr(sys, "frozen", False):
        return False
    exe = Path(sys.executable)
    return exe.suffix.lower() not in (".py", ".pyw")


class ScriptRunner(QObject):
    """Run one Python script at a time, streaming its output.

    Uses an in-process runner when frozen (PyInstaller GUI bundle), otherwise
    spawns a subprocess via QProcess.
    """

    output = pyqtSignal(str)          # decoded stdout/stderr chunks
    failed_to_start = pyqtSignal()
    process_error = pyqtSignal(str)   # Chinese error name (FailedToStart deduped)
    finished = pyqtSignal(int, bool)  # exit_code, normal_exit

    def __init__(self, program: str | None = None, parent=None):
        super().__init__(parent)
        self._program = str(program) if program else sys.executable
        self._process: QProcess | None = None
        self._temp_script_path: Path | None = None
        self._inproc_runner: InProcessScriptRunner | None = None

    @property
    def is_running(self) -> bool:
        if self._inproc_runner is not None:
            return self._inproc_runner.is_running
        return self._process is not None

    def run(
        self,
        script_text: str,
        working_dir: Path | str,
        *,
        extra_env: dict[str, str] | None = None,
    ) -> bool:
        """Write ``script_text`` to a temp file and start it.

        Returns False when a script is already running or the process failed
        to start (the latter also emits ``failed_to_start``).

        ``extra_env``, if provided, is merged into the subprocess environment
        so the script can read the injected values via ``os.environ``.
        """
        if self.is_running:
            return False

        # ── In-process mode (frozen GUI bundle) ───────────────────────────────
        if _should_use_inprocess():
            self._inproc_runner = InProcessScriptRunner(parent=self)
            self._inproc_runner.output.connect(self.output)
            self._inproc_runner.finished.connect(self._on_inproc_finished)
            return self._inproc_runner.run(
                script_text, working_dir, extra_env=extra_env
            )

        # ── Subprocess mode (dev / console frozen) ───────────────────────────
        self.cleanup_temp()
        fd, script_path = tempfile.mkstemp(prefix="autolabel_script_", suffix=".py")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(script_text)
                if not script_text.endswith("\n"):
                    f.write("\n")
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            Path(script_path).unlink(missing_ok=True)
            raise

        self._temp_script_path = Path(script_path)

        process = QProcess(self)
        process.setProgram(self._program)
        process.setArguments(["-u", str(self._temp_script_path)])
        process.setWorkingDirectory(str(working_dir))

        # Explicitly separate stdout/stderr so readyRead* signals fire independently.
        process.setProcessChannelMode(QProcess.SeparateChannels)

        # Merge extra environment variables into the inherited environment.
        if extra_env:
            from PyQt5.QtCore import QProcessEnvironment

            base_env = os.environ.copy()
            base_env.update(extra_env)
            proc_env = QProcessEnvironment()
            for key, val in base_env.items():
                proc_env.insert(key, val)
            process.setProcessEnvironment(proc_env)

        process.readyReadStandardOutput.connect(self._on_stdout)
        process.readyReadStandardError.connect(self._on_stderr)
        process.errorOccurred.connect(self._on_process_error)
        process.finished.connect(self._on_finished)

        self._process = process

        process.start()
        if not process.waitForStarted(START_TIMEOUT_MS):
            process.deleteLater()
            self._process = None
            self.cleanup_temp()
            self.failed_to_start.emit()
            return False
        return True

    def _on_inproc_finished(self, exit_code: int, normal_exit: bool) -> None:
        self._inproc_runner = None
        self.finished.emit(exit_code, normal_exit)

    def stop(self) -> None:
        """Stop the running script; blocks until the process has terminated."""
        if self._inproc_runner is not None:
            self._inproc_runner.stop()
            return
        if self._process is None:
            return

        self._process.terminate()
        if not self._process.waitForFinished(TERMINATE_TIMEOUT_MS):
            self._process.kill()
            self._process.waitForFinished(KILL_TIMEOUT_MS)

    def cleanup_temp(self) -> None:
        """Remove the temp script file if one is left over."""
        if self._temp_script_path is None:
            return
        try:
            self._temp_script_path.unlink(missing_ok=True)
        finally:
            self._temp_script_path = None

    def _on_stdout(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput())
        self.output.emit(data.decode("utf-8", errors="replace"))

    def _on_stderr(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardError())
        self.output.emit(data.decode("utf-8", errors="replace"))

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.FailedToStart:
            return
        self.process_error.emit(_ERROR_NAMES.get(error, str(int(error))))

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        normal_exit = exit_status == QProcess.NormalExit

        if self._process is not None:
            self._process.deleteLater()
            self._process = None
        self.cleanup_temp()

        self.finished.emit(exit_code, normal_exit)
