"""Script tool panel — editable Python script runner view with live output.

Pure view: tool-file CRUD delegates to ``core.script_tools.ToolRepository``
and the process lifecycle to ``controllers.script_tools.ScriptRunner``. The
panel keeps only UI construction, dirty/unsaved-confirm flow, running-state
visuals and output presentation (timestamp lines, separators, ``[系统]``
prefixes — the runner emits data only).

Preview tab:
  After a script finishes, the panel automatically scans the script's
  working directory for new image files (PNG / JPG / BMP) and displays the
  first one found in the Preview tab. This lets scripts that produce
  visual output (plots, annotated images, etc.) show results inline without
  leaving the tool.

RunScriptDialog:
  A modal shown before every script run that lets the user select a model
  and/or an image/video file. Selected values are injected as environment
  variables (``AL_MODEL_PATH``, ``AL_INPUT_PATH``, etc.) so the script can
  ``os.environ.get(...)`` without any import changes.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFontDatabase, QKeySequence, QTextCursor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QPlainTextEdit,
    QPushButton,
    QLabel,
    QMessageBox,
    QScrollArea,
    QShortcut,
    QInputDialog,
    QTabWidget,
    QFileDialog,
    QLineEdit,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QSizePolicy,
    QDoubleSpinBox,
)

from src.controllers.script_tools import ScriptRunner
from src.core.config import AppConfig
from src.core.script_tools import ToolRepository
from src.ui.theme import set_button_role, text_style


# ── RunScriptDialog ────────────────────────────────────────────────────────────
class RunScriptDialog(QDialog):
    """Modal dialog to configure environment variables before running a script.

    Shows a list of registered models and a file picker for image/video input.
    The selected values are injected as environment variables so the script can
    read them via ``os.environ`` without any import changes.

    Variables injected:
        AL_MODEL_NAME   — display name of the selected model
        AL_MODEL_PATH   — absolute path to the model's weights file
        AL_TASK_TYPE    — "detect", "classify", or "pose"
        AL_CLASSES     — comma-separated class names
        AL_INPUT_PATH   — absolute path to the selected image or video file
        AL_PROJECT_DIR  — parent project directory (for resolving relative paths)

    The dialog is informational-first: model and input are both optional.
    A script that doesn't need them runs unchanged; a script that does can
    ``os.environ.get("AL_MODEL_PATH")`` and fall back gracefully.
    """

    def __init__(
        self,
        model_registry=None,
        parent=None,
        last_model_id=None,
        last_input_path=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("执行脚本 — 配置")
        self.setMinimumWidth(520)
        self._model_registry = model_registry
        self._selected_model = None
        self._selected_input = None

        lay = QVBoxLayout(self)
        lay.setSpacing(12)

        # ── Model ─────────────────────────────────────────────────────────────
        model_grp = QGroupBox("模型 (可选)")
        model_lay = QGridLayout(model_grp)

        model_lay.addWidget(QLabel("选择模型:"), 0, 0)
        self._model_combo = QComboBox()
        self._model_combo.setEditable(False)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_lay.addWidget(self._model_combo, 0, 1, 1, 2)

        self._model_info_label = QLabel("未选择模型 — 如脚本需要请先选择")
        self._model_info_label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self._model_info_label.setWordWrap(True)
        model_lay.addWidget(self._model_info_label, 1, 0, 1, 3)

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.0, 1.0)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.25)
        self._conf_spin.setPrefix("置信度阈值: ")
        model_lay.addWidget(self._conf_spin, 2, 0, 1, 2)

        lay.addWidget(model_grp)

        # ── Input file ───────────────────────────────────────────────────────
        input_grp = QGroupBox("输入文件 (可选)")
        input_lay = QGridLayout(input_grp)

        input_lay.addWidget(QLabel("选择文件:"), 0, 0)
        self._input_path_edit = QLineEdit()
        self._input_path_edit.setPlaceholderText("点击右侧按钮选择图片或视频...")
        self._input_path_edit.setReadOnly(True)
        input_lay.addWidget(self._input_path_edit, 0, 1)

        self._btn_browse_input = QPushButton("浏览...")
        self._btn_browse_input.clicked.connect(self._on_browse_input)
        input_lay.addWidget(self._btn_browse_input, 0, 2)

        lay.addWidget(input_grp)

        # ── Info note ────────────────────────────────────────────────────────
        note = QLabel(
            "脚本可通过 ``os.environ`` 读取以上变量:\n"
            "  AL_MODEL_NAME / AL_MODEL_PATH / AL_TASK_TYPE / AL_CLASSES\n"
            "  AL_INPUT_PATH / AL_PROJECT_DIR"
        )
        note.setStyleSheet("color: #64748b; font-size: 11px; padding: 4px;")
        note.setWordWrap(True)
        lay.addWidget(note)

        # ── Buttons ──────────────────────────────────────────────────────────
        # Use plain QPushButtons instead of QDialogButtonBox, which can
        # have button-role ambiguity on Windows that prevents the accepted
        # signal from firing correctly.
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_ok = QPushButton("确定 (OK)")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        lay.addLayout(btn_layout)

        # Load the model list AFTER the dialog is shown so the user sees the
        # dialog immediately instead of a 1-2 s freeze during ``list_models()``.
        # ``singleShot(0, ...)`` defers to the next event-loop iteration, which
        # happens after the window has been painted for the first time.
        from PyQt5.QtCore import QTimer

        QTimer.singleShot(0, self._populate_models)

        # Restore last selection if provided and still valid in this registry.
        if last_input_path:
            p = Path(last_input_path)
            if p.exists():
                self._selected_input = p
                self._input_path_edit.setText(str(p))

        # Store last_model_id for the deferred _populate_models to use.
        self._last_model_id_to_restore = last_model_id

    # ── Model helpers ──────────────────────────────────────────────────────────

    def _populate_models(self) -> None:
        self._model_combo.clear()
        self._model_combo.addItem("— 选择模型 —", None)
        if self._model_registry:
            for model in self._model_registry.list_models():
                label = f"{model.name}  ({model.task})"
                self._model_combo.addItem(label, model)
        self._model_combo.setCurrentIndex(0)

        # Restore the previously selected model (if any) after the list is loaded.
        last_id = getattr(self, "_last_model_id_to_restore", None)
        if last_id:
            for i in range(self._model_combo.count()):
                model = self._model_combo.itemData(i)
                if model and model.id == last_id:
                    self._model_combo.setCurrentIndex(i)
                    break

    def _on_model_changed(self, idx: int) -> None:
        self._selected_model = self._model_combo.currentData()
        if self._selected_model:
            classes_str = ", ".join(self._selected_model.classes[:5])
            if len(self._selected_model.classes) > 5:
                classes_str += " ..."
            info = (
                f"路径: {self._selected_model.path}\n"
                f"类型: {self._selected_model.task}  |  "
                f"类别: {classes_str}"
            )
            self._model_info_label.setText(info)
            self._model_info_label.setStyleSheet(
                "color: #38bdf8; font-size: 12px;"
            )
        else:
            self._model_info_label.setText("未选择模型")
            self._model_info_label.setStyleSheet(
                "color: #94a3b8; font-size: 12px;"
            )

    # ── Input helpers ──────────────────────────────────────────────────────────

    def _on_browse_input(self) -> None:
        paths, selected_filter = QFileDialog.getOpenFileNames(
            self,
            "选择图片或视频",
            "",
            (
                "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tiff);;"
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;"
                "全部文件 (*.*)"
            ),
        )
        if not paths:
            return
        self._selected_input = Path(paths[0])
        self._input_path_edit.setText(str(self._selected_input))

    # ── Public API ────────────────────────────────────────────────────────────

    def selected_model(self):
        """Return the chosen ModelInfo or None."""
        return self._selected_model

    def selected_input(self) -> Path | None:
        """Return the chosen input Path or None."""
        return self._selected_input

    def selected_confidence(self) -> float:
        """Return the chosen confidence threshold (0–1)."""
        return self._conf_spin.value()

    def environment_variables(self) -> dict[str, str]:
        """Return the environment dict to inject into the subprocess."""
        env = {}
        if self._selected_model and self._model_registry:
            env["AL_MODEL_NAME"] = self._selected_model.name
            # Resolve the model's weights path. ``ModelInfo.path`` is stored as a
            # relative path from the models directory. Some older registry entries
            # contain a redundant "models/" prefix (e.g. "models/detect-.../weights/best.pt")
            # while newer ones store the path without it (e.g. "detect-.../weights/best.pt").
            # Normalise the path by stripping any leading "models/" before joining, so
            # both formats resolve to the same absolute location.
            raw_path = self._selected_model.path
            normalized = raw_path.removeprefix("models/").removeprefix("models\\")
            model_abs = (self._model_registry.models_dir / normalized).resolve()
            env["AL_MODEL_PATH"] = str(model_abs)
            env["AL_TASK_TYPE"] = self._selected_model.task
            env["AL_CLASSES"] = ",".join(self._selected_model.classes)
        if self._selected_input:
            env["AL_INPUT_PATH"] = str(self._selected_input.resolve())
            env["AL_PROJECT_DIR"] = str(self._selected_input.parent.resolve())
        env["AL_CONFIDENCE"] = str(self._conf_spin.value())
        return env


# ── ScriptToolPanel ──────────────────────────────────────────────────────────


class ScriptToolPanel(QWidget):
    """Python script editor and runner panel with inline preview."""

    status_changed = pyqtSignal(str)

    def __init__(
        self,
        app_config: AppConfig | None = None,
        config_path: Path | str | None = None,
        tools_dir: Path | str | None = None,
        parent=None,
        *,
        repository: ToolRepository | None = None,
        runner: ScriptRunner | None = None,
        model_registry=None,
    ):
        super().__init__(parent)
        self._repo = repository if repository is not None else ToolRepository(tools_dir)
        self._runner = runner if runner is not None else ScriptRunner(parent=self)
        self._model_registry = model_registry

        self._working_dir = Path.cwd()

        self._current_tool_path: Path | None = None
        self._tool_buttons: dict[Path, QPushButton] = {}
        self._tool_files: list[Path] = []
        self._is_dirty = False
        self._loading_script = False

        # Snapshot of file modification times at script start — used to
        # detect which files the script actually created / modified.
        self._pre_run_mtimes: dict[Path, float] = {}
        # Wall-clock time of the most recent run() call — the secondary
        # signal that picks up files written into subdirectories that
        # weren't present when the snapshot was taken.
        self._run_started_at: float = 0.0
        # Last model-id and input-path chosen in the RunScriptDialog so
        # they are pre-filled on the next run.
        self._last_model_id: str | None = None
        self._last_input_path: str | None = None
        # QMovie handle for animated previews; stopped on each refresh.
        self._preview_movie = None
        # OpenCV-driven video player state. All fields are reset on stop.
        self._video_cap = None
        self._video_timer = None
        self._video_media = None
        self._video_slider = None
        self._video_play_btn = None
        self._video_time_label = None
        self._video_duration = 0.0
        self._video_fps = 25.0
        self._video_frame_count = 0
        self._video_path: Path | None = None
        self._video_seeking = False

        self._repo.ensure_builtin_tools()
        self._repo.migrate_legacy(app_config, Path(config_path) if config_path else None)

        self._init_ui()
        self._connect_signals()
        self._refresh_tool_buttons()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        top_layout = QHBoxLayout()

        # ── Left: tool list ─────────────────────────────────────────────────
        left_group = QGroupBox("快捷小工具")
        left_group.setMaximumWidth(260)
        left_layout = QVBoxLayout(left_group)

        self._tools_scroll = QScrollArea()
        self._tools_scroll.setWidgetResizable(True)

        self._tool_list_widget = QWidget()
        self._tool_list_layout = QVBoxLayout(self._tool_list_widget)
        self._tool_list_layout.setContentsMargins(4, 4, 4, 4)
        self._tool_list_layout.setSpacing(6)
        self._tool_list_layout.addStretch()
        self._tools_scroll.setWidget(self._tool_list_widget)

        self._btn_add_tool = QPushButton("+ 添加工具")
        set_button_role(self._btn_add_tool, "secondary")

        left_layout.addWidget(self._tools_scroll)
        left_layout.addWidget(self._btn_add_tool)

        top_layout.addWidget(left_group)

        # ── Right: editor / preview tab widget ───────────────────────────────
        self._editor_tabs = QTabWidget()
        self._editor_tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Tab 1: editor
        editor_group = QGroupBox("Python 脚本")
        editor_layout = QVBoxLayout(editor_group)

        info_layout = QHBoxLayout()
        self._python_label = QLabel(f"解释器: {sys.executable}")
        self._cwd_label = QLabel()
        self._cwd_label.setToolTip(str(self._working_dir))
        self._save_state_label = QLabel("状态: 已保存")
        self._save_state_label.setStyleSheet(text_style("success"))
        info_layout.addWidget(self._python_label)
        info_layout.addStretch()
        info_layout.addWidget(self._cwd_label)
        info_layout.addSpacing(12)
        info_layout.addWidget(self._save_state_label)
        editor_layout.addLayout(info_layout)

        self._editor = QPlainTextEdit()
        self._editor.setPlaceholderText('在这里输入 Python 脚本，然后点击"执行脚本"。')
        mono_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self._editor.setFont(mono_font)
        editor_layout.addWidget(self._editor)

        btn_layout = QHBoxLayout()
        self._btn_run = QPushButton("执行脚本")
        set_button_role(self._btn_run, "primary")
        self._btn_stop = QPushButton("停止执行")
        set_button_role(self._btn_stop, "danger")
        self._btn_stop.setEnabled(False)
        self._btn_clear = QPushButton("清空输出")
        set_button_role(self._btn_clear, "secondary")
        self._save_hint_label = QLabel("快捷键: Ctrl+S 保存")
        self._save_hint_label.setStyleSheet(text_style("hint"))

        btn_layout.addWidget(self._btn_run)
        btn_layout.addWidget(self._btn_stop)
        btn_layout.addWidget(self._btn_clear)
        btn_layout.addStretch()
        btn_layout.addWidget(self._save_hint_label)
        editor_layout.addLayout(btn_layout)

        self._editor_tabs.addTab(editor_group, "编辑")

        # Tab 2: preview
        self._preview_tab = self._build_preview_tab()
        self._editor_tabs.addTab(self._preview_tab, "预览")

        top_layout.addWidget(self._editor_tabs, 1)

        layout.addLayout(top_layout, 2)

        # ── Bottom: output ───────────────────────────────────────────────────
        output_group = QGroupBox("执行输出")
        output_layout = QVBoxLayout(output_group)
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(mono_font)
        output_layout.addWidget(self._output)
        layout.addWidget(output_group, 1)

        self._save_shortcut = QShortcut(QKeySequence.Save, self)
        self.set_working_directory(self._working_dir)

    def _build_preview_tab(self) -> QWidget:
        """Build the image preview tab shown in the editor tab widget."""
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(0, 0, 0, 0)

        # Toolbar: refresh button + path label
        toolbar = QHBoxLayout()
        self._preview_label = QLabel("运行脚本后，预览将自动显示")
        self._preview_label.setStyleSheet("color: #64748b; font-size: 12px;")
        self._btn_refresh_preview = QPushButton("刷新预览")
        set_button_role(self._btn_refresh_preview, "secondary")
        self._btn_refresh_preview.clicked.connect(self._refresh_preview)
        toolbar.addWidget(self._preview_label, 1)
        toolbar.addWidget(self._btn_refresh_preview)
        lay.addLayout(toolbar)

        # Scroll area holding the image
        self._preview_scroll = QScrollArea()
        self._preview_scroll.setWidgetResizable(True)
        self._preview_scroll.setAlignment(Qt.AlignCenter)
        # Dark background so the image pops
        self._preview_scroll.setStyleSheet(
            "QScrollArea { background-color: #0f172a; border: none; }"
        )

        self._preview_image_label = QLabel()
        self._preview_image_label.setAlignment(Qt.AlignCenter)
        self._preview_image_label.setStyleSheet(
            "QLabel { background-color: #0f172a; color: #475569; "
            "font-size: 13px; }"
        )
        self._preview_image_label.setText(
            "  暂无预览  \n\n运行脚本后，预览将自动显示"
        )
        self._preview_scroll.setWidget(self._preview_image_label)

        lay.addWidget(self._preview_scroll)
        return tab

    def _connect_signals(self) -> None:
        self._btn_add_tool.clicked.connect(self._on_add_tool_clicked)
        self._btn_run.clicked.connect(self._on_run_clicked)
        self._btn_stop.clicked.connect(self.stop_script)
        self._btn_clear.clicked.connect(self._output.clear)
        self._editor.textChanged.connect(self._on_editor_text_changed)
        self._save_shortcut.activated.connect(self._on_save_shortcut)

        self._runner.output.connect(self._append_text)
        self._runner.process_error.connect(self._on_process_error)
        self._runner.finished.connect(self._on_script_finished)

    # ── Tool list ────────────────────────────────────────────────────────────

    def _clear_tool_buttons(self) -> None:
        while self._tool_list_layout.count():
            item = self._tool_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _refresh_tool_buttons(self, select_path: Path | None = None) -> None:
        self._tool_files = self._repo.list_tools()
        self._clear_tool_buttons()
        self._tool_buttons = {}

        if not self._tool_files:
            default_path = self._repo.create_tool("新建工具")
            self._tool_files = [default_path]

        for tool_path in self._tool_files:
            btn = QPushButton(tool_path.stem)
            btn.setCheckable(True)
            set_button_role(btn, "list-item")
            btn.setToolTip(str(tool_path))
            btn.clicked.connect(
                lambda _checked=False, p=tool_path: self._on_tool_button_clicked(p)
            )
            self._tool_buttons[tool_path] = btn
            self._tool_list_layout.addWidget(btn)

        self._tool_list_layout.addStretch()

        target = select_path
        if target is None or target not in self._tool_buttons:
            if self._current_tool_path in self._tool_buttons:
                target = self._current_tool_path
            else:
                target = self._tool_files[0]

        self._load_tool(target)

    def _set_selected_tool_button(self) -> None:
        for path, btn in self._tool_buttons.items():
            is_current = path == self._current_tool_path
            btn.setChecked(is_current)

    def _update_button_labels(self) -> None:
        for path, btn in self._tool_buttons.items():
            text = path.stem
            if self._is_dirty and path == self._current_tool_path:
                text += " *"
            btn.setText(text)

    def _set_dirty(self, dirty: bool) -> None:
        self._is_dirty = dirty
        if dirty:
            self._save_state_label.setText("状态: 未保存*")
            self._save_state_label.setStyleSheet(text_style("warning"))
        else:
            self._save_state_label.setText("状态: 已保存")
            self._save_state_label.setStyleSheet(text_style("success"))
        self._update_button_labels()

    def _load_tool(self, tool_path: Path) -> None:
        try:
            script = self._repo.load_tool(tool_path)
        except OSError as exc:
            QMessageBox.warning(
                self, "读取失败", f"无法读取脚本: {tool_path.name}\n{exc}"
            )
            return

        self._loading_script = True
        try:
            self._editor.setPlainText(script)
        finally:
            self._loading_script = False

        self._current_tool_path = tool_path
        self._set_dirty(False)
        self._set_selected_tool_button()
        self.status_changed.emit(f"已加载工具: {tool_path.stem}")

    def _maybe_resolve_unsaved(self) -> bool:
        if not self._is_dirty:
            return True

        reply = QMessageBox.question(
            self,
            "未保存更改",
            "当前脚本有未保存修改，是否先保存？",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )

        if reply == QMessageBox.Save:
            return self._save_current_script()
        if reply == QMessageBox.Discard:
            self._set_dirty(False)
            return True
        return False

    def _on_tool_button_clicked(self, tool_path: Path) -> None:
        if tool_path == self._current_tool_path:
            self._set_selected_tool_button()
            return

        if not self._maybe_resolve_unsaved():
            self._set_selected_tool_button()
            return

        self._load_tool(tool_path)

    def _on_add_tool_clicked(self) -> None:
        if self._runner.is_running:
            QMessageBox.information(
                self, "提示", "脚本执行中，停止后再添加工具"
            )
            return

        if not self._maybe_resolve_unsaved():
            return

        name, ok = QInputDialog.getText(self, "添加工具", "请输入工具名称:")
        if not ok:
            return

        existing = self._repo.find_tool(name)
        if existing is not None:
            QMessageBox.information(
                self, "工具已存在", "工具已存在，已切换到该工具"
            )
            self._refresh_tool_buttons(select_path=existing)
            return

        tool_path = self._repo.create_tool(name)
        if tool_path is None:
            QMessageBox.warning(self, "名称无效", "工具名称不能为空")
            return

        self._refresh_tool_buttons(select_path=tool_path)
        self.status_changed.emit(f"已添加工具: {tool_path.stem}")

    def _on_editor_text_changed(self) -> None:
        if self._loading_script:
            return
        if self._current_tool_path is None:
            return
        if not self._is_dirty:
            self._set_dirty(True)

    def _on_save_shortcut(self) -> None:
        self._save_current_script()

    def _save_current_script(self) -> bool:
        if self._current_tool_path is None:
            QMessageBox.warning(self, "保存失败", "当前没有可保存的工具")
            return False

        script = self._editor.toPlainText()
        try:
            self._repo.save_tool(self._current_tool_path, script)
        except OSError as exc:
            QMessageBox.warning(
                self,
                "保存失败",
                f"无法保存脚本: {self._current_tool_path.name}\n{exc}",
            )
            return False

        self._set_dirty(False)
        self.status_changed.emit(f"已保存工具: {self._current_tool_path.stem}")
        return True

    def set_working_directory(self, path: Path | str | None) -> None:
        """Set script working directory shown in UI and used for execution."""
        if path is None:
            self._working_dir = Path.cwd()
        else:
            candidate = Path(path)
            self._working_dir = candidate if candidate.exists() else Path.cwd()
        cwd_text = f"工作目录: {self._working_dir}"
        self._cwd_label.setText(cwd_text)
        self._cwd_label.setToolTip(str(self._working_dir))

    # ── Script execution ──────────────────────────────────────────────────────

    def _on_run_clicked(self) -> None:
        if self._runner.is_running:
            return

        script = self._editor.toPlainText().strip()
        if not script:
            QMessageBox.warning(
                self, "脚本为空", "请先输入 Python 脚本内容"
            )
            return

        # ── Show config dialog ─────────────────────────────────────────────
        dialog = RunScriptDialog(
            model_registry=self._model_registry,
            parent=self,
            last_model_id=self._last_model_id,
            last_input_path=self._last_input_path,
        )
        result = dialog.exec_()
        if result != QDialog.Accepted:
            return

        extra_env = dialog.environment_variables()
        model = dialog.selected_model()
        input_path = dialog.selected_input()

        # Persist so the next dialog pre-fills these values.
        self._last_model_id = model.id if model else None
        self._last_input_path = str(input_path) if input_path else None

        # ── Snapshot file modification times before run ────────────────────
        self._pre_run_mtimes = {}
        work_dir = Path(self._working_dir)
        if work_dir.exists():
            for p in work_dir.iterdir():
                try:
                    self._pre_run_mtimes[p] = p.stat().st_mtime
                except OSError:
                    pass
        self._run_started_at = time.time()

        started = self._runner.run(script, self._working_dir, extra_env=extra_env)

        # ── Log header ────────────────────────────────────────────────────
        self._append_line(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] 开始执行脚本"
        )
        self._append_line(f"工作目录: {self._working_dir}")
        if model:
            self._append_line(f"模型: {model.name}  ({model.task})")
        if input_path:
            self._append_line(f"输入: {input_path}")
        if extra_env:
            injected = [
                f"  {k}={v}" for k, v in extra_env.items()
            ]
            self._append_line("注入环境变量:\n" + "\n".join(injected))
        self._append_line("-" * 50)

        if not started:
            self._append_line("脚本启动失败")
            self._set_running_state(False)
            self.status_changed.emit("脚本启动失败")
            return

        self._set_running_state(True)
        self.status_changed.emit("脚本执行中...")

    def stop_script(self) -> None:
        """Stop currently running script if needed."""
        if not self._runner.is_running:
            return

        self._append_line("\n[系统] 正在停止脚本...")
        self._runner.stop()

    def _on_process_error(self, error_name: str) -> None:
        self._append_line(f"\n[系统] 进程错误: {error_name}")

    def _on_script_finished(self, exit_code: int, normal_exit: bool) -> None:
        status = "正常退出" if normal_exit else "异常终止"
        self._append_line(
            f"\n[系统] 脚本结束: {status} | 退出码: {exit_code}"
        )

        self._set_running_state(False)
        self.status_changed.emit(f"脚本执行完成 (退出码: {exit_code})")

        # ── Auto-refresh preview ─────────────────────────────────────────
        self._refresh_preview()

    def _set_running_state(self, running: bool) -> None:
        self._btn_run.setEnabled(not running)
        self._btn_stop.setEnabled(running)
        self._editor.setReadOnly(running)
        self._btn_add_tool.setEnabled(not running)
        for btn in self._tool_buttons.values():
            btn.setEnabled(not running)

    def _append_text(self, text: str) -> None:
        if not text:
            return
        self._output.moveCursor(QTextCursor.End)
        self._output.insertPlainText(text)
        self._output.moveCursor(QTextCursor.End)

    def _append_line(self, text: str) -> None:
        self._output.appendPlainText(text)
        self._output.moveCursor(QTextCursor.End)

    # ── Preview ───────────────────────────────────────────────────────────────

    def _refresh_preview(self) -> None:
        """Scan the working directory for newly created/modified result files
        and display the most recent one in the preview tab.

        Heuristic: pick the newest file (by mtime) whose mtime is >= the
        start of this script run. If no such file exists, show the most
        recently modified image in the directory. Image files are displayed
        inline; video files are shown as an info card with metadata (path,
        size, duration, frame count when available) since embedding a video
        player would require platform-specific backends.

        Recurses one level deep so results written into common output folders
        (e.g. ``data-validation/``, ``crops/``) are still picked up — most
        built-in scripts write their outputs into such subfolders.
        """
        work_dir = Path(self._working_dir)
        if not work_dir.exists():
            self._set_preview_placeholder("工作目录不存在")
            return

        IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}
        VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
        MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS

        candidates: list[tuple[float, Path]] = []
        try:
            # One level deep: top-level files + immediate subdirectories
            # (e.g. data-validation/, crops/). Going deeper tends to surface
            # per-class subfolders or unrelated assets, which would drown out
            # the actual result file.
            roots = [work_dir] + [p for p in work_dir.iterdir() if p.is_dir()]
            for root in roots:
                for p in root.iterdir():
                    if not p.is_file():
                        continue
                    if p.suffix.lower() not in MEDIA_EXTS:
                        continue
                    try:
                        mtime = p.stat().st_mtime
                    except OSError:
                        continue
                    # Treat a file as new this run when either:
                    #   (a) it was present at start and has been modified, OR
                    #   (b) it appears under a subdirectory that wasn't snapshotted
                    #       (e.g. data-validation/) and its mtime is after start.
                    pre = self._pre_run_mtimes.get(p)
                    is_new = (
                        (pre is not None and mtime > pre)
                        or (pre is None and self._run_started_at > 0 and mtime >= self._run_started_at - 1.0)
                    )
                    # Prefer new media files; otherwise fall back to the most
                    # recently modified one (gives a useful preview even on
                    # re-open).
                    if is_new:
                        candidates.append((mtime + 10_000, p))  # boost post-run files
                    else:
                        candidates.append((mtime, p))
        except OSError:
            self._set_preview_placeholder("无法读取工作目录")
            return

        if not candidates:
            self._set_preview_placeholder("工作目录中没有找到可预览的媒体文件")
            return

        # Pick the most recent.
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_path = candidates[0][1]

        if best_path.suffix.lower() in VIDEO_EXTS:
            self._display_video_info(best_path)
            return

        # Stop any previously-running QMovie (for gif previews).
        if self._preview_movie is not None:
            self._preview_movie.stop()
            self._preview_movie = None

        img = QImage()
        if not img.load(str(best_path)):
            self._set_preview_placeholder(f"无法加载文件: {best_path.name}")
            return

        self._display_image(img, best_path)

    def _display_image(self, img: QImage, path: Path) -> None:
        """Show an image in the preview label, scaled to fit the scroll area."""
        # Switch to the Preview tab
        self._editor_tabs.setCurrentWidget(self._preview_tab)

        # Make sure the scroll area shows the regular image-label widget
        # (the video player replaces it on the fly).
        self._ensure_preview_image_widget()

        # Scale to fit the scroll area width (minus some padding)
        scroll_width = self._preview_scroll.viewport().width()
        if scroll_width <= 0:
            scroll_width = 600
        scaled = img.scaledToWidth(
            scroll_width,
            Qt.SmoothTransformation,
        )
        pixmap = QPixmap.fromImage(scaled)

        self._preview_image_label.setPixmap(pixmap)
        self._preview_image_label.setText("")
        self._preview_image_label.setAlignment(Qt.AlignCenter)

        self._preview_label.setText(
            f"预览: {path.name}  ({img.width()}×{img.height()})"
        )
        self._preview_label.setStyleSheet("color: #38bdf8; font-size: 12px;")

        self._preview_scroll.horizontalScrollBar().setValue(0)
        self._preview_scroll.verticalScrollBar().setValue(0)

    def _ensure_preview_image_widget(self) -> None:
        """Re-attach the original ``_preview_image_label`` to the scroll area
        if the video player has swapped it out for a player widget."""
        if self._preview_scroll.widget() is self._preview_image_label:
            return
        old = self._preview_scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self._preview_scroll.setWidget(self._preview_image_label)
        self._preview_scroll.setWidgetResizable(True)

    def _display_video_info(self, path: Path) -> None:
        """Show a video result preview card with a thumbnail strip and an
        open-with-system-player button.

        Inline playback is implemented separately (``_video_*``) for the
        *preview* loop, but here we always render a deterministic static
        card so the user can see the result exists, where it is, and can
        launch their OS default player to actually watch it. A thumbnail
        strip (6 evenly-spaced frames) gives a visual smoke-test without
        requiring QtMultimedia.
        """
        # Tear down any previous player so we don't leak cv2 readers.
        self._stop_video_player()

        self._editor_tabs.setCurrentWidget(self._preview_tab)

        # Always restore the standard image-label as the host for the card.
        self._ensure_preview_image_widget()

        size_bytes = path.stat().st_size if path.exists() else 0
        size_mb = size_bytes / (1024 * 1024)

        # Gather cv2 metadata + thumbnail strip; fail quietly if missing.
        fps, frame_count, w, h, duration_s, thumb_b64 = (
            0.0, 0, 0, 0, 0.0, ""
        )
        cv2_error = ""
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if fps > 0:
                    duration_s = frame_count / fps

                # Sample 6 evenly-spaced frames as a preview strip.
                sample_count = 6
                if frame_count > 1:
                    indices = [
                        int(i * (frame_count - 1) / (sample_count - 1))
                        for i in range(sample_count)
                    ]
                else:
                    indices = [0]

                import base64
                from PyQt5.QtCore import QBuffer, QIODevice

                frames: list[QImage] = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    fh, fw = rgb.shape[:2]
                    qimg = QImage(
                        rgb.data, fw, fh, rgb.strides[0], QImage.Format_RGB888
                    ).copy()
                    # Downscale frames so the strip stays manageable.
                    qimg = qimg.scaledToWidth(220, Qt.SmoothTransformation)
                    frames.append(qimg)

                if frames:
                    cols = len(frames)
                    tw = max(f.width() for f in frames)
                    th = max(f.height() for f in frames)
                    grid = QImage(tw * cols, th, QImage.Format_RGB888)
                    grid.fill(Qt.black)
                    from PyQt5.QtGui import QPainter
                    painter = QPainter(grid)
                    for i, f in enumerate(frames):
                        painter.drawImage(i * tw, 0, f)
                    painter.end()
                    pix = QPixmap.fromImage(grid)
                    buf = QBuffer()
                    buf.open(QIODevice.WriteOnly)
                    pix.save(buf, "PNG")
                    thumb_b64 = base64.b64encode(bytes(buf.data())).decode("ascii")
            cap.release()
        except Exception as exc:
            cv2_error = str(exc)

        thumb_html = (
            f"<img src='data:image/png;base64,{thumb_b64}' "
            f"style='display:block; margin: 12px auto 0; max-width:100%; "
            f"border:1px solid #334155; border-radius:6px;'/>"
            if thumb_b64
            else ""
        )
        cv2_warn = (
            f"<div style='color:#94a3b8; font-size:11px; margin-top:8px;'>"
            f"无法读取视频元数据（{cv2_error}）</div>"
            if cv2_error
            else ""
        )

        html = f"""
<div style="font-family: 'Segoe UI', sans-serif; color:#e2e8f0;">
  <div style="font-size:16px; color:#38bdf8; font-weight:600;">
    🎬 视频结果已生成
  </div>
  <div style="margin-top:10px; font-size:12px; line-height:1.7;">
    <div><b>文件:</b> {path.name}</div>
    <div><b>路径:</b> {path}</div>
    <div><b>大小:</b> {size_mb:.2f} MB</div>
    <div><b>分辨率:</b> {w}×{h} &nbsp;·&nbsp;
         <b>帧率:</b> {fps:.2f} fps &nbsp;·&nbsp;
         <b>总帧:</b> {frame_count} &nbsp;·&nbsp;
         <b>时长:</b> {duration_s_str(duration_s)}</div>
  </div>
  {thumb_html}
  {cv2_warn}
  <div style="margin-top:14px; font-size:11px; color:#94a3b8;">
    点击下方按钮调用系统默认播放器打开视频进行回放检查。
  </div>
</div>
"""

        label = QLabel()
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        label.setText(html)
        label.setStyleSheet(
            "QLabel { background-color: #0f172a; color: #e2e8f0; font-size: 12px; "
            "padding: 18px; border: 1px solid #1e293b; border-radius: 8px; }"
        )

        # Wrap in a small layout so we can stack the open button under the label.
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        v.addWidget(label)

        from PyQt5.QtWidgets import QPushButton
        open_btn = QPushButton("▶  使用系统播放器打开")
        open_btn.setFixedHeight(34)
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.setStyleSheet(
            "QPushButton { background:#2563eb; color:white; border:none; "
            "border-radius:6px; font-size:13px; font-weight:600; }"
            "QPushButton:hover { background:#1d4ed8; }"
            "QPushButton:pressed { background:#1e40af; }"
        )
        open_btn.clicked.connect(lambda: self._open_video_in_external_player(path))
        v.addWidget(open_btn)

        old = self._preview_scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self._preview_scroll.setWidget(host)
        self._preview_scroll.setWidgetResizable(True)

        self._preview_label.setText(
            f"预览: {path.name}  ({size_mb:.2f} MB, {duration_s_str(duration_s)})"
        )
        self._preview_label.setStyleSheet("color: #38bdf8; font-size: 12px;")

    def _open_video_in_external_player(self, path: Path) -> None:
        """Open ``path`` with the OS-registered default video player."""
        import subprocess
        import sys as _sys
        import os as _os

        try:
            if _sys.platform.startswith("win"):
                _os.startfile(str(path))  # type: ignore[attr-defined]
            elif _sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            QMessageBox.warning(
                self,
                "无法打开",
                f"调用系统播放器失败: {exc}\n\n文件位置: {path}",
            )

    # ── (legacy inline player loop, retained but unused — left here for
    #    occasional debugging when an operator wants to scrub frames in-app
    #    without leaving the dock. Disabled by default.) ──────────────────────
    def _advance_video_frame(self) -> None:  # pragma: no cover - retained
        return

    def _toggle_play_pause(self) -> None:  # pragma: no cover - retained
        return

    def _on_slider_pressed(self) -> None:  # pragma: no cover - retained
        return

    def _on_slider_released(self) -> None:  # pragma: no cover - retained
        return

    def _on_slider_moved(self, value: int) -> None:  # pragma: no cover
        return

    def _update_video_time_label(self, frame_idx: int) -> None:  # pragma
        return

    def _stop_video_player(self) -> None:
        """Clear any retained player state. Inline playback is no longer used
        but the indirection keeps widget lifecycle manageable if it returns."""
        if getattr(self, "_video_timer", None) is not None:
            try:
                self._video_timer.stop()
            except Exception:
                pass
            self._video_timer = None
        if getattr(self, "_video_cap", None) is not None:
            try:
                self._video_cap.release()
            except Exception:
                pass
            self._video_cap = None
        self._video_media = None
        self._video_slider = None
        self._video_play_btn = None
        self._video_time_label = None
        self._video_seeking = False
        # Restore the original image-preview widget if we swapped it out.
        if (
            getattr(self, "_preview_scroll", None) is not None
            and getattr(self, "_preview_image_label", None) is not None
            and self._preview_scroll.widget() is not self._preview_image_label
        ):
            old = self._preview_scroll.takeWidget()
            if old is not None:
                old.deleteLater()
            self._preview_scroll.setWidget(self._preview_image_label)
            self._preview_scroll.setWidgetResizable(True)

    def _show_video_fallback(
        self,
        path: Path,
        size_mb: float,
        error: str = "",
    ) -> None:
        """Used by older callers that hit this path before cv2 metadata runs."""
        self._display_video_info(path)

    # ── (Inline player loop end-region — methods kept here in case any
    #    external code calls them; they're no-ops now that the preview
    #    opens the system player instead.) ───────────────────────────────────
    def _advance_video_frame(self) -> None:  # legacy no-op
        return

    def _render_video_frame(self, qimg: QImage) -> None:  # legacy no-op
        return

    def _toggle_play_pause(self) -> None:  # legacy no-op
        return

    def _on_slider_pressed(self) -> None:  # legacy no-op
        return

    def _on_slider_released(self) -> None:  # legacy no-op
        return

    def _on_slider_moved(self, value: int) -> None:  # legacy no-op
        return

    def _update_video_time_label(self, frame_idx: int) -> None:  # legacy
        return

    def _stop_video_player(self) -> None:
        if getattr(self, "_video_timer", None) is not None:
            self._video_timer.stop()
            self._video_timer = None
        if getattr(self, "_video_cap", None) is not None:
            self._video_cap.release()
            self._video_cap = None
        self._video_media = None
        self._video_slider = None
        self._video_play_btn = None
        self._video_time_label = None
        self._video_seeking = False
        # Restore the original image-preview widget if we swapped it out.
        if self._preview_scroll.widget() is not self._preview_image_label:
            old = self._preview_scroll.takeWidget()
            if old is not None:
                old.deleteLater()
            self._preview_scroll.setWidget(self._preview_image_label)
            self._preview_scroll.setWidgetResizable(True)

    def _show_video_fallback(
        self,
        path: Path,
        size_mb: float,
        error: str = "",
    ) -> None:
        """Static info card used when cv2 is unavailable or read fails."""
        self._ensure_preview_image_widget()

        label = QLabel(
            f"<div style='font-size:15px; color:#38bdf8; font-weight:600;'>"
            f"🎬 视频结果</div>"
            f"<div style='margin-top:8px; color:#e2e8f0;'>"
            f"<b>文件:</b> {path.name}<br>"
            f"<b>路径:</b> {path}<br>"
            f"<b>大小:</b> {size_mb:.2f} MB"
            f"</div>"
            f"<div style='margin-top:10px; color:#94a3b8; font-size:11px;'>"
            f"无法播放 ({error})</div>"
        )
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        label.setStyleSheet(
            "QLabel { background-color: #0f172a; color: #e2e8f0; font-size: 12px; "
            "padding: 24px; border: 1px solid #1e293b; border-radius: 8px; }"
        )
        old = self._preview_scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self._preview_scroll.setWidget(label)
        self._preview_scroll.setWidgetResizable(True)
        self._preview_label.setText(
            f"预览: {path.name}  ({size_mb:.2f} MB)"
        )
        self._preview_label.setStyleSheet("color: #38bdf8; font-size: 12px;")

    def _set_preview_placeholder(self, message: str) -> None:
        self._preview_image_label.clear()
        self._preview_image_label.setText(f"\n  {message}  \n")
        self._preview_image_label.setAlignment(Qt.AlignCenter)
        self._preview_image_label.setStyleSheet(
            "QLabel { background-color: #0f172a; color: #475569; font-size: 13px; }"
        )
        self._preview_label.setText("运行脚本后，预览将自动显示")
        self._preview_label.setStyleSheet("color: #64748b; font-size: 12px;")

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def prepare_close(self) -> bool:
        """Handle save/cancel flow before app close. Return True if close can continue."""
        if self._runner.is_running:
            self.stop_script()
        return self._maybe_resolve_unsaved()

    def closeEvent(self, event) -> None:
        if not self.prepare_close():
            event.ignore()
            return
        if self._preview_movie is not None:
            self._preview_movie.stop()
            self._preview_movie = None
        self._stop_video_player()
        self._runner.cleanup_temp()
        super().closeEvent(event)


# ── Module-level helpers ────────────────────────────────────────────────────────


def _qpixmap_to_base64(pixmap: QPixmap) -> str:
    """Encode a QPixmap to a base64 PNG data URI for use in <img src=...>."""
    import base64
    from PyQt5.QtCore import QBuffer, QIODevice
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    pixmap.save(buffer, "PNG")
    return base64.b64encode(buffer.data()).decode("ascii")


def duration_s_str(seconds: float | None) -> str:
    """Format seconds as ``HH:MM:SS`` or ``MM:SS``."""
    if seconds is None:
        return ""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def cv2_bgr_to_rgb(frame):
    """Convert an OpenCV BGR frame (numpy.ndarray) to an RGB array."""
    import cv2
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def cv2_pos_frames():
    """Lazy attribute lookup for cv2.CAP_PROP_POS_FRAMES (kept here so
    the player loop doesn't pay the import cost on every refresh)."""
    import cv2
    return cv2.CAP_PROP_POS_FRAMES
