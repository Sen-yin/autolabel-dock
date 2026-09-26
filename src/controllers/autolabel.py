"""AutoLabel controller — the deep module for the auto-label pipeline.

Owns everything between "label this image / label this batch" and persisted
label records: task_type dispatch, sync-vs-async thread decision, worker
lifecycle, conflict merging (via ``core.autolabel.merge_predictions``),
classify apply decisions (via ``core.autolabel.decide_classify_apply``), and
per-image persistence through the shared ``LabelStore``. MainWindow shrinks
to two one-line forwarding slots plus a progress-dialog shell driven by this
controller's signals.

Design decisions (see docs/adr/0002-autolabel-orchestration.md):
- Thread decision by injected predicate ``slow_backend_active`` (MainWindow
  injects ``la_ctrl.is_active``) — the controller never depends on
  LocateAnythingController. When the predicate is False the single-image
  path stays fully synchronous (YOLO contract; tests pin it).
- The view is driven only through LabelPanel's public face
  (``add_auto_annotations`` / ``add_auto_class_prediction`` /
  ``begin·end_bulk_auto_label_update`` / ``set_auto_label_busy`` /
  ``get_current_image_path`` / ``get_unlabeled_image_paths`` /
  ``set_image_status`` / ``reload_current``), injected via ``set_context``.
  The LabelPanel type is imported under TYPE_CHECKING only.
- Interactive decision points are injectable callables (``scope_chooser``,
  ``class_register_confirm``); the defaults are the original Qt dialogs.
  The BatchProgressDialog stays in MainWindow, driven by
  ``batch_started`` / ``batch_progress`` / ``batch_finished``; cancellation
  flows back through ``cancel_batch()``.
- Supersede rebuilds go through the ``classes_registered`` signal —
  MainWindow owns the panel/project lifecycle and performs
  ``set_project(project, discard_pending=True)``.

Batch choreography (must not change — see the LabelStore contract):
classify only → ``begin_bulk_auto_label_update`` (before ``worker.start()``)
→ per-image application on the main thread via queued worker signals (disk
writes happen HERE, not in the worker) → on ``finished_ok``: summary +
``reload_current()`` when an image is focused (relies on the view's
clean-skip so stale memory can't clobber the records the batch just wrote)
→ on worker ``finished`` (always fires, delivered after ``finished_ok``):
``end_bulk`` (unconditional; no-op for detect) + cancel detection.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QInputDialog, QMessageBox, QWidget

from src.core.autolabel import (
    STATUS_SKIPPED_CONFIRMED,
    STATUS_UNRECOGNIZED,
    InferenceParams,
    decide_classify_apply,
    merge_predictions,
)
from src.core.label_store import LabelStore
from src.core.project import ProjectManager
from src.utils.workers import BatchPredictWorker, SinglePredictWorker

if TYPE_CHECKING:  # controller never imports src.ui at runtime
    from src.controllers.model import ModelController
    from src.controllers.project import ProjectController
    from src.ui.label_panel import LabelPanel

logger = logging.getLogger(__name__)


class AutoLabelController(QObject):
    """Orchestrates single/batch auto-labeling across detect/pose/classify.

    Signals:
        status_message(str): Chinese status-bar text (moved verbatim from the
            pre-controller MainWindow slots).
        busy_changed(bool): a slow-backend single-image inference went
            in-flight / finished (mirrors ``LabelPanel.set_auto_label_busy``,
            which the controller also drives directly).
        batch_started(int): a batch run started with ``total`` images —
            MainWindow builds the progress dialog.
        batch_progress(int, int): per-image progress (current, total).
        batch_finished(str): terminal summary for every batch outcome
            (completed / failed / cancelled) — MainWindow closes the dialog
            and shows the text.
        classes_registered(): new classes were registered from model output —
            MainWindow must rebuild via
            ``set_project(project, discard_pending=True)`` (Supersede).
    """

    status_message = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)
    batch_started = pyqtSignal(int)
    batch_progress = pyqtSignal(int, int)
    batch_finished = pyqtSignal(str)
    classes_registered = pyqtSignal()

    def __init__(
        self,
        model_ctrl: "ModelController",
        project_ctrl: "ProjectController",
        label_store: LabelStore,
        slow_backend_active: Callable[[], bool],
        params_provider: Callable[[], InferenceParams],
        parent_widget: QWidget | None,
        scope_chooser: Callable[
            [list[Path], list[Path]], list[Path] | None
        ] | None = None,
        class_register_confirm: Callable[
            [Sequence], list[str] | None
        ] | None = None,
    ):
        super().__init__(parent_widget)
        self._model_ctrl = model_ctrl
        self._project_ctrl = project_ctrl
        self._label_store = label_store
        self._slow_backend_active = slow_backend_active
        self._params_provider = params_provider
        self._parent = parent_widget
        self._scope_chooser = scope_chooser or self._default_scope_chooser
        self._class_register_confirm = (
            class_register_confirm or self._default_class_register_confirm
        )

        self._project: ProjectManager | None = None
        self._label_panel: "LabelPanel | None" = None
        # Background worker for single-image inference on slow backends (LA).
        self._single_worker: SinglePredictWorker | None = None
        self._batch_worker: BatchPredictWorker | None = None
        self._batch_skipped = 0
        self._batch_failed = 0
        self._batch_conflicts_dropped = 0
        # True once a terminal summary (completed / failed) was emitted for
        # the current batch; the worker-finished slot uses it to detect the
        # cancel path (neither finished_ok nor error fired).
        self._batch_summary_emitted = False

    # ── Context ──────────────────────────────────────────────────

    def set_context(self, project: ProjectManager, label_panel: "LabelPanel") -> None:
        """Bind the open project and the label panel (its public face only)."""
        self._project = project
        self._label_panel = label_panel

    # ── Single image ─────────────────────────────────────────────

    def label_current(self) -> None:
        """Auto-label the currently focused image (task_type dispatch inside)."""
        if not self._label_panel or not self._project:
            return
        # Re-entrancy guard: a slow-backend (LA) single-image inference is
        # still running on the worker thread. Disabling the toolbar buttons
        # blocks mouse clicks, but the Shift+A shortcut emits the request
        # directly, so guard here to avoid stacking overlapping workers (and
        # leaking the in-flight QThread reference).
        if self._single_worker is not None and self._single_worker.isRunning():
            return
        img_path = self._label_panel.get_current_image_path()
        if not img_path:
            return
        if self._project.config.task_type == "classify":
            self._label_current_classify(img_path)
            return

        params = self._params_provider()
        # Slow backends (LocateAnything) block for seconds inside predict().
        # On a single-GPU box the X server shares the same card, so running
        # that on the Qt/X event loop stalls — and can crash — the desktop.
        # Route slow backends through a background worker; keep YOLO
        # synchronous (fast, and existing tests depend on the sync path).
        if self._slow_backend_active():
            self._start_async_single(img_path, params)
            return

        annotations = self._model_ctrl.predict_single(
            img_path,
            self._project.config.classes,
            conf=params.conf,
            iou=params.iou,
            class_match_mode=params.class_match_mode,
        )
        self._apply_single_result(annotations, params.overlap_iou)

    def _label_current_classify(self, img_path: Path) -> None:
        result = self._model_ctrl.predict_single_classify(
            img_path, self._project.config.classes,
        )
        if result is None:
            self.status_message.emit(STATUS_UNRECOGNIZED)
            return
        raw_name, conf = result
        reg = self._project_ctrl.register_auto_class(raw_name)
        decision = decide_classify_apply(
            reg.action, reg.applied_name, raw_name, conf,
            self._project.config.classes,
        )
        if decision.rebuild_classes:
            # Refresh class buttons & filter combo so the new class is
            # visible. Supersede rebuild — MainWindow performs
            # set_project(project, discard_pending=True).
            self.classes_registered.emit()
        if not decision.apply:
            self.status_message.emit(decision.status_text)
            return
        applied = self._label_panel.add_auto_class_prediction(
            img_path, decision.class_name, conf,
        )
        if not applied:
            self.status_message.emit(STATUS_SKIPPED_CONFIRMED)
        else:
            self.status_message.emit(decision.status_text)

    def _start_async_single(self, img_path: Path, params: InferenceParams) -> None:
        """Run single-image inference on a worker thread (slow backends).

        Disables the auto-label buttons for the duration so the user can't
        stack overlapping inference requests, then applies the result on the
        main thread via the same ``add_auto_annotations`` path as the sync
        flow.
        """
        worker = self._model_ctrl.create_single_predict_worker(
            img_path,
            self._project.config.classes,
            conf=params.conf,
            iou=params.iou,
            class_match_mode=params.class_match_mode,
        )
        if worker is None:
            return
        self._set_busy(True)
        self.status_message.emit("自动标注进行中…")
        self._single_worker = worker
        # Bind overlap_iou for the result slot without re-reading the panel.
        worker.done.connect(
            lambda anns: self._apply_single_result(anns, params.overlap_iou)
        )
        worker.error.connect(self._on_single_error)
        worker.finished.connect(self._on_single_finished)
        worker.start()

    def _on_single_error(self, message: str) -> None:
        self.status_message.emit("自动标注失败")
        QMessageBox.warning(self._parent, "自动标注失败", message)

    def _on_single_finished(self) -> None:
        self._set_busy(False)
        self._single_worker = None

    def _set_busy(self, busy: bool) -> None:
        if self._label_panel:
            self._label_panel.set_auto_label_busy(busy)
        self.busy_changed.emit(busy)

    def _apply_single_result(self, annotations, overlap_iou: float) -> None:
        """Apply single-image predictions to the canvas (shared sync/async).

        Surfaces the open-vocabulary 'dropped unmatched class' count when the
        active predictor reports one (LocateAnything).
        """
        if not self._label_panel:
            return
        # Open-vocabulary backends (e.g. LocateAnything) drop detections whose
        # name didn't match any project class — surface that count if present.
        predictor = self._model_ctrl.predictor
        dropped = predictor.last_dropped if predictor else 0
        drop_suffix = f"，丢弃 {dropped} 个未匹配类别" if dropped else ""
        if annotations:
            self._label_panel.add_auto_annotations(
                annotations, overlap_iou=overlap_iou,
            )
            self.status_message.emit(
                f"自动标注: 检测到 {len(annotations)} 个目标{drop_suffix}"
            )
        else:
            self.status_message.emit(f"自动标注: 未检测到目标{drop_suffix}")

    # ── Batch ────────────────────────────────────────────────────

    def label_batch(self) -> None:
        """Batch auto-label: pre-registration → scope → worker → merge → store."""
        if not self._model_ctrl.predictor:
            QMessageBox.information(
                self._parent, "提示", "请先在模型面板中加载一个模型",
            )
            return
        if not self._label_panel or not self._project:
            return
        # No explicit flush: the first store-mediated read below
        # (get_unlabeled_image_paths) flushes pending edits.

        # Classification pre-flight: validate / register new model classes.
        if self._project.config.task_type == "classify":
            if not self._classify_batch_preflight():
                return

        all_images = self._project.list_images()
        unlabeled = self._label_panel.get_unlabeled_image_paths()
        target_images = self._scope_chooser(unlabeled, all_images)
        if target_images is None:
            return
        if not target_images:
            QMessageBox.information(self._parent, "提示", "没有需要处理的图片")
            return

        params = self._params_provider()
        self._batch_worker = BatchPredictWorker(
            predictor=self._model_ctrl.predictor,
            image_paths=target_images,
            conf=params.conf, iou=params.iou,
            project_classes=self._project.config.classes,
            class_match_mode=params.class_match_mode,
            task=self._project.config.task_type,
        )
        self._batch_skipped = 0
        self._batch_failed = 0
        self._batch_conflicts_dropped = 0
        self._batch_summary_emitted = False
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.image_done.connect(self._on_batch_image_done)
        self._batch_worker.finished_ok.connect(self._on_batch_finished_ok)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.finished.connect(self._on_batch_worker_finished)
        if self._project.config.task_type == "classify":
            self._label_panel.begin_bulk_auto_label_update()
        self._batch_worker.start()

        # Emitted after start() to mirror the historical choreography (the
        # progress dialog was created after the worker started; worker
        # signals are queued, so the dialog exists before the first one).
        self.batch_started.emit(len(target_images))
        self.status_message.emit(f"批量标注进行中: 0/{len(target_images)}")

    def _classify_batch_preflight(self) -> bool:
        """Validate classes / run the pre-registration dialog. False = abort."""
        cfg = self._project.config
        if not cfg.classes and not cfg.auto_register_classes:
            QMessageBox.information(
                self._parent, "提示",
                "项目当前没有类别，且未开启自动登记。\n"
                "请先在『类别管理』中添加类别，或开启自动登记后重试。",
            )
            return False
        if cfg.auto_register_classes:
            preview_items = self._project_ctrl.preview_model_classes(
                self._model_ctrl.predictor,
            )
            if preview_items:
                selected = self._class_register_confirm(preview_items)
                if selected is None:
                    return False
                new_count = 0
                for raw in selected:
                    result = self._project_ctrl.register_auto_class(raw, force=True)
                    if result.action == "registered":
                        new_count += 1
                if new_count:
                    # Refresh class bar / filter combo for the newly
                    # registered classes. Supersede-adjacent rebuild:
                    # classify writes through, nothing pending to flush.
                    self.classes_registered.emit()
        return True

    def cancel_batch(self) -> None:
        """Request cancellation of the running batch (dialog cancel button)."""
        if self._batch_worker is not None:
            self._batch_worker.cancel()

    def _on_batch_progress(self, current: int, total: int) -> None:
        self.status_message.emit(f"批量标注进行中: {current}/{total}")
        self.batch_progress.emit(current, total)

    def _on_batch_image_done(self, path_str: str, payload, img_size) -> None:
        """Apply one worker result on the main thread (disk writes live here)."""
        if not self._project:
            return
        img_path = Path(path_str)
        if self._project.config.task_type == "classify":
            if payload is None:
                self._batch_failed += 1
                return
            if not self._label_panel:
                return
            raw_name, conf = payload
            # Worker returns raw class names (filter_to_project=False); names
            # not in project.classes (either user did not approve in pre-dialog
            # or auto_register is OFF) are skipped here. We deliberately do not
            # call register_auto_class — that would mutate project state from a
            # non-GUI context if/when this slot is ever invoked off the main
            # thread.
            if raw_name not in self._project.config.classes:
                self._batch_skipped += 1
                return
            applied = self._label_panel.add_auto_class_prediction(
                img_path, raw_name, conf,
            )
            if not applied:
                self._batch_skipped += 1
            return
        annotations = payload
        label_path = self._project.label_path_for(img_path)
        ia = self._label_store.load_or_empty(
            label_path, img_path.name, image_size=img_size,
        )
        # Merge through the single core entry point: predictions overlapping
        # existing confirmed annotations are dropped (and counted) on the
        # batch surface. overlap_iou is re-read per image, matching the
        # historical behavior of reading the panel spinner in this slot.
        outcome = merge_predictions(
            ia.annotations, annotations, self._params_provider().overlap_iou,
        )
        self._batch_conflicts_dropped += len(outcome.conflict_pairs)
        for ann in outcome.accepted:
            ia.annotations.append(ann)
        self._label_store.save(ia, label_path)
        if self._label_panel:
            self._label_panel.set_image_status(img_path, ia.status)

    def _on_batch_finished_ok(self) -> None:
        notes = []
        if self._batch_skipped > 0:
            notes.append(f"跳过 {self._batch_skipped} 张已确认")
        if self._batch_failed > 0:
            notes.append(f"失败 {self._batch_failed} 张未识别")
        if self._batch_conflicts_dropped > 0:
            notes.append(
                f"丢弃 {self._batch_conflicts_dropped} 个与已确认标注重叠的预测"
            )
        if notes:
            summary = "批量自动标注完成（" + "，".join(notes) + "）"
        else:
            summary = "批量自动标注完成"
        self._batch_summary_emitted = True
        self.batch_finished.emit(summary)
        # Supersede for the focused image: the batch wrote its record behind
        # the view; reload replaces stale memory (the view's clean-skip keeps
        # the implicit flush from clobbering the newer record first).
        if self._label_panel and self._label_panel.get_current_image_path():
            self._label_panel.reload_current()

    def _on_batch_error(self, msg: str) -> None:
        self._batch_summary_emitted = True
        self.batch_finished.emit("批量标注失败")
        QMessageBox.warning(self._parent, "批量标注失败", msg)

    def _on_batch_worker_finished(self) -> None:
        """Worker ``finished`` (always fires): end bulk mode + cancel cleanup."""
        if self._label_panel:
            self._label_panel.end_bulk_auto_label_update()
        self._batch_worker = None
        # With BatchPredictWorker, finishing without finished_ok/error means
        # the run was cancelled — surface the terminal summary so MainWindow
        # closes the progress dialog.
        if not self._batch_summary_emitted:
            self._batch_summary_emitted = True
            self.batch_finished.emit("批量标注已取消")

    # ── Shutdown ─────────────────────────────────────────────────

    def shutdown(self, timeout_ms: int = 30000) -> None:
        """Wait for any in-flight single-image inference (slow backend) so the
        worker isn't using the predictor while the caller releases it.

        Batch workers keep the historical close semantics (not waited on).
        """
        if self._single_worker is not None and self._single_worker.isRunning():
            self._single_worker.wait(timeout_ms)

    # ── Default interactive callables ────────────────────────────

    def _default_scope_chooser(
        self, unlabeled: list[Path], all_images: list[Path],
    ) -> list[Path] | None:
        items = ["仅未标注图片", "全部图片"]
        choice, ok = QInputDialog.getItem(
            self._parent, "批量自动标注",
            f"选择范围 (未标注: {len(unlabeled)} / 全部: {len(all_images)})",
            items, 0, False,
        )
        if not ok:
            return None
        return unlabeled if choice == items[0] else all_images

    def _default_class_register_confirm(self, preview_items) -> list[str] | None:
        from src.ui.dialogs import ClassRegisterDialog  # lazy: keep src.ui out of import time

        dlg = ClassRegisterDialog(preview_items, parent=self._parent)
        if not dlg.exec_():
            return None
        return dlg.get_selected()
