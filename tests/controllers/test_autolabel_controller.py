"""Tests for AutoLabelController — fake predictor / fake panel, no real model.

Pins the deep-module contract from the 07-10 refactor:
- sync vs async single-image dispatch by injected predicate (YOLO stays sync);
- re-entrancy guard while a single worker is in flight;
- batch detect per-image persistence through the shared LabelStore with
  conflict predictions dropped + counted, file-list status via the panel's
  public face;
- batch classify per-image skip/fail counting WITHOUT register_auto_class;
- terminal batch summaries incl. the cancelled path;
- shutdown waits for the in-flight single worker.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from PyQt5.QtCore import QObject, pyqtSignal

from src.controllers.autolabel import AutoLabelController
from src.controllers.project import RegistrationResult
from src.core.annotation import Annotation, ImageAnnotation
from src.core.autolabel import InferenceParams
from src.core.label_io import load_annotation, save_annotation
from src.core.label_store import LabelStore
from src.core.project import ProjectManager


# ── Fakes ──────────────────────────────────────────────────────


class _FakePanel:
    """Records every public-face call the controller is allowed to make."""

    def __init__(self, current_image=None, unlabeled=None):
        self.current_image = current_image
        self.unlabeled = unlabeled or []
        self.calls: list[tuple] = []

    def get_current_image_path(self):
        return self.current_image

    def get_unlabeled_image_paths(self):
        return list(self.unlabeled)

    def add_auto_annotations(self, anns, overlap_iou=0.5):
        self.calls.append(("add_auto_annotations", list(anns), overlap_iou))

    def add_auto_class_prediction(self, path, class_name, confidence):
        self.calls.append(("add_auto_class_prediction", path, class_name, confidence))
        return True

    def begin_bulk_auto_label_update(self):
        self.calls.append(("begin_bulk",))

    def end_bulk_auto_label_update(self):
        self.calls.append(("end_bulk",))

    def set_auto_label_busy(self, busy):
        self.calls.append(("set_auto_label_busy", busy))

    def set_image_status(self, path, status):
        self.calls.append(("set_image_status", path, status))

    def reload_current(self):
        self.calls.append(("reload_current",))

    def named(self, name):
        return [c for c in self.calls if c[0] == name]


class _FakePredictor:
    last_dropped = 0


class _FakeModelCtrl:
    def __init__(self, annotations=None, classify_result=None, worker=None):
        self.predictor = _FakePredictor()
        self._annotations = annotations or []
        self._classify_result = classify_result
        self._worker = worker
        self.predict_single_calls: list = []
        self.classify_calls: list = []
        self.worker_calls: list = []

    def predict_single(self, img_path, classes, **kwargs):
        self.predict_single_calls.append((img_path, list(classes), kwargs))
        return list(self._annotations)

    def predict_single_classify(self, img_path, classes):
        self.classify_calls.append((img_path, list(classes)))
        return self._classify_result

    def create_single_predict_worker(self, img_path, classes, **kwargs):
        self.worker_calls.append((img_path, list(classes), kwargs))
        return self._worker


class _FakeProjectCtrl:
    def __init__(self, reg_result=None, preview_items=None):
        self._reg_result = reg_result
        self._preview_items = preview_items or []
        self.register_calls: list = []

    def register_auto_class(self, raw, force=False):
        self.register_calls.append((raw, force))
        return self._reg_result

    def preview_model_classes(self, predictor):
        return list(self._preview_items)


class _SyncWorker(QObject):
    """SinglePredictWorker stand-in that runs synchronously on start()."""

    done = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, annotations=None, err=None):
        super().__init__()
        self._annotations = annotations or []
        self._err = err

    def isRunning(self):
        return False

    def start(self):
        if self._err is not None:
            self.error.emit(self._err)
        else:
            self.done.emit(self._annotations)
        self.finished.emit()


class _RunningWorker:
    """A worker that reports itself as permanently in flight."""

    def __init__(self):
        self.wait_calls: list = []

    def isRunning(self):
        return True

    def wait(self, timeout):
        self.wait_calls.append(timeout)


# ── Helpers ────────────────────────────────────────────────────


def _pred(cls="cat", bbox=(0.5, 0.5, 0.2, 0.2)):
    return Annotation(
        class_name=cls, class_id=0, bbox=bbox,
        confidence=0.9, confirmed=False, source="auto",
    )


def _make_project(tmp_path, task_type="detect", classes=("cat", "dog")):
    return ProjectManager.create(
        tmp_path / "proj", "p", classes=list(classes), task_type=task_type,
    )


def _make_ctrl(
    project,
    panel,
    model_ctrl=None,
    project_ctrl=None,
    slow=False,
    store=None,
    params=None,
    scope_chooser=None,
    class_register_confirm=None,
):
    ctrl = AutoLabelController(
        model_ctrl or _FakeModelCtrl(),
        project_ctrl or _FakeProjectCtrl(),
        store or LabelStore(),
        slow_backend_active=(lambda: slow) if isinstance(slow, bool) else slow,
        params_provider=lambda: params or InferenceParams(),
        parent_widget=None,
        scope_chooser=scope_chooser,
        class_register_confirm=class_register_confirm,
    )
    ctrl.set_context(project, panel)
    return ctrl


def _record(signal, into):
    signal.connect(lambda *args: into.append(args))


# ── Single image: dispatch ─────────────────────────────────────


class TestSingleDispatch:
    def test_sync_path_when_predicate_false(self, qapp, tmp_path):
        """YOLO contract: predicate False → inline predict_single, applied
        immediately, no worker built, no busy toggling."""
        pm = _make_project(tmp_path)
        ann = _pred()
        model = _FakeModelCtrl(annotations=[ann])
        panel = _FakePanel(current_image=Path("/x/img0.png"))
        ctrl = _make_ctrl(pm, panel, model_ctrl=model, slow=False,
                          params=InferenceParams(conf=0.7, iou=0.4, overlap_iou=0.6))
        statuses: list = []
        busy: list = []
        _record(ctrl.status_message, statuses)
        _record(ctrl.busy_changed, busy)

        ctrl.label_current()

        assert len(model.predict_single_calls) == 1
        path, classes, kwargs = model.predict_single_calls[0]
        assert classes == ["cat", "dog"]
        assert kwargs == {"conf": 0.7, "iou": 0.4, "class_match_mode": "class_id"}
        assert model.worker_calls == []
        assert panel.named("add_auto_annotations") == [
            ("add_auto_annotations", [ann], 0.6)
        ]
        assert statuses == [("自动标注: 检测到 1 个目标",)]
        assert busy == []
        assert ctrl._single_worker is None

    def test_async_path_when_predicate_true(self, qapp, tmp_path):
        pm = _make_project(tmp_path)
        ann = _pred()
        worker = _SyncWorker(annotations=[ann])
        model = _FakeModelCtrl(worker=worker)
        panel = _FakePanel(current_image=Path("/x/img0.png"))
        ctrl = _make_ctrl(pm, panel, model_ctrl=model, slow=True)
        busy: list = []
        _record(ctrl.busy_changed, busy)

        ctrl.label_current()

        # busy_changed emitted as a pair, panel driven through public face.
        assert busy == [(True,), (False,)]
        assert panel.named("set_auto_label_busy") == [
            ("set_auto_label_busy", True), ("set_auto_label_busy", False),
        ]
        assert model.predict_single_calls == []
        assert len(model.worker_calls) == 1
        assert panel.named("add_auto_annotations") == [
            ("add_auto_annotations", [ann], 0.5)
        ]
        assert ctrl._single_worker is None

    def test_async_error_surfaces_message_and_clears_busy(
        self, qapp, tmp_path, monkeypatch,
    ):
        from PyQt5.QtWidgets import QMessageBox

        pm = _make_project(tmp_path)
        model = _FakeModelCtrl(worker=_SyncWorker(err="CUDA OOM"))
        panel = _FakePanel(current_image=Path("/x/img0.png"))
        ctrl = _make_ctrl(pm, panel, model_ctrl=model, slow=True)
        shown: list = []
        monkeypatch.setattr(
            QMessageBox, "warning", lambda *a, **k: shown.append(a[-1]),
        )
        statuses: list = []
        _record(ctrl.status_message, statuses)

        ctrl.label_current()

        assert any("OOM" in s for s in shown)
        assert ("自动标注失败",) in statuses
        assert panel.named("set_auto_label_busy")[-1] == ("set_auto_label_busy", False)

    def test_reentrancy_guard_blocks_second_dispatch(self, qapp, tmp_path):
        pm = _make_project(tmp_path)
        model = _FakeModelCtrl(worker=_SyncWorker())
        panel = _FakePanel(current_image=Path("/x/img0.png"))
        ctrl = _make_ctrl(pm, panel, model_ctrl=model, slow=True)
        in_flight = _RunningWorker()
        ctrl._single_worker = in_flight

        ctrl.label_current()

        assert model.worker_calls == []
        assert model.predict_single_calls == []
        assert ctrl._single_worker is in_flight

    def test_worker_build_refusal_keeps_idle(self, qapp, tmp_path):
        """create_single_predict_worker → None (no predictor) must not flip busy."""
        pm = _make_project(tmp_path)
        model = _FakeModelCtrl(worker=None)
        panel = _FakePanel(current_image=Path("/x/img0.png"))
        ctrl = _make_ctrl(pm, panel, model_ctrl=model, slow=True)
        busy: list = []
        _record(ctrl.busy_changed, busy)

        ctrl.label_current()

        assert busy == []
        assert ctrl._single_worker is None


# ── Single image: classify decision integration ───────────────


class TestSingleClassify:
    def test_registered_rebuilds_then_applies(self, qapp, tmp_path):
        pm = _make_project(tmp_path, task_type="classify")
        img = Path("/x/img0.png")
        model = _FakeModelCtrl(classify_result=("bird", 0.87))
        proj_ctrl = _FakeProjectCtrl(
            reg_result=RegistrationResult(
                action="registered", applied_name="bird", reason="",
            ),
        )
        panel = _FakePanel(current_image=img)
        ctrl = _make_ctrl(pm, panel, model_ctrl=model, project_ctrl=proj_ctrl)
        order: list = []
        ctrl.classes_registered.connect(lambda: order.append("rebuild"))
        statuses: list = []
        _record(ctrl.status_message, statuses)

        ctrl.label_current()

        # Rebuild signal fires BEFORE the prediction is applied (the class
        # widgets must know the new class when the visual updates land).
        applied_idx = panel.calls.index(
            ("add_auto_class_prediction", img, "bird", 0.87)
        )
        assert order == ["rebuild"]
        assert applied_idx >= 0
        assert statuses[-1] == ("自动标注: bird (0.87) (新增类别 'bird')",)

    def test_unrecognized_result_emits_status_only(self, qapp, tmp_path):
        pm = _make_project(tmp_path, task_type="classify")
        model = _FakeModelCtrl(classify_result=None)
        panel = _FakePanel(current_image=Path("/x/img0.png"))
        ctrl = _make_ctrl(pm, panel, model_ctrl=model)
        statuses: list = []
        _record(ctrl.status_message, statuses)

        ctrl.label_current()

        assert statuses == [("自动标注: 未识别",)]
        assert panel.named("add_auto_class_prediction") == []

    def test_view_veto_shows_skip_text(self, qapp, tmp_path):
        pm = _make_project(tmp_path, task_type="classify")
        img = Path("/x/img0.png")
        model = _FakeModelCtrl(classify_result=("cat", 0.9))
        proj_ctrl = _FakeProjectCtrl(
            reg_result=RegistrationResult(
                action="existing", applied_name="cat", reason="",
            ),
        )
        panel = _FakePanel(current_image=img)
        panel.add_auto_class_prediction = lambda *a, **k: False  # confirmed tag
        ctrl = _make_ctrl(pm, panel, model_ctrl=model, project_ctrl=proj_ctrl)
        statuses: list = []
        _record(ctrl.status_message, statuses)

        ctrl.label_current()

        assert statuses == [("自动标注: 已存在确认标签，跳过",)]


# ── Batch: per-image application ───────────────────────────────


class TestBatchImageDone:
    def test_detect_persists_through_store_and_updates_status(self, qapp, tmp_path):
        pm = _make_project(tmp_path)
        img = pm.project_dir / "images" / "img0.png"
        store = LabelStore()
        panel = _FakePanel()
        ctrl = _make_ctrl(pm, panel, store=store)

        pred = _pred()
        ctrl._on_batch_image_done(str(img), [pred], (40, 40))

        ia = load_annotation(pm.label_path_for(img))
        assert ia is not None and len(ia.annotations) == 1
        assert ia.annotations[0].class_name == "cat"
        assert ia.image_size == (40, 40)
        # File-list status via the panel's public face (no private reach-in).
        assert panel.named("set_image_status") == [
            ("set_image_status", img, "pending")
        ]

    def test_detect_drops_and_counts_conflicts(self, qapp, tmp_path):
        pm = _make_project(tmp_path)
        img = pm.project_dir / "images" / "img0.png"
        existing = ImageAnnotation(image_path="img0.png", image_size=(40, 40))
        existing.annotations.append(Annotation(
            class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2),
            confirmed=True, source="manual",
        ))
        save_annotation(existing, pm.label_path_for(img))
        panel = _FakePanel()
        ctrl = _make_ctrl(pm, panel)

        overlapping = _pred(bbox=(0.5, 0.5, 0.2, 0.2))   # IoU 1.0 vs confirmed
        clean = _pred(bbox=(0.1, 0.1, 0.05, 0.05))
        ctrl._on_batch_image_done(str(img), [overlapping, clean], (40, 40))

        ia = load_annotation(pm.label_path_for(img))
        assert len(ia.annotations) == 2  # existing + clean only
        assert ctrl._batch_conflicts_dropped == 1

    def test_classify_unknown_class_skipped_without_registration(
        self, qapp, tmp_path,
    ):
        pm = _make_project(tmp_path, task_type="classify", classes=("cat",))
        img = pm.project_dir / "images" / "img0.png"
        proj_ctrl = _FakeProjectCtrl()
        panel = _FakePanel()
        ctrl = _make_ctrl(pm, panel, project_ctrl=proj_ctrl)

        ctrl._on_batch_image_done(str(img), ("bird", 0.9), (0, 0))

        assert ctrl._batch_skipped == 1
        # Deliberate design: per-image slot must never register classes.
        assert proj_ctrl.register_calls == []
        assert panel.named("add_auto_class_prediction") == []

    def test_classify_none_payload_counts_failed(self, qapp, tmp_path):
        pm = _make_project(tmp_path, task_type="classify", classes=("cat",))
        img = pm.project_dir / "images" / "img0.png"
        ctrl = _make_ctrl(pm, _FakePanel())

        ctrl._on_batch_image_done(str(img), None, (0, 0))
        assert ctrl._batch_failed == 1

    def test_classify_known_class_applied_and_veto_counts_skipped(
        self, qapp, tmp_path,
    ):
        pm = _make_project(tmp_path, task_type="classify", classes=("cat",))
        img = pm.project_dir / "images" / "img0.png"
        panel = _FakePanel()
        ctrl = _make_ctrl(pm, panel)

        ctrl._on_batch_image_done(str(img), ("cat", 0.8), (0, 0))
        assert panel.named("add_auto_class_prediction") == [
            ("add_auto_class_prediction", img, "cat", 0.8)
        ]
        assert ctrl._batch_skipped == 0

        panel.add_auto_class_prediction = lambda *a, **k: False
        ctrl._on_batch_image_done(str(img), ("cat", 0.8), (0, 0))
        assert ctrl._batch_skipped == 1


# ── Batch: lifecycle / summaries ───────────────────────────────


class TestBatchLifecycle:
    def test_finished_ok_summary_and_reload(self, qapp, tmp_path):
        pm = _make_project(tmp_path)
        panel = _FakePanel(current_image=Path("/x/img0.png"))
        ctrl = _make_ctrl(pm, panel)
        done: list = []
        _record(ctrl.batch_finished, done)

        ctrl._batch_skipped = 1
        ctrl._batch_failed = 2
        ctrl._batch_conflicts_dropped = 3
        ctrl._on_batch_finished_ok()

        assert done == [(
            "批量自动标注完成（跳过 1 张已确认，失败 2 张未识别，"
            "丢弃 3 个与已确认标注重叠的预测）",
        )]
        assert panel.named("reload_current") == [("reload_current",)]

    def test_finished_ok_plain_summary_without_notes(self, qapp, tmp_path):
        pm = _make_project(tmp_path)
        panel = _FakePanel(current_image=None)  # no focus → no reload
        ctrl = _make_ctrl(pm, panel)
        done: list = []
        _record(ctrl.batch_finished, done)

        ctrl._on_batch_finished_ok()

        assert done == [("批量自动标注完成",)]
        assert panel.named("reload_current") == []

    def test_cancel_path_emits_cancelled_summary_once(self, qapp, tmp_path):
        pm = _make_project(tmp_path)
        panel = _FakePanel()
        ctrl = _make_ctrl(pm, panel)
        fake_worker = MagicMock()
        ctrl._batch_worker = fake_worker
        done: list = []
        _record(ctrl.batch_finished, done)

        ctrl.cancel_batch()
        fake_worker.cancel.assert_called_once()

        # Worker finished without finished_ok/error → cancelled semantics.
        ctrl._on_batch_worker_finished()
        assert done == [("批量标注已取消",)]
        assert panel.named("end_bulk") == [("end_bulk",)]
        assert ctrl._batch_worker is None

    def test_worker_finished_after_ok_does_not_double_emit(self, qapp, tmp_path):
        pm = _make_project(tmp_path)
        panel = _FakePanel()
        ctrl = _make_ctrl(pm, panel)
        done: list = []
        _record(ctrl.batch_finished, done)

        ctrl._on_batch_finished_ok()
        ctrl._on_batch_worker_finished()

        assert len(done) == 1
        assert panel.named("end_bulk") == [("end_bulk",)]

    def test_error_path_emits_failure_and_warns(self, qapp, tmp_path, monkeypatch):
        from PyQt5.QtWidgets import QMessageBox

        pm = _make_project(tmp_path)
        ctrl = _make_ctrl(pm, _FakePanel())
        shown: list = []
        monkeypatch.setattr(
            QMessageBox, "warning", lambda *a, **k: shown.append(a[-1]),
        )
        done: list = []
        _record(ctrl.batch_finished, done)

        ctrl._on_batch_error("boom")

        assert done == [("批量标注失败",)]
        assert shown == ["boom"]

    def test_label_batch_classify_choreography(self, qapp, tmp_path, monkeypatch):
        """begin_bulk (classify only) precedes worker.start(); batch_started
        carries the total; worker gets the classify task."""
        pm = _make_project(tmp_path, task_type="classify", classes=("cat",))
        pm.config.auto_register_classes = False
        pm.save()
        imgs = [pm.project_dir / "images" / f"i{i}.png" for i in range(2)]
        panel = _FakePanel(unlabeled=imgs)
        model = _FakeModelCtrl()
        events: list = []

        class _FakeBatchWorker(QObject):
            progress = pyqtSignal(int, int)
            image_done = pyqtSignal(str, object, object)
            finished_ok = pyqtSignal()
            error = pyqtSignal(str)
            finished = pyqtSignal()

            def __init__(self, **kwargs):
                super().__init__()
                events.append(("constructed", kwargs["task"], list(kwargs["image_paths"])))

            def start(self):
                events.append(("start",))

        monkeypatch.setattr(
            "src.controllers.autolabel.BatchPredictWorker", _FakeBatchWorker,
        )
        ctrl = _make_ctrl(
            pm, panel, model_ctrl=model,
            scope_chooser=lambda unlabeled, all_images: unlabeled,
        )
        panel.begin_bulk_auto_label_update = (
            lambda: events.append(("begin_bulk",))
        )
        started: list = []
        _record(ctrl.batch_started, started)
        statuses: list = []
        _record(ctrl.status_message, statuses)

        ctrl.label_batch()

        assert events == [
            ("constructed", "classify", imgs),
            ("begin_bulk",),
            ("start",),
        ]
        assert started == [(2,)]
        assert statuses[-1] == ("批量标注进行中: 0/2",)

    def test_label_batch_detect_skips_begin_bulk(self, qapp, tmp_path, monkeypatch):
        pm = _make_project(tmp_path)
        imgs = [pm.project_dir / "images" / "i0.png"]
        panel = _FakePanel(unlabeled=imgs)
        events: list = []

        class _FakeBatchWorker(QObject):
            progress = pyqtSignal(int, int)
            image_done = pyqtSignal(str, object, object)
            finished_ok = pyqtSignal()
            error = pyqtSignal(str)
            finished = pyqtSignal()

            def __init__(self, **kwargs):
                super().__init__()

            def start(self):
                events.append(("start",))

        monkeypatch.setattr(
            "src.controllers.autolabel.BatchPredictWorker", _FakeBatchWorker,
        )
        ctrl = _make_ctrl(
            pm, panel, scope_chooser=lambda unlabeled, all_images: unlabeled,
        )
        panel.begin_bulk_auto_label_update = (
            lambda: events.append(("begin_bulk",))
        )

        ctrl.label_batch()

        assert events == [("start",)]

    def test_label_batch_without_predictor_warns_and_aborts(
        self, qapp, tmp_path, monkeypatch,
    ):
        from PyQt5.QtWidgets import QMessageBox

        pm = _make_project(tmp_path)
        model = _FakeModelCtrl()
        model.predictor = None
        scope_calls: list = []
        ctrl = _make_ctrl(
            pm, _FakePanel(), model_ctrl=model,
            scope_chooser=lambda *a: scope_calls.append(a) or None,
        )
        shown: list = []
        monkeypatch.setattr(
            QMessageBox, "information", lambda *a, **k: shown.append(a[-1]),
        )

        ctrl.label_batch()

        assert shown == ["请先在模型面板中加载一个模型"]
        assert scope_calls == []

    def test_scope_chooser_cancel_aborts(self, qapp, tmp_path):
        pm = _make_project(tmp_path)
        ctrl = _make_ctrl(
            pm, _FakePanel(), scope_chooser=lambda *a: None,
        )
        started: list = []
        _record(ctrl.batch_started, started)

        ctrl.label_batch()

        assert started == []
        assert ctrl._batch_worker is None

    def test_classify_preflight_registers_selected_and_emits_rebuild(
        self, qapp, tmp_path, monkeypatch,
    ):
        pm = _make_project(tmp_path, task_type="classify", classes=("cat",))
        assert pm.config.auto_register_classes is True
        proj_ctrl = _FakeProjectCtrl(
            reg_result=RegistrationResult(
                action="registered", applied_name="bird", reason="",
            ),
            preview_items=[object()],  # non-empty → confirm callable runs
        )
        rebuilds: list = []
        ctrl = _make_ctrl(
            pm, _FakePanel(unlabeled=[]),
            project_ctrl=proj_ctrl,
            scope_chooser=lambda *a: None,  # abort right after preflight
            class_register_confirm=lambda items: ["bird"],
        )
        ctrl.classes_registered.connect(lambda: rebuilds.append(True))

        ctrl.label_batch()

        assert proj_ctrl.register_calls == [("bird", True)]
        assert rebuilds == [True]

    def test_classify_preflight_confirm_cancel_aborts(self, qapp, tmp_path):
        pm = _make_project(tmp_path, task_type="classify", classes=("cat",))
        proj_ctrl = _FakeProjectCtrl(preview_items=[object()])
        scope_calls: list = []
        ctrl = _make_ctrl(
            pm, _FakePanel(),
            project_ctrl=proj_ctrl,
            scope_chooser=lambda *a: scope_calls.append(a) or None,
            class_register_confirm=lambda items: None,  # user cancelled
        )

        ctrl.label_batch()

        assert proj_ctrl.register_calls == []
        assert scope_calls == []


# ── Shutdown ───────────────────────────────────────────────────


class TestShutdown:
    def test_shutdown_waits_for_in_flight_single_worker(self, qapp, tmp_path):
        pm = _make_project(tmp_path)
        ctrl = _make_ctrl(pm, _FakePanel())
        worker = _RunningWorker()
        ctrl._single_worker = worker

        ctrl.shutdown(30000)

        assert worker.wait_calls == [30000]

    def test_shutdown_noop_without_worker(self, qapp, tmp_path):
        pm = _make_project(tmp_path)
        ctrl = _make_ctrl(pm, _FakePanel())
        ctrl.shutdown(30000)  # must not raise
