"""Regression tests for the LabelStore flush invariant in the GUI layer.

Pins the three data-loss scenarios the store was built to close (PRD
07-09): pending canvas edits vs. the tag-manager cascade, vs. the
class-manager rebuild (LabelPanel.set_project default flush), and the
Supersede rule (set_project(discard_pending=True) must NOT flush stale
memory over records that were intentionally advanced on disk). Also pins
the _save_current clean-skip contract (flush must be cheap when clean,
but full-content dirty so geometry-only edits still hit disk).
"""
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QColor

from src.core import label_io
from src.core.annotation import Annotation, ImageAnnotation
from src.core.label_store import LabelStore


def _make_project(tmp_path, n_images=2, task_type="detect"):
    from src.core.project import ProjectManager

    pm = ProjectManager.create(
        tmp_path / "proj", "test", classes=["cat", "dog"], task_type=task_type,
    )
    img_dir = pm.project_dir / pm.config.image_dir
    for i in range(n_images):
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.blue))
        img.save(str(img_dir / f"img{i}.png"), "PNG")
    return pm


def _pending_annotation(ann_id="pending-0"):
    return Annotation(
        id=ann_id, class_name="cat", class_id=0,
        bbox=(0.5, 0.5, 0.2, 0.2), confirmed=False, source="auto",
    )


def _make_panel(tmp_path, pm, store, tag_ctrl=None):
    from src.ui.label_panel import LabelPanel

    panel = LabelPanel(
        config_path=tmp_path / "config.json",
        tag_controller=tag_ctrl,
        label_store=store,
    )
    panel.set_project(pm)
    return panel


class TestSharedStoreWiring:
    def test_shared_store_wiring_identity(self, qapp, tmp_path):
        """Every consumer constructor falls back to `label_store or
        LabelStore()`, so a production construction site that forgets to
        pass MainWindow's shared instance silently gets a bare store with
        NO flush callback — the flush-before-read invariant dissolves
        without any test failing. This pins the identity of the shared
        store across every wired consumer, plus the installed callback."""
        from src.app import MainWindow

        pm = _make_project(tmp_path)
        win = MainWindow(config_path=tmp_path / "config.json")
        try:
            win.open_project(pm)
            qapp.processEvents()

            store = win._label_store
            assert win._label_panel._label_store is store
            assert win._tag_ctrl._store is store
            assert win._project_ctrl._store is store
            assert win._train_ctrl._store is store
            assert win._autolabel_ctrl._label_store is store
            # Active view (detect) shares the same store.
            assert win._label_panel._view._store is store
            # Flush callback installed and pointing at the panel.
            assert win._label_store._flush_cb == win._label_panel.save_and_cleanup
        finally:
            win.close()

    def test_shared_store_wiring_identity_classify_preview(self, qapp, tmp_path):
        """Classify view forwards the shared store into its PreviewPane."""
        from src.app import MainWindow

        pm = _make_project(tmp_path, task_type="classify")
        win = MainWindow(config_path=tmp_path / "config.json")
        try:
            win.open_project(pm)
            qapp.processEvents()

            store = win._label_store
            view = win._label_panel._view
            assert view._store is store
            assert view._preview._store is store
        finally:
            win.close()


class TestTagCascadeFlush:
    def test_rename_tag_cascade_sees_pending_canvas_edits(self, qapp, tmp_path):
        """Gap ②: the rename cascade reads through the store → the focused
        image's in-memory bbox is flushed to disk BEFORE the cascade rewrites
        the record — no window where disk lacks the user's edit."""
        from src.controllers.tags import TagController

        pm = _make_project(tmp_path)
        pm.config.tags = ["old"]
        pm.save()
        imgs = pm.list_images()
        # img0 carries the tag on disk; the bbox exists only in memory below.
        ia = ImageAnnotation(image_path=imgs[0].name, image_size=(100, 80), tags=["old"])
        label_io.save_annotation(ia, pm.label_path_for(imgs[0]))

        store = LabelStore()
        ctrl = TagController(label_store=store)
        ctrl.set_project(pm)
        panel = _make_panel(tmp_path, pm, store, tag_ctrl=ctrl)
        try:
            view = panel._view
            qapp.processEvents()
            assert view._current_image_path == imgs[0]

            # Pending edit: in-memory only, nothing written yet.
            view.add_auto_annotations([_pending_annotation()])
            on_disk = label_io.load_annotation(pm.label_path_for(imgs[0]))
            assert on_disk is not None and len(on_disk.annotations) == 0

            ctrl.rename_tag("old", "new")

            after = label_io.load_annotation(pm.label_path_for(imgs[0]))
            assert after is not None
            assert len(after.annotations) == 1, \
                "cascade dropped the pending canvas edit (flush-before-read broken)"
            assert after.tags == ["new"]
        finally:
            panel.deleteLater()


class TestSetProjectFlush:
    def test_default_set_project_flushes_old_view_before_teardown(
        self, qapp, tmp_path,
    ):
        """Gap ①: the class-manager flow rebuilds the view via set_project;
        the default must commit the old view's pending edit first."""
        pm = _make_project(tmp_path)
        imgs = pm.list_images()
        store = LabelStore()
        panel = _make_panel(tmp_path, pm, store)
        try:
            view = panel._view
            qapp.processEvents()
            assert view._current_image_path == imgs[0]
            view.add_auto_annotations([_pending_annotation()])
            assert label_io.load_annotation(pm.label_path_for(imgs[0])) is None

            panel.set_project(pm)  # default: flush before teardown

            after = label_io.load_annotation(pm.label_path_for(imgs[0]))
            assert after is not None and len(after.annotations) == 1, \
                "set_project teardown dropped the pending edit"
        finally:
            panel.deleteLater()

    def test_discard_pending_does_not_overwrite_superseded_records(
        self, qapp, tmp_path,
    ):
        """Supersede: after an external bulk write (import / batch auto-label)
        the reload must NOT flush stale memory over the newer records."""
        pm = _make_project(tmp_path)
        imgs = pm.list_images()
        store = LabelStore()
        panel = _make_panel(tmp_path, pm, store)
        try:
            view = panel._view
            qapp.processEvents()
            assert view._current_image_path == imgs[0]
            # Stale pending edit in memory…
            view.add_auto_annotations([_pending_annotation("stale-mem")])
            # …while disk is intentionally advanced behind the view's back.
            newer = ImageAnnotation(image_path=imgs[0].name, image_size=(100, 80))
            newer.annotations.append(_pending_annotation("external-1"))
            newer.annotations.append(_pending_annotation("external-2"))
            label_io.save_annotation(newer, pm.label_path_for(imgs[0]))

            panel.set_project(pm, discard_pending=True)

            after = label_io.load_annotation(pm.label_path_for(imgs[0]))
            assert after is not None
            assert sorted(a.id for a in after.annotations) == [
                "external-1", "external-2",
            ], "discard_pending flushed stale memory over the superseded records"
        finally:
            panel.deleteLater()


class TestSaveCurrentCleanSkip:
    @pytest.fixture
    def counted_saves(self, monkeypatch):
        """Count physical writes issued through label_io.save_annotation."""
        calls = []
        real = label_io.save_annotation

        def counting(ia, path):
            calls.append(Path(path))
            return real(ia, path)

        monkeypatch.setattr(label_io, "save_annotation", counting)
        return calls

    def test_clean_save_skips_disk_write(self, qapp, tmp_path, counted_saves):
        pm = _make_project(tmp_path)
        store = LabelStore()
        panel = _make_panel(tmp_path, pm, store)
        try:
            view = panel._view
            qapp.processEvents()
            counted_saves.clear()

            view._save_current()  # nothing changed since load
            assert counted_saves == [], "clean _save_current still wrote to disk"

            # Dirty → exactly one write; repeat clean call → still one.
            view._canvas.add_annotations([_pending_annotation()])
            view._save_current()
            assert len(counted_saves) == 1
            view._save_current()
            assert len(counted_saves) == 1
        finally:
            panel.deleteLater()

    def test_geometry_only_change_is_not_treated_as_clean(
        self, qapp, tmp_path, counted_saves,
    ):
        """The dirty check must be full-content: moving a bbox changes no
        class/confirmed stats, but it MUST still reach disk."""
        pm = _make_project(tmp_path)
        imgs = pm.list_images()
        ia = ImageAnnotation(image_path=imgs[0].name, image_size=(100, 80))
        ia.annotations.append(Annotation(
            id="a0", class_name="cat", class_id=0,
            bbox=(0.5, 0.5, 0.2, 0.2), confirmed=True,
        ))
        label_io.save_annotation(ia, pm.label_path_for(imgs[0]))

        store = LabelStore()
        panel = _make_panel(tmp_path, pm, store)
        try:
            view = panel._view
            qapp.processEvents()
            assert view._current_image_path == imgs[0]
            counted_saves.clear()

            view._canvas.annotations[0].bbox = (0.3, 0.3, 0.2, 0.2)
            view._save_current()

            assert len(counted_saves) == 1, "bbox move was skipped as 'clean'"
            after = label_io.load_annotation(pm.label_path_for(imgs[0]))
            assert after.annotations[0].bbox == (0.3, 0.3, 0.2, 0.2)
        finally:
            panel.deleteLater()

    def test_scan_loop_flushes_pending_edit_exactly_once(
        self, qapp, tmp_path, counted_saves,
    ):
        """N store reads in a scan loop → the pending edit hits disk on the
        first read and the remaining flushes are clean no-ops."""
        pm = _make_project(tmp_path, n_images=3)
        imgs = pm.list_images()
        store = LabelStore()
        panel = _make_panel(tmp_path, pm, store)
        try:
            view = panel._view
            qapp.processEvents()
            view.add_auto_annotations([_pending_annotation()])
            counted_saves.clear()

            # Store-mediated scan (same shape as _collect_unconfirmed).
            for p in imgs:
                store.load(pm.label_path_for(p))

            assert counted_saves == [pm.label_path_for(imgs[0])]
            after = label_io.load_annotation(pm.label_path_for(imgs[0]))
            assert after is not None and len(after.annotations) == 1
        finally:
            panel.deleteLater()
