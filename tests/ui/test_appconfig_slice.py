"""AppConfig single-authority + ClassifyViewState slice regressions (07-11).

Pins the double-authority last-writer-wins bug: ClassifyView used to do its
own disk round-trip (AppConfig.load → mutate → save) against a hardcoded
~/.autolabel path, so any whole-config save from the MainWindow/controller
in-memory copy silently reverted classify settings. Now the view mutates the
shared ``AppConfig.classify`` slice in place and persists via an injected
saver — one in-memory authority, zero AppConfig.load in src/ui.
"""
from collections import OrderedDict
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage

from src.core.config import AppConfig, ClassifyViewState


def _make_classify_project(tmp_path, n_imgs=2, classes=("cat", "dog")):
    from src.core.project import ProjectManager

    pm = ProjectManager.create(
        tmp_path / "proj", "p", classes=list(classes), task_type="classify",
    )
    img_dir = pm.project_dir / pm.config.image_dir
    for i in range(n_imgs):
        img = QImage(20, 20, QImage.Format_RGB32)
        img.fill(QColor(Qt.blue))
        img.save(str(img_dir / f"img{i}.png"), "PNG")
    return pm


def _teardown(qapp, panel):
    import gc

    panel._view.cleanup()
    panel.deleteLater()
    qapp.processEvents()
    del panel
    gc.collect()
    qapp.processEvents()


class TestClobberRegression:
    def test_controller_whole_save_keeps_view_density(self, qapp, tmp_path):
        """Bug ① regression: view changes density → a controller-path whole
        config save (recent-projects update, closeEvent) must NOT revert it.

        Before the fix the view wrote 128 straight to disk while the
        controller's in-memory copy still said 96 and last-writer-won."""
        from src.ui.label_panel import LabelPanel

        cfg_path = tmp_path / "config.json"
        cfg = AppConfig()  # MainWindow's startup load (density=96)
        cfg.save(cfg_path)

        pm = _make_classify_project(tmp_path)
        panel = LabelPanel(config_path=cfg_path, app_config=cfg)
        try:
            panel.set_project(pm)
            panel._view._density_slider.setValue(128)  # view persists 128

            # Controller path: whole save of the shared in-memory config
            # (what ProjectController / closeEvent do).
            cfg.add_recent_project(str(pm.project_dir))
            cfg.save(cfg_path)

            reloaded = AppConfig.load(cfg_path)
            assert reloaded.classify.grid_density == 128  # not reverted to 96
        finally:
            _teardown(qapp, panel)


class TestSliceWiring:
    def test_view_slice_is_shared_config_slice(self, qapp, tmp_path):
        """Wiring identity (label-store precedent): the view built through
        LabelPanel(app_config=...) must hold the SAME ClassifyViewState
        instance as the shared AppConfig — a private fallback slice would
        silently restore the double authority."""
        from src.ui.label_panel import LabelPanel

        cfg = AppConfig()
        pm = _make_classify_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json", app_config=cfg)
        try:
            panel.set_project(pm)
            assert panel._view._state is cfg.classify
        finally:
            _teardown(qapp, panel)

    def test_custom_config_path_receives_view_saves(self, qapp, tmp_path):
        """Bug ② regression: the view must persist to the injected
        config_path, never a hardcoded ~/.autolabel/config.json."""
        from src.ui.label_panel import LabelPanel

        cfg_path = tmp_path / "custom" / "cfg.json"
        cfg = AppConfig()
        pm = _make_classify_project(tmp_path)
        panel = LabelPanel(config_path=cfg_path, app_config=cfg)
        try:
            panel.set_project(pm)
            panel._view._on_preview_close()
            assert cfg_path.exists()
            assert AppConfig.load(cfg_path).classify.preview_visible is False
        finally:
            _teardown(qapp, panel)


class TestViewBehavior:
    def _make_view(self, state=None, save_config=None):
        from src.ui.views.classify import ClassifyView
        from src.utils.image import ImageCache

        return ClassifyView(
            ImageCache(max_count=2, max_memory_mb=8.0), OrderedDict(),
            classify_state=state, save_config=save_config,
        )

    def test_changes_update_slice_and_call_saver(self, qapp, tmp_path):
        state = ClassifyViewState()
        calls = []
        view = self._make_view(state, lambda: calls.append(1))
        try:
            view._on_density_changed(160)
            assert state.grid_density == 160

            pm = _make_classify_project(tmp_path)
            view.set_project(pm)
            view._sort_combo.setCurrentIndex(1)
            assert state.grid_sort == "class"

            view._save_preview_state(width=280, visible=False)
            assert state.preview_width == 280
            assert state.preview_visible is False

            assert len(calls) >= 4
        finally:
            view.cleanup()
            view.deleteLater()
            qapp.processEvents()

    def test_none_injection_is_safe_and_writes_nothing(self, qapp, tmp_path, monkeypatch):
        import pathlib

        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
        view = self._make_view()  # isolated: private slice + no-op saver
        try:
            view._on_density_changed(128)
            view._save_preview_state(width=200, visible=False)
            assert view._state.grid_density == 128
            assert not (tmp_path / ".autolabel" / "config.json").exists()
        finally:
            view.cleanup()
            view.deleteLater()
            qapp.processEvents()


class TestNoConfigLoadInUi:
    def test_src_ui_never_loads_appconfig_from_disk(self):
        """Structural guard (test_label_io_guard precedent): src/ui must not
        contain AppConfig.load / a hardcoded config path — mid-session disk
        round-trips recreate the double authority."""
        ui_dir = Path(__file__).resolve().parents[2] / "src" / "ui"
        offenders = []
        for py in ui_dir.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            if "AppConfig.load" in text or "_APP_CONFIG_PATH" in text:
                offenders.append(str(py))
        assert offenders == []
