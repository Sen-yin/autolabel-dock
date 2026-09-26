"""Tests for global app configuration."""
import json
from pathlib import Path

from src.core.config import AppConfig, ClassifyViewState


class TestAppConfig:
    def test_default_values(self):
        cfg = AppConfig()
        assert cfg.recent_projects == []
        assert cfg.script_tools == {}
        assert cfg.enable_locateanything is True

    def test_save_and_load(self, tmp_path):
        config_path = tmp_path / "config.json"
        cfg = AppConfig(recent_projects=["/path/a", "/path/b"])
        cfg.save(config_path)
        loaded = AppConfig.load(config_path)
        assert loaded.recent_projects == ["/path/a", "/path/b"]

    def test_load_missing_returns_default(self, tmp_path):
        cfg = AppConfig.load(tmp_path / "missing.json")
        assert cfg.recent_projects == []

    def test_add_recent_project(self):
        cfg = AppConfig()
        cfg.add_recent_project("/proj/a")
        cfg.add_recent_project("/proj/b")
        cfg.add_recent_project("/proj/a")  # move to front
        assert cfg.recent_projects[0] == "/proj/a"
        assert cfg.recent_projects[1] == "/proj/b"
        assert len(cfg.recent_projects) == 2

    def test_recent_projects_max_10(self):
        cfg = AppConfig()
        for i in range(15):
            cfg.add_recent_project(f"/proj/{i}")
        assert len(cfg.recent_projects) == 10
        assert cfg.recent_projects[0] == "/proj/14"

    def test_script_tools_roundtrip(self, tmp_path):
        config_path = tmp_path / "config.json"
        cfg = AppConfig(script_tools={"crop": "print('crop')\n"})
        cfg.save(config_path)
        loaded = AppConfig.load(config_path)
        assert loaded.script_tools == {"crop": "print('crop')\n"}


class TestClassifySlice:
    def test_slice_defaults(self):
        cfg = AppConfig()
        assert isinstance(cfg.classify, ClassifyViewState)
        assert cfg.classify.grid_density == 96
        assert cfg.classify.grid_sort == "filename"
        assert cfg.classify.preview_width == 320
        assert cfg.classify.preview_visible is True

    def test_slice_instances_are_independent(self):
        # default_factory — two configs must not share one slice.
        a, b = AppConfig(), AppConfig()
        a.classify.grid_density = 128
        assert b.classify.grid_density == 96

    def test_to_dict_flattens_to_legacy_flat_keys(self):
        cfg = AppConfig(classify=ClassifyViewState(
            grid_density=128, grid_sort="class",
            preview_width=400, preview_visible=False,
        ))
        d = cfg.to_dict()
        assert d["classify_grid_density"] == 128
        assert d["classify_grid_sort"] == "class"
        assert d["classify_preview_width"] == 400
        assert d["classify_preview_visible"] is False
        assert "classify" not in d  # no nested key on disk

    def test_from_dict_picks_flat_keys(self):
        cfg = AppConfig.from_dict({
            "classify_grid_density": 144,
            "classify_grid_sort": "class",
            "classify_preview_width": 280,
            "classify_preview_visible": False,
        })
        assert cfg.classify.grid_density == 144
        assert cfg.classify.grid_sort == "class"
        assert cfg.classify.preview_width == 280
        assert cfg.classify.preview_visible is False

    def test_from_dict_sanitizes_invalid_types(self):
        cfg = AppConfig.from_dict({
            "classify_grid_density": "big",
            "classify_grid_sort": 7,
            "classify_preview_width": None,
            "classify_preview_visible": "yes",
        })
        assert cfg.classify.grid_density == 96
        assert cfg.classify.grid_sort == "filename"
        assert cfg.classify.preview_width == 320
        assert cfg.classify.preview_visible is True

    def test_roundtrip(self, tmp_path):
        config_path = tmp_path / "config.json"
        cfg = AppConfig()
        cfg.classify.grid_density = 128
        cfg.classify.grid_sort = "class"
        cfg.classify.preview_width = 400
        cfg.classify.preview_visible = False
        cfg.save(config_path)
        loaded = AppConfig.load(config_path)
        assert loaded.classify == cfg.classify

    def test_legacy_file_without_classify_keys(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"recent_projects": []}))
        cfg = AppConfig.load(config_path)
        assert cfg.classify == ClassifyViewState()


class TestLegacyCompat:
    """Compatibility matrix (a): a full 16-key legacy file must load losslessly
    for all 11 live concepts, with the 5 dead keys silently ignored."""

    LEGACY_16_KEYS = {
        "recent_projects": ["/x", "/y"],
        "theme": "dark",
        "auto_save": True,
        "default_conf_threshold": 0.5,
        "default_iou_threshold": 0.45,
        "overlap_iou_threshold": 0.5,
        "script_tools": {"crop": "print('c')\n"},
        "window_geometry": {"x": 1, "y": 2, "width": 800, "height": 600},
        "classify_grid_density": 128,
        "classify_grid_sort": "class",
        "classify_preview_width": 400,
        "classify_preview_visible": False,
        "annotation_panel_splitter_sizes": [120, 220],
        "annotation_panel_collapsed": {"Tag": True},
        "enable_locateanything": False,
    }

    def test_full_legacy_file_loads_live_fields(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(self.LEGACY_16_KEYS))
        cfg = AppConfig.load(config_path)
        assert cfg.recent_projects == ["/x", "/y"]
        assert cfg.script_tools == {"crop": "print('c')\n"}
        assert cfg.window_geometry == {"x": 1, "y": 2, "width": 800, "height": 600}
        assert cfg.classify.grid_density == 128
        assert cfg.classify.grid_sort == "class"
        assert cfg.classify.preview_width == 400
        assert cfg.classify.preview_visible is False
        assert cfg.annotation_panel_splitter_sizes == [120, 220]
        assert cfg.annotation_panel_collapsed == {"Tag": True}
        assert cfg.enable_locateanything is False

    def test_dead_fields_are_gone(self):
        cfg = AppConfig()
        for dead in ("theme", "auto_save", "default_conf_threshold",
                     "default_iou_threshold", "overlap_iou_threshold"):
            assert not hasattr(cfg, dead)
            assert dead not in cfg.to_dict()

    def test_resave_drops_dead_keys_keeps_live_key_names(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(self.LEGACY_16_KEYS))
        cfg = AppConfig.load(config_path)
        cfg.save(config_path)
        d = json.loads(config_path.read_text())
        assert "theme" not in d and "auto_save" not in d
        # Live flat key names unchanged on disk.
        assert d["classify_grid_density"] == 128
        assert d["classify_preview_visible"] is False


class TestAnnotationPanelPersistence:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.annotation_panel_splitter_sizes == []
        assert cfg.annotation_panel_collapsed == {}

    def test_round_trip(self):
        cfg = AppConfig()
        cfg.annotation_panel_splitter_sizes = [120, 220, 90, 80, 160]
        cfg.annotation_panel_collapsed = {"属性": True, "Tag": False}

        restored = AppConfig.from_dict(cfg.to_dict())

        assert restored.annotation_panel_splitter_sizes == [120, 220, 90, 80, 160]
        assert restored.annotation_panel_collapsed == {"属性": True, "Tag": False}

    def test_backward_compat_missing_keys(self):
        # Older config.json without the new keys must load with defaults.
        legacy = {"recent_projects": ["/x"], "theme": "dark"}
        cfg = AppConfig.from_dict(legacy)
        assert cfg.annotation_panel_splitter_sizes == []
        assert cfg.annotation_panel_collapsed == {}
        assert cfg.recent_projects == ["/x"]

    def test_invalid_types_fall_back_to_defaults(self):
        # Defensive: corrupted entries should not crash from_dict.
        bad = {
            "annotation_panel_splitter_sizes": "not a list",
            "annotation_panel_collapsed": ["not", "a", "dict"],
        }
        cfg = AppConfig.from_dict(bad)
        assert cfg.annotation_panel_splitter_sizes == []
        assert cfg.annotation_panel_collapsed == {}
