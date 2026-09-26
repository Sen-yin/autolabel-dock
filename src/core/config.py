"""Global application configuration (~/.autolabel/config.json).

Single in-memory authority: MainWindow loads AppConfig once at startup and
every writer (ProjectController, closeEvent, ClassifyView via its injected
slice + saver) mutates that shared instance before saving — no other code
should call ``AppConfig.load`` mid-session (a disk round-trip would create a
second authority and last-writer-wins data loss).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClassifyViewState:
    """Persisted UI state for the classify view (a slice of AppConfig).

    The view holds the SAME mutable instance as ``AppConfig.classify`` —
    mutating a field updates the in-memory authority; persisting is the
    owner-injected saver's job. Serialized flat (``classify_grid_density``
    etc.) for zero disk-format change.
    """

    grid_density: int = 96
    grid_sort: str = "filename"  # "filename" | "class"
    preview_width: int = 320
    preview_visible: bool = True


@dataclass
class AppConfig:
    """Global app settings persisted across sessions."""

    recent_projects: list[str] = field(default_factory=list)
    script_tools: dict[str, str] = field(default_factory=dict)
    window_geometry: dict[str, int] = field(
        default_factory=lambda: {"x": 100, "y": 100, "width": 1400, "height": 900}
    )
    classify: ClassifyViewState = field(default_factory=ClassifyViewState)
    annotation_panel_splitter_sizes: list[int] = field(default_factory=list)
    annotation_panel_collapsed: dict[str, bool] = field(default_factory=dict)
    # Experimental: master switch to fully hide the optional LocateAnything
    # text-labeling backend. Defaults True (visible); set False to hide it
    # entirely. Old config.json files without the key load as True.
    enable_locateanything: bool = True
    # Annotation canvas guides (辅助线). Persisted so the user's preferred
    # helper layout survives restart. Defaults: ONLY the mouse cursor
    # crosshair is on. The rule-of-thirds / center crosshair and the
    # extra grid are opt-in — at full image scale they're busy enough
    # that almost everyone wants them off.
    canvas_center_lines: bool = False
    canvas_grid_density: int = 0
    canvas_cursor_crosshair: bool = True

    def to_dict(self) -> dict:
        return {
            "recent_projects": self.recent_projects,
            "script_tools": self.script_tools,
            "window_geometry": self.window_geometry,
            # Classify slice is flattened back to the legacy flat keys so the
            # on-disk format is unchanged for live fields.
            "classify_grid_density": self.classify.grid_density,
            "classify_grid_sort": self.classify.grid_sort,
            "classify_preview_width": self.classify.preview_width,
            "classify_preview_visible": self.classify.preview_visible,
            "annotation_panel_splitter_sizes": self.annotation_panel_splitter_sizes,
            "annotation_panel_collapsed": self.annotation_panel_collapsed,
            "enable_locateanything": self.enable_locateanything,
            "canvas_center_lines": self.canvas_center_lines,
            "canvas_grid_density": self.canvas_grid_density,
            "canvas_cursor_crosshair": self.canvas_cursor_crosshair,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AppConfig:
        raw_script_tools = d.get("script_tools", {})
        script_tools: dict[str, str] = {}
        if isinstance(raw_script_tools, dict):
            script_tools = {
                str(k): v for k, v in raw_script_tools.items() if isinstance(v, str)
            }

        raw_sizes = d.get("annotation_panel_splitter_sizes", [])
        splitter_sizes: list[int] = []
        if isinstance(raw_sizes, list):
            splitter_sizes = [int(x) for x in raw_sizes if isinstance(x, int)]

        raw_collapsed = d.get("annotation_panel_collapsed", {})
        collapsed: dict[str, bool] = {}
        if isinstance(raw_collapsed, dict):
            collapsed = {
                str(k): bool(v) for k, v in raw_collapsed.items()
                if isinstance(v, bool)
            }

        # Classify slice — picked from the legacy flat keys, with per-field
        # type sanitization (invalid types fall back to defaults).
        defaults = ClassifyViewState()
        raw_density = d.get("classify_grid_density", defaults.grid_density)
        grid_density = (
            raw_density
            if isinstance(raw_density, int) and not isinstance(raw_density, bool)
            else defaults.grid_density
        )
        raw_sort = d.get("classify_grid_sort", defaults.grid_sort)
        grid_sort = raw_sort if isinstance(raw_sort, str) else defaults.grid_sort
        raw_width = d.get("classify_preview_width", defaults.preview_width)
        preview_width = (
            raw_width
            if isinstance(raw_width, int) and not isinstance(raw_width, bool)
            else defaults.preview_width
        )
        raw_visible = d.get("classify_preview_visible", defaults.preview_visible)
        preview_visible = (
            raw_visible if isinstance(raw_visible, bool) else defaults.preview_visible
        )

        return cls(
            recent_projects=d.get("recent_projects", []),
            script_tools=script_tools,
            window_geometry=d.get("window_geometry", {"x": 100, "y": 100, "width": 1400, "height": 900}),
            classify=ClassifyViewState(
                grid_density=grid_density,
                grid_sort=grid_sort,
                preview_width=preview_width,
                preview_visible=preview_visible,
            ),
            annotation_panel_splitter_sizes=splitter_sizes,
            annotation_panel_collapsed=collapsed,
            enable_locateanything=bool(d.get("enable_locateanything", True)),
            canvas_center_lines=bool(d.get("canvas_center_lines", False)),
            canvas_grid_density=(
                int(d["canvas_grid_density"])
                if isinstance(d.get("canvas_grid_density"), int)
                and not isinstance(d.get("canvas_grid_density"), bool)
                else 0
            ),
            canvas_cursor_crosshair=bool(
                d.get("canvas_cursor_crosshair", True)
            ),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> AppConfig:
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return cls()

    def add_recent_project(self, project_path: str) -> None:
        """Add a project to recent list (most recent first, max 10)."""
        if project_path in self.recent_projects:
            self.recent_projects.remove(project_path)
        self.recent_projects.insert(0, project_path)
        self.recent_projects = self.recent_projects[:10]
