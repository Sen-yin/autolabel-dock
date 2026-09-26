"""Annotation canvas widget for image display and annotation editing."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QMenu, QAction, QInputDialog
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import (
    QPainter,
    QPen,
    QBrush,
    QColor,
    QPixmap,
    QImage,
    QImageReader,
    QFont,
    QCursor,
    QPolygonF,
    QWheelEvent,
    QMouseEvent,
    QPaintEvent,
    QResizeEvent,
)

from src.core.annotation import Annotation, Keypoint
from src.core.polygon import bbox_from_polygon, nearest_vertex, nearest_edge
from src.ui.theme import PALETTE

logger = logging.getLogger(__name__)

# Visual constants
HANDLE_SIZE = 6
KEYPOINT_RADIUS = 5
LABEL_FONT_SIZE = 11
LABEL_PADDING = 3
MIN_SCALE = 0.1
MAX_SCALE = 20.0
ZOOM_FACTOR = 1.15


class AnnotationCanvas(QWidget):
    """Canvas widget for displaying images and editing annotations.

    Signals:
        annotation_created(Annotation): New annotation drawn by user.
        annotation_modified(str): Annotation with given ID was moved/resized.
        annotation_selected(str): Annotation with given ID was selected (or None).
        annotation_deleted(str): Annotation with given ID should be deleted.
        class_requested(float, float): Request class picker at pixel position (after drawing).
        annotations_changed(): Any change to annotations occurred.
    """

    annotation_created = pyqtSignal(object)   # Annotation
    annotation_modified = pyqtSignal(str)     # annotation id
    annotation_selected = pyqtSignal(object)  # annotation id or None
    annotation_deleted = pyqtSignal(str)      # annotation id
    annotation_copied = pyqtSignal(str)       # annotation id (for copy via right-click)
    class_requested = pyqtSignal(float, float)
    class_change_requested = pyqtSignal(str, float, float)  # ann_id, px, py
    annotations_changed = pyqtSignal()
    keypoint_attach_requested = pyqtSignal(str, float, float)  # ann_id, px, py
    keypoint_selected = pyqtSignal(str, int)  # ann_id, kp_index
    # SAM prompt forwarding (normalized coords). The canvas only relays the
    # interaction — it has no knowledge of any segmentation backend.
    sam_point_requested = pyqtSignal(float, float)              # nx, ny
    sam_box_requested = pyqtSignal(float, float, float, float)  # x1, y1, x2, y2

    zoom_changed = pyqtSignal(float)  # current scale factor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(200, 200)

        # Image state
        self._image: QPixmap | None = None
        self._image_w: int = 0
        self._image_h: int = 0

        # View transform
        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0

        # Tool mode: "select", "draw_bbox", "draw_keypoint", "draw_polygon",
        # "sam_prompt"
        self.tool_mode: str = "select"

        # Annotations
        self._annotations: list[Annotation] = []
        self._selected_id: str | None = None
        self._selected_kp_idx: int | None = None
        self._class_colors: dict[str, str] = {}

        # Drawing state
        self._drawing: bool = False
        self._draw_start: tuple[float, float] | None = None  # normalized
        self._draw_current: tuple[float, float] | None = None  # normalized
        # In-progress polygon vertices (normalized). Non-empty only while a
        # draw_polygon session is open.
        self._polygon_points: list[list[float]] = []

        # Dragging state (move/resize)
        self._dragging: bool = False
        self._drag_type: str = ""  # "move", "resize_tl", "resize_br", etc., "move_kp", "move_poly_vertex"
        self._drag_ann_id: str | None = None
        self._drag_kp_idx: int = -1
        self._drag_vertex_idx: int = -1
        self._drag_start_norm: tuple[float, float] | None = None
        self._drag_ann_snapshot: dict | None = None

        # Panning state
        self._panning: bool = False
        self._pan_start: tuple[float, float] | None = None

        # Conflict pairs: {ann_id: paired_ann_id} (bidirectional)
        self._conflict_pairs: dict[str, str] = {}

        # ── Annotation guides (辅助线) ─────────────────────────
        # Three layers, each opt-in. All render only over the image rect —
        # never bleed into the empty canvas area — so they never fight with
        # the "无图片" placeholder.
        #
        #   _center_lines_visible (default False):
        #       "rule of thirds" — two horizontal + two vertical lines at
        #       33% / 67%, plus a brighter center crosshair. Opt-in because
        #       at full image scale these lines dominate the picture — most
        #       users only want the cursor crosshair.
        #   _grid_density (int):
        #       Extra N-equally-spaced grid (per axis). 0 = off, 3 = thirds
        #       overlay, 5 = fifths, etc. Stacks cleanly on top of the
        #       rule-of-thirds lines.
        #   _cursor_crosshair_visible (default True):
        #       Mouse-tracking cross + live pixel coordinate badge. The
        #       default-on helper: unobtrusive when the cursor isn't near
        #       the image, immediately useful when it is.
        self._center_lines_visible: bool = False
        self._grid_density: int = 0
        self._cursor_crosshair_visible: bool = True
        # Mouse position in widget coords (for the live crosshair). Stored
        # independently of any drag state so we can clear it cleanly on
        # mouse-leave and keep the tracker pinned to the cursor without
        # fighting the various drag handlers above.
        self._mouse_pos_widget: tuple[int, int] | None = None

    # Cached pens / brushes for the cursor crosshair. Recomputed lazily
    # so the colors stay in sync with PALETTE without paying the cost
    # on every paintEvent. Cache is invalidated by ``_invalidate_guide_cache``
    # if we ever want to support live theme switching; for now the
    # values are stable across a session.
    _GUIDE_PENS: dict[str, object] | None = None

    def _guide_pens(self) -> dict[str, object]:
        """Return the (reusable) pen/brush cache for guide rendering."""
        if AnnotationCanvas._GUIDE_PENS is None:
            AnnotationCanvas._GUIDE_PENS = {
                # Halo line — drawn first, then the bright line on top
                # gives a "high-contrast outline" look that survives any
                # image content. Width kept thin (1.5) so the crosshair
                # reads as a hairline, not a fat bar. Halo alpha is high
                # enough that the line is still clearly visible against
                # near-white or near-black pictures.
                "cursor_halo": QPen(QColor(0, 0, 0, 140), 1.5, Qt.SolidLine),
                # Bright cyan line — single-pixel hairline. Drawn on top
                # of the halo for the standard image-editor crosshair look.
                "cursor_main": QPen(QColor(56, 189, 248, 255), 1, Qt.SolidLine),
                # Outer ring of the center marker. Kept small (radius 3)
                # so it doesn't look like a thumbtack stuck on the image.
                "marker_halo": QPen(QColor(0, 0, 0, 180), 1, Qt.SolidLine),
                # Inner ring — the "you are here" pip. Tiny but obvious.
                "marker_main": QPen(QColor(125, 211, 252, 255), 1, Qt.SolidLine),
                # Badge background — darker & more opaque than before so
                # the white text always wins.
                "badge_bg": QColor(12, 18, 28, 220),
                # Badge text color — near-white, full alpha.
                "badge_fg": QColor(245, 250, 255, 255),
                # Badge border — same color as the cursor main line so
                # the badge reads as part of the same overlay.
                "badge_border": QPen(QColor(56, 189, 248, 255), 1, Qt.SolidLine),
            }
        return AnnotationCanvas._GUIDE_PENS


    # ── Coordinate transforms ──────────────────────────────────

    def norm_to_pixel(self, nx: float, ny: float) -> tuple[float, float]:
        """Convert normalized [0,1] image coords to widget pixel coords."""
        px = nx * self._image_w * self._scale + self._offset_x
        py = ny * self._image_h * self._scale + self._offset_y
        return px, py

    def pixel_to_norm(self, px: float, py: float) -> tuple[float, float]:
        """Convert widget pixel coords to normalized [0,1] image coords."""
        if self._image_w == 0 or self._image_h == 0 or self._scale == 0:
            return 0.0, 0.0
        nx = (px - self._offset_x) / (self._image_w * self._scale)
        ny = (py - self._offset_y) / (self._image_h * self._scale)
        return nx, ny

    def _clamp_norm(self, nx: float, ny: float) -> tuple[float, float]:
        """Clamp normalized coords to [0, 1]."""
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    # ── Public API ─────────────────────────────────────────────

    def load_image(self, path: str) -> None:
        """Load and display an image, fit to window."""
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        qimage = reader.read()
        if qimage.isNull():
            logger.warning("Failed to load image: %s", path)
            return
        self.set_pixmap(QPixmap.fromImage(qimage))

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Set a pre-loaded pixmap as the display image."""
        self._image = pixmap
        self._image_w = pixmap.width()
        self._image_h = pixmap.height()
        self._fit_to_window()
        self.update()

    def set_annotations(self, annotations: list[Annotation]) -> None:
        """Set the annotations to display."""
        self._annotations = list(annotations)
        self._selected_id = None
        # The annotation set was replaced (image switch, undo restore, reload):
        # an in-progress polygon draft is stale — drop it so it can't render
        # over (or be committed onto) the wrong image. Bbox/keypoint drafts
        # have no such window (they only exist while the mouse button is held).
        if self._polygon_points:
            self._polygon_points = []
            if not self._drawing:
                self._draw_start = None
                self._draw_current = None
        self.update()

    def set_class_colors(self, colors: dict[str, str]) -> None:
        """Set class name -> hex color mapping."""
        self._class_colors = colors
        self.update()

    def select_annotation(self, ann_id: str | None) -> None:
        """Select an annotation by ID, or deselect with None."""
        self._selected_id = ann_id
        self._selected_kp_idx = None
        self.annotation_selected.emit(ann_id)
        self.update()

    def set_tool_mode(self, mode: str) -> None:
        """Set tool mode: 'select', 'draw_bbox', 'draw_keypoint', 'draw_polygon', 'sam_prompt'."""
        self.tool_mode = mode
        self._drawing = False
        self._draw_start = None
        self._draw_current = None
        self._polygon_points = []
        if mode == "select":
            self.setCursor(Qt.ArrowCursor)
        elif mode in ("draw_bbox", "draw_keypoint", "draw_polygon", "sam_prompt"):
            self.setCursor(Qt.CrossCursor)

    def set_locked(self, locked: bool) -> None:
        """Set lock state (no-op, kept for API compatibility)."""
        pass

    # ── Annotation guides (辅助线) setters ─────────────────────
    # Each setter mutates one bit of state and triggers a repaint. Cheap
    # enough to call on every toggle — there's no expensive allocation
    # involved, only a flag flip + ``update()``.
    #
    # Names mirror the underlying attributes so callers (toolbar, command
    # palette, future hotkey bindings) can find them via grep:
    # ``set_center_lines_visible``, ``set_grid_density``,
    # ``set_cursor_crosshair_visible``.
    #
    # ``grid_density`` semantics — N is the number of cells per axis, not
    # the number of lines:
    #   * 0 — no extra grid (only rule-of-thirds + center, if enabled)
    #   * 1 — one center line per axis (rare; degenerate; behaves like 2)
    #   * 2 — halves
    #   * 3 — thirds (rule-of-thirds is already this; renderer skips the
    #         duplicate if rule-of-thirds is on)
    #   * 4 — quarters
    #   * ... etc.
    #
    # Negative input is clamped to 0 — defensive, no exception.

    def set_center_lines_visible(self, visible: bool) -> None:
        """Show or hide the rule-of-thirds + center crosshair guide."""
        self._center_lines_visible = bool(visible)
        self.update()

    def set_grid_density(self, n: int) -> None:
        """Set the extra N-cells-per-axis grid density. 0 disables it."""
        try:
            n = int(n)
        except (TypeError, ValueError):
            return
        self._grid_density = max(0, n)
        self.update()

    def set_cursor_crosshair_visible(self, visible: bool) -> None:
        """Show or hide the mouse-tracking crosshair + live coord badge."""
        self._cursor_crosshair_visible = bool(visible)
        if not self._cursor_crosshair_visible:
            # Clear the coordinate badge so it stops lingering on the last
            # known mouse position when the user disables the helper.
            self._mouse_pos_norm = None
        self.update()

    # Public read-only views for the settings UI to echo current values
    # back into its combo / checkbox state when rebuilding itself.
    @property
    def center_lines_visible(self) -> bool:
        return self._center_lines_visible

    @property
    def grid_density(self) -> int:
        return self._grid_density

    @property
    def cursor_crosshair_visible(self) -> bool:
        return self._cursor_crosshair_visible

    @property
    def annotations(self) -> list[Annotation]:
        """Return the current annotations list (mutable reference)."""
        return self._annotations

    @annotations.setter
    def annotations(self, value: list[Annotation]) -> None:
        """Replace annotations list."""
        self._annotations = value
        self.update()

    @property
    def is_locked(self) -> bool:
        """Return whether the canvas is in locked (view-only) mode."""
        return False

    def add_annotation(self, ann: Annotation) -> None:
        """Append an annotation to the canvas and repaint."""
        self._annotations.append(ann)
        self.update()

    def add_annotations(self, anns: list[Annotation]) -> None:
        """Append multiple annotations and repaint once."""
        self._annotations.extend(anns)
        self.update()

    def remove_annotation(self, ann_id: str) -> None:
        """Remove annotation by ID and clear selection."""
        self._annotations = [a for a in self._annotations if a.id != ann_id]
        self._selected_id = None
        self.update()

    def clear_draw_state(self) -> None:
        """Clear in-progress drawing state."""
        self._draw_start = None
        self._draw_current = None
        self._polygon_points = []
        self.update()

    def consume_draw_start(self) -> tuple[float, float] | None:
        """Return and clear the pending draw-origin (normalized), or None.

        Single-shot read used by the keypoint-attach flow: snapshotting at
        entry means later canvas events (e.g. while a picker popup is open)
        cannot rewrite the origin under the caller.
        """
        start = self._draw_start
        self._draw_start = None
        return start

    def clear(self) -> None:
        """Clear image and annotations."""
        self._image = None
        self._image_w = 0
        self._image_h = 0
        self._annotations = []
        self._selected_id = None
        self._drawing = False
        self._draw_start = None
        self._draw_current = None
        self._polygon_points = []
        self._conflict_pairs.clear()
        self.update()

    # ── Conflict pair management ──────────────────────────────

    def set_conflict_pairs(self, pairs: list[tuple[str, str]]) -> None:
        """Set conflict pairs. Each pair is (existing_id, pred_id)."""
        for eid, pid in pairs:
            self._conflict_pairs[eid] = pid
            self._conflict_pairs[pid] = eid
        self.update()

    def resolve_conflict(self, keep_id: str) -> None:
        """Keep one annotation from a conflict pair and remove the other."""
        remove_id = self._conflict_pairs.get(keep_id)
        if not remove_id:
            return
        # Clean up mapping (both directions)
        self._conflict_pairs.pop(keep_id, None)
        self._conflict_pairs.pop(remove_id, None)
        # Remove the losing annotation
        self._annotations = [a for a in self._annotations if a.id != remove_id]
        if self._selected_id == remove_id:
            self._selected_id = None
        self.annotation_deleted.emit(remove_id)
        self.annotations_changed.emit()
        self.update()

    def clear_conflicts(self) -> None:
        """Clear all conflict state."""
        self._conflict_pairs.clear()
        self.update()

    def get_selected_annotation(self) -> Annotation | None:
        """Return the currently selected annotation."""
        if self._selected_id is None:
            return None
        for ann in self._annotations:
            if ann.id == self._selected_id:
                return ann
        return None

    def hit_test(self, px: float, py: float) -> str | None:
        """Find annotation at pixel position. Returns annotation ID or None."""
        nx, ny = self.pixel_to_norm(px, py)

        # Check keypoints first (smaller targets, higher priority)
        kp_radius_norm_x = KEYPOINT_RADIUS / (self._image_w * self._scale) if self._image_w * self._scale > 0 else 0
        kp_radius_norm_y = KEYPOINT_RADIUS / (self._image_h * self._scale) if self._image_h * self._scale > 0 else 0

        for ann in reversed(self._annotations):
            for kp in ann.keypoints:
                if abs(kp.x - nx) < kp_radius_norm_x * 2 and abs(kp.y - ny) < kp_radius_norm_y * 2:
                    return ann.id

        # Check bboxes
        for ann in reversed(self._annotations):
            if ann.bbox:
                cx, cy, w, h = ann.bbox
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
                if x1 <= nx <= x2 and y1 <= ny <= y2:
                    return ann.id

        return None

    def _fit_to_window(self) -> None:
        """Scale and offset so image fits in widget."""
        if self._image_w == 0 or self._image_h == 0:
            return
        ww, wh = self.width(), self.height()
        if ww <= 0 or wh <= 0:
            return
        sx = ww / self._image_w
        sy = wh / self._image_h
        self._scale = min(sx, sy)
        # Center the image
        self._offset_x = (ww - self._image_w * self._scale) / 2
        self._offset_y = (wh - self._image_h * self._scale) / 2

    # ── Paint ──────────────────────────────────────────────────

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(PALETTE["canvas"]))

        if self._image is None:
            painter.setPen(QColor(PALETTE["text_subtle"]))
            painter.drawText(self.rect(), Qt.AlignCenter, "无图片")
            painter.end()
            return

        # Draw image
        dest = QRectF(
            self._offset_x, self._offset_y,
            self._image_w * self._scale, self._image_h * self._scale,
        )
        painter.drawPixmap(dest.toRect(), self._image)

        # ── Annotation guides (辅助线) ─────────────────────────
        # Drawn after the pixmap, BEFORE annotations + in-progress
        # previews. This ordering is intentional:
        #   1. Image -> 2. Guides -> 3. Existing boxes -> 4. Draft box
        # so the user's draft always wins on top — they never see the
        # guide line "cover up" what they're currently drawing.
        # All guide rendering is clipped to the image rect via a saved
        # painter state so that as the user pans away, the lines stop at
        # the image edge instead of marching across empty canvas.
        self._paint_guides(painter, dest)

        # Viewport bounds for culling
        vp_left = 0.0
        vp_top = 0.0
        vp_right = float(self.width())
        vp_bottom = float(self.height())

        # LOD: skip labels at very small zoom
        draw_labels = self._scale >= 0.3

        # Draw annotations with viewport culling
        for ann in self._annotations:
            # Cull: skip if annotation is entirely outside viewport
            if ann.bbox and not self._ann_in_viewport(ann, vp_left, vp_top, vp_right, vp_bottom):
                continue
            is_selected = ann.id == self._selected_id
            color = QColor(self._class_colors.get(ann.class_name, PALETTE["primary"]))
            self._paint_annotation(painter, ann, color, is_selected, draw_labels)

        # Draw in-progress bbox
        if self._drawing and self._draw_start and self._draw_current:
            self._paint_drawing_preview(painter)

        # Draw in-progress polygon
        if self.tool_mode == "draw_polygon" and self._polygon_points:
            self._paint_polygon_preview(painter)

        # Zoom level indicator + lock badge
        if self._image is not None:
            font = QFont()
            font.setPixelSize(11)
            painter.setFont(font)
            painter.setPen(QColor(PALETTE["text_subtle"]))
            zoom_pct = int(self._scale * 100)
            status_text = f"{zoom_pct}%"
            painter.drawText(8, self.height() - 8, status_text)

        painter.end()

    def _ann_in_viewport(
        self, ann: Annotation, vp_left: float, vp_top: float, vp_right: float, vp_bottom: float
    ) -> bool:
        """Check if annotation bbox overlaps the viewport."""
        if not ann.bbox:
            return True  # keypoint-only annotations always drawn
        cx, cy, w, h = ann.bbox
        x1, y1 = self.norm_to_pixel(cx - w / 2, cy - h / 2)
        x2, y2 = self.norm_to_pixel(cx + w / 2, cy + h / 2)
        # Annotation is outside if entirely to the left, right, above, or below viewport
        if x2 < vp_left or x1 > vp_right or y2 < vp_top or y1 > vp_bottom:
            return False
        return True

    def _paint_annotation(
        self, painter: QPainter, ann: Annotation, color: QColor, selected: bool,
        draw_labels: bool = True,
    ) -> None:
        """Paint a single annotation (polygon/bbox + keypoints + label)."""
        in_conflict = ann.id in self._conflict_pairs
        if ann.bbox:
            cx, cy, w, h = ann.bbox
            x1, y1 = self.norm_to_pixel(cx - w / 2, cy - h / 2)
            x2, y2 = self.norm_to_pixel(cx + w / 2, cy + h / 2)

            # Pen shared by polygon and rect: conflict teal / unconfirmed
            # dashed / selected thicker.
            if in_conflict and not ann.confirmed:
                # Conflict prediction: teal dashed, thicker
                pen = QPen(QColor(PALETTE["teal"]), 3, Qt.DashLine)
            else:
                pen = QPen(color, 2)
                if not ann.confirmed:
                    pen.setStyle(Qt.DashLine)
            if selected:
                pen.setWidth(pen.width() + 1)
            painter.setPen(pen)

            if ann.polygon:
                # Filled polygon; the derived bbox drives label/culling but is
                # not itself drawn (avoids a double outline).
                fill = QColor(color)
                fill.setAlpha(50)
                painter.setBrush(QBrush(fill))
                qpoly = QPolygonF(
                    [QPointF(*self.norm_to_pixel(vx, vy)) for vx, vy in ann.polygon]
                )
                painter.drawPolygon(qpoly)
            else:
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

            # Label background (skip at low zoom for performance)
            if draw_labels or selected:
                label_text = ann.class_name
                if in_conflict:
                    label_text += " \u21c4"
                elif not ann.confirmed:
                    label_text += " \u26a1"
                font = QFont()
                font.setPixelSize(LABEL_FONT_SIZE)
                painter.setFont(font)
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(label_text) + LABEL_PADDING * 2
                th = fm.height() + LABEL_PADDING * 2
                label_rect = QRectF(x1, y1 - th, tw, th)
                if label_rect.top() < 0:
                    label_rect.moveTop(y1)
                bg_color = QColor(color)
                bg_color.setAlpha(200)
                painter.fillRect(label_rect, bg_color)
                painter.setPen(QColor(PALETTE["ink"]))
                painter.drawText(label_rect, Qt.AlignCenter, label_text)

            # Control handles when selected
            if selected:
                if ann.polygon:
                    self._paint_vertex_handles(painter, ann.polygon)
                else:
                    self._paint_handles(painter, x1, y1, x2, y2)

        # Keypoints
        for i, kp in enumerate(ann.keypoints):
            px, py = self.norm_to_pixel(kp.x, kp.y)
            is_kp_selected = selected and self._selected_kp_idx == i
            r = KEYPOINT_RADIUS + (3 if is_kp_selected else (2 if selected else 0))

            if kp.visible == 0:
                painter.setPen(QPen(QColor(PALETTE["text_subtle"]), 1))
                painter.setBrush(Qt.NoBrush)
            elif kp.visible == 1:
                painter.setPen(QPen(color, 1))
                painter.setBrush(Qt.NoBrush)
            else:
                painter.setPen(QPen(color, 1))
                painter.setBrush(QBrush(color))

            if is_kp_selected:
                painter.setPen(QPen(QColor(PALETTE["text"]), 2))

            painter.drawEllipse(QPointF(px, py), r, r)

            # Label for keypoint (show when selected or when individual kp is selected)
            if (selected and draw_labels) or is_kp_selected:
                painter.setPen(QColor(PALETTE["text"]))
                font = QFont()
                font.setPixelSize(10)
                painter.setFont(font)
                vis_text = ["inv", "occ", "vis"][kp.visible]
                label_text = f"{kp.label} ({vis_text})" if is_kp_selected else kp.label
                painter.drawText(int(px + r + 2), int(py - 2), label_text)

    def _paint_handles(self, painter: QPainter, x1: float, y1: float, x2: float, y2: float) -> None:
        """Paint resize handles on selected bbox corners."""
        painter.setPen(QPen(QColor(PALETTE["text"]), 1))
        painter.setBrush(QBrush(QColor(PALETTE["primary"])))
        hs = HANDLE_SIZE
        for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            painter.drawRect(QRectF(hx - hs, hy - hs, hs * 2, hs * 2))

    def _paint_vertex_handles(self, painter: QPainter, polygon: list[list[float]]) -> None:
        """Paint draggable handles on a selected polygon's vertices."""
        painter.setPen(QPen(QColor(PALETTE["text"]), 1))
        painter.setBrush(QBrush(QColor(PALETTE["primary"])))
        hs = HANDLE_SIZE
        for vx, vy in polygon:
            hx, hy = self.norm_to_pixel(vx, vy)
            painter.drawRect(QRectF(hx - hs, hy - hs, hs * 2, hs * 2))

    def _paint_guides(self, painter: QPainter, image_rect: QRectF) -> None:
        """Render annotation guides (rule-of-thirds, grid, cursor crosshair).

        Three independent layers, each gated by its own flag so users can
        mix-and-match. The brightest layer is the rule-of-thirds + center
        crosshair (the most useful for asymmetric box placement); the grid
        is the dimmest overlay (a pure reference, never a focal point); the
        cursor crosshair sits on top so it can never be hidden by the
        other two.

        Every layer is clipped to ``image_rect`` so the guides stop at
        the edge of the picture — they never march across the empty
        canvas area when the user pans.
        """
        # Empty image / not loaded yet — nothing to overlay.
        if self._image is None or image_rect.width() <= 0 or image_rect.height() <= 0:
            return
        # Skip rendering at very small zoom levels (the lines turn into a
        # solid blur that obscures the image content instead of helping).
        # 0.2 is below the existing label-LOD cutoff, so the helpers
        # disappear at the same zoom the labels do — visually consistent.
        if self._scale < 0.2:
            return

        ix = image_rect.x()
        iy = image_rect.y()
        iw = image_rect.width()
        ih = image_rect.height()

        # ── 1. Extra N-cells-per-axis grid (dimmest, rendered first) ──
        # The grid stays opt-in because it can be busy on small images;
        # the rule-of-thirds (later) is the always-on, calmer default.
        if self._grid_density > 1:
            n = self._grid_density
            grid_pen = QPen(
                QColor(120, 140, 155, 90),  # muted blue-gray, see-through
                1,
                Qt.DashDotLine,
            )
            painter.setPen(grid_pen)
            painter.setBrush(Qt.NoBrush)
            painter.save()
            painter.setClipRect(image_rect)
            # N cells means N-1 interior lines per axis (the image edges
            # are the outer walls of the first/last cell). For the
            # degenerate N=1 case we still want a meaningful reference
            # (a single center cross) so the user sees something is set.
            interior = max(0, n - 1)
            for k in range(1, n):
                frac = k / n
                vx = ix + iw * frac
                painter.drawLine(QPointF(vx, iy), QPointF(vx, iy + ih))
                hy = iy + ih * frac
                painter.drawLine(QPointF(ix, hy), QPointF(ix + iw, hy))
            if interior == 0:
                painter.drawLine(QPointF(ix + iw / 2, iy), QPointF(ix + iw / 2, iy + ih))
                painter.drawLine(QPointF(ix, iy + ih / 2), QPointF(ix + iw, iy + ih / 2))
            painter.restore()

        # ── 2. Rule-of-thirds + brighter center crosshair ─────────────
        # This is the default "obvious" helper — designed to be
        # instantly readable without ever overpowering the image.
        if self._center_lines_visible:
            painter.save()
            painter.setClipRect(image_rect)
            # Thirds lines at 1/3 and 2/3: dashed & slightly translucent.
            # They hint at structure; they don't fight with the picture.
            thirds_pen = QPen(
                QColor(56, 189, 248, 170),  # PALETTE primary @ ~67% alpha
                1,
                Qt.DashLine,
            )
            painter.setPen(thirds_pen)
            painter.setBrush(Qt.NoBrush)
            for frac in (1.0 / 3.0, 2.0 / 3.0):
                vx = ix + iw * frac
                painter.drawLine(QPointF(vx, iy), QPointF(vx, iy + ih))
                hy = iy + ih * frac
                painter.drawLine(QPointF(ix, hy), QPointF(ix + iw, hy))
            # Center crosshair — solid + brighter than the thirds lines,
            # because it's the single most useful reference when
            # eyeballing "is this box centered on the target?".
            cx = ix + iw / 2
            cy = iy + ih / 2
            center_pen = QPen(
                QColor(186, 230, 253, 230),  # primary_hover tone
                1,
                Qt.SolidLine,
            )
            painter.setPen(center_pen)
            painter.drawLine(QPointF(cx, iy), QPointF(cx, iy + ih))
            painter.drawLine(QPointF(ix, cy), QPointF(ix + iw, cy))
            # Short "+" caps right at the center pixel (±9 px). Subtle
            # but instantly readable — this is what makes the helper
            # feel "明显" without dominating the picture.
            cap = 9
            cap_pen = QPen(QColor(186, 230, 253, 255), 2, Qt.SolidLine)
            painter.setPen(cap_pen)
            painter.drawLine(QPointF(cx - cap, cy), QPointF(cx + cap, cy))
            painter.drawLine(QPointF(cx, cy - cap), QPointF(cx, cy + cap))
            painter.restore()

        # ── 3. Cursor crosshair + live pixel coord badge (top-most) ───
        # This is the only helper that's on by default, so we go the
        # extra mile on readability: solid lines (no dashes) + a dark
        # halo underneath the bright color + a filled ring at the
        # exact pixel, so the cursor is visible on any background
        # (white snow, black coal, bright neon, gray sky).
        if self._cursor_crosshair_visible and self._mouse_pos_widget is not None:
            mx, my = self._mouse_pos_widget
            # Quick reject: outside the image rect. Keeps the helpers
            # tied to the picture so they don't follow the cursor into
            # the empty canvas area.
            if ix <= mx <= ix + iw and iy <= my <= iy + ih:
                pens = self._guide_pens()
                painter.save()
                # Crosshair lines: full image rect width/height, solid.
                # Two passes — dark halo first, then the bright cyan
                # on top — gives a high-contrast outline that survives
                # any underlying image content.
                painter.setPen(pens["cursor_halo"])
                painter.setBrush(Qt.NoBrush)
                painter.drawLine(QPointF(ix, my), QPointF(ix + iw, my))
                painter.drawLine(QPointF(mx, iy), QPointF(mx, iy + ih))
                painter.setPen(pens["cursor_main"])
                painter.drawLine(QPointF(ix, my), QPointF(ix + iw, my))
                painter.drawLine(QPointF(mx, iy), QPointF(mx, iy + ih))

                # Small filled ring at the exact pixel — the "you are
                # here" marker. Halo + inner ring, both hairline and
                # tiny so it doesn't look like a thumbtack. The whole
                # marker is only ~7 px across — visible at a glance but
                # never blocking the pixel the user is about to click.
                painter.setPen(pens["marker_halo"])
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(mx, my), 4, 4)
                painter.setPen(pens["marker_main"])
                painter.drawEllipse(QPointF(mx, my), 2.5, 2.5)

                # Pixel-coordinate badge — anchored to the BOTTOM-LEFT
                # of the cursor. Why not the natural top-right: that
                # quadrant is where the user's hand naturally moves the
                # cursor next, so a badge there is constantly under the
                # pointer. Bottom-left sits next to the cursor in the
                # "outside of the action" quadrant, like Photoshop's
                # info readouts.
                nx, ny = self.pixel_to_norm(mx, my)
                px_i = max(0, min(self._image_w - 1, int(nx * self._image_w)))
                py_i = max(0, min(self._image_h - 1, int(ny * self._image_h)))
                label = f"{px_i}, {py_i}"
                font = QFont()
                font.setPixelSize(13)
                font.setBold(True)
                painter.setFont(font)
                metrics = painter.fontMetrics()
                text_w = metrics.horizontalAdvance(label)
                text_h = metrics.height()
                pad_x = 7
                pad_y = 5
                # Default: place the badge's top-right corner 12 px to
                # the LEFT of the cursor (so the badge body sits in the
                # bottom-left quadrant of the cursor). Falls back to
                # top-right when there's no room on the left (cursor
                # near the image's left edge) and falls back to top
                # when there's no room below (cursor near the bottom
                # edge). This 4-corner fallback keeps the badge fully
                # on the image at every cursor position.
                box_w = text_w + pad_x * 2
                box_h = text_h + pad_y * 2
                offset = 12
                # Preferred: bottom-left of cursor.
                box_x = mx - box_w - offset
                box_y = my + offset
                if box_x < ix:
                    # No room on the left — flip to the right.
                    box_x = mx + offset
                if box_y + box_h > iy + ih:
                    # No room below — flip above the cursor. Keep the
                    # same X axis the fallback chose so the badge
                    # doesn't appear to teleport horizontally.
                    box_y = my - box_h - offset
                badge_rect = QRectF(box_x, box_y, box_w, box_h)
                # Dark background — high opacity so the text reads
                # even on near-black image areas.
                painter.setPen(pens["badge_border"])
                painter.setBrush(pens["badge_bg"])
                painter.drawRect(badge_rect)
                # Text inside. ``AlignLeft | AlignVCenter`` is what we
                # want — left-aligned to the badge's padding, centered
                # vertically inside the (text_h + pad*2) box.
                painter.setPen(pens["badge_fg"])
                painter.drawText(
                    badge_rect.adjusted(pad_x, pad_y - 1, -pad_x, -pad_y - 1),
                    Qt.AlignLeft | Qt.AlignVCenter,
                    label,
                )
                painter.restore()

    def _paint_drawing_preview(self, painter: QPainter) -> None:
        """Paint the bbox being drawn with size HUD."""
        sx, sy = self.norm_to_pixel(*self._draw_start)
        ex, ey = self.norm_to_pixel(*self._draw_current)
        painter.setPen(QPen(QColor(PALETTE["primary"]), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        x = min(sx, ex)
        y = min(sy, ey)
        w = abs(ex - sx)
        h = abs(ey - sy)
        painter.drawRect(QRectF(x, y, w, h))

        # Size HUD (pixel dimensions)
        if self._image_w > 0 and self._image_h > 0:
            ns = self._draw_start
            nc = self._draw_current
            pw = int(abs(nc[0] - ns[0]) * self._image_w)
            ph = int(abs(nc[1] - ns[1]) * self._image_h)
            size_text = f"{pw} x {ph}"
            font = QFont()
            font.setPixelSize(11)
            painter.setFont(font)
            bg = QColor(PALETTE["panel_raised"])
            bg.setAlpha(200)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(size_text) + 8
            th = fm.height() + 4
            label_x = x + w / 2 - tw / 2
            label_y = y + h + 4
            painter.fillRect(QRectF(label_x, label_y, tw, th), bg)
            painter.setPen(QColor(PALETTE["text"]))
            painter.drawText(QRectF(label_x, label_y, tw, th), Qt.AlignCenter, size_text)

    def _paint_polygon_preview(self, painter: QPainter) -> None:
        """Paint the polygon being drawn: placed vertices + rubber-band edges."""
        pts_px = [self.norm_to_pixel(vx, vy) for vx, vy in self._polygon_points]

        # Edges between placed vertices
        painter.setPen(QPen(QColor(PALETTE["primary"]), 2))
        painter.setBrush(Qt.NoBrush)
        for i in range(len(pts_px) - 1):
            ax, ay = pts_px[i]
            bx, by = pts_px[i + 1]
            painter.drawLine(QPointF(ax, ay), QPointF(bx, by))

        # Rubber-band edges to the cursor: last vertex → cursor, and cursor →
        # first vertex (the closing hint) when ≥2 vertices are down.
        if self._draw_current is not None:
            cx, cy = self.norm_to_pixel(*self._draw_current)
            lx, ly = pts_px[-1]
            painter.setPen(QPen(QColor(PALETTE["primary"]), 1, Qt.DashLine))
            painter.drawLine(QPointF(lx, ly), QPointF(cx, cy))
            if len(pts_px) >= 2:
                fx, fy = pts_px[0]
                painter.setPen(QPen(QColor(PALETTE["teal"]), 1, Qt.DotLine))
                painter.drawLine(QPointF(cx, cy), QPointF(fx, fy))

        # Vertex dots (first vertex emphasized as the closing target)
        painter.setPen(QPen(QColor(PALETTE["text"]), 1))
        for i, (vx, vy) in enumerate(pts_px):
            if i == 0:
                painter.setBrush(QBrush(QColor(PALETTE["teal"])))
                painter.drawEllipse(QPointF(vx, vy), HANDLE_SIZE, HANDLE_SIZE)
            else:
                painter.setBrush(QBrush(QColor(PALETTE["primary"])))
                painter.drawEllipse(QPointF(vx, vy), HANDLE_SIZE - 1, HANDLE_SIZE - 1)

    # ── Mouse events ───────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        px, py = event.x(), event.y()

        # Middle button, or Ctrl + Left button → pan
        ctrl_left = (
            event.button() == Qt.LeftButton
            and event.modifiers() & Qt.ControlModifier
        )
        if event.button() == Qt.MiddleButton or ctrl_left:
            self._panning = True
            self._pan_start = (px, py)
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() != Qt.LeftButton:
            return

        if self.tool_mode in ("draw_bbox", "sam_prompt"):
            # sam_prompt shares the rubber-band gesture: a click becomes a
            # point prompt, a drag becomes a box prompt (decided on release).
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._drawing = True
            self._draw_start = (nx, ny)
            self._draw_current = (nx, ny)

        elif self.tool_mode == "draw_keypoint":
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._draw_start = (nx, ny)
            # Don't emit class_requested here — do it on mouse release
            # to avoid popup appearing while mouse button is still pressed

        elif self.tool_mode == "draw_polygon":
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            # First vertex also seeds _draw_start so resizeEvent's refit guard
            # and clear_draw_state semantics apply during the polygon session.
            if not self._polygon_points:
                self._draw_start = (nx, ny)
            self._polygon_points.append([nx, ny])
            self._draw_current = (nx, ny)
            self.update()

        elif self.tool_mode == "select":
            # Check if clicking a handle first (for selected bbox)
            handle = self._hit_test_handle(px, py)
            if handle:
                self._dragging = True
                self._drag_type = handle
                self._drag_ann_id = self._selected_id
                self._drag_start_norm = self.pixel_to_norm(px, py)
                ann = self.get_selected_annotation()
                if ann:
                    self._drag_ann_snapshot = ann.to_dict()
                return

            # Check if clicking a polygon vertex to drag (selected polygon only)
            vtx_hit = self._hit_test_polygon_vertex(px, py)
            if vtx_hit:
                ann_id, vtx_idx = vtx_hit
                self._dragging = True
                self._drag_type = "move_poly_vertex"
                self._drag_ann_id = ann_id
                self._drag_vertex_idx = vtx_idx
                self._drag_start_norm = self.pixel_to_norm(px, py)
                ann = self.get_selected_annotation()
                if ann:
                    self._drag_ann_snapshot = ann.to_dict()
                return

            # Check if clicking a keypoint to drag
            kp_hit = self._hit_test_keypoint(px, py)
            if kp_hit:
                ann_id, kp_idx = kp_hit
                self._dragging = True
                self._drag_type = "move_kp"
                self._drag_ann_id = ann_id
                self._drag_kp_idx = kp_idx
                self._drag_start_norm = self.pixel_to_norm(px, py)
                return

            # Hit test annotations
            hit_id = self.hit_test(px, py)
            if hit_id:
                self.select_annotation(hit_id)
                self._dragging = True
                self._drag_type = "move"
                self._drag_ann_id = hit_id
                self._drag_start_norm = self.pixel_to_norm(px, py)
                ann = self.get_selected_annotation()
                if ann:
                    self._drag_ann_snapshot = ann.to_dict()
            else:
                self.select_annotation(None)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        px, py = event.x(), event.y()

        # Track the cursor for the live crosshair. Always recorded (even
        # when an overlay drag is active) so the moment the user releases
        # the mouse, the crosshair is back on the cursor — not lagging one
        # event behind. The renderer only draws it when the cursor is
        # inside the image rect.
        self._mouse_pos_widget = (px, py)
        self.update()  # crosshair is the cheapest possible repaint payload

        if self._panning and self._pan_start:
            dx = px - self._pan_start[0]
            dy = py - self._pan_start[1]
            self._offset_x += dx
            self._offset_y += dy
            self._pan_start = (px, py)
            self.update()
            return

        if self._drawing and self._draw_start:
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._draw_current = (nx, ny)
            self.update()
            return

        # Polygon rubber-band: follow the cursor between vertex clicks.
        if self.tool_mode == "draw_polygon" and self._polygon_points:
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._draw_current = (nx, ny)
            self.update()
            return

        if self._dragging and self._drag_ann_id:
            nx, ny = self.pixel_to_norm(px, py)
            self._handle_drag(nx, ny)
            self.update()
            return

        # Cursor feedback in select mode
        if self.tool_mode == "select" and self._image is not None:
            if event.modifiers() & Qt.ControlModifier:
                self.setCursor(Qt.OpenHandCursor)
                return
            handle = self._hit_test_handle(px, py)
            if handle:
                if "tl" in handle or "br" in handle:
                    self.setCursor(Qt.SizeFDiagCursor)
                else:
                    self.setCursor(Qt.SizeBDiagCursor)
            elif self._hit_test_polygon_vertex(px, py):
                self.setCursor(Qt.SizeAllCursor)
            elif self._hit_test_keypoint(px, py):
                self.setCursor(Qt.SizeAllCursor)
            elif self.hit_test(px, py):
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        elif self._image is not None:
            # Draw modes: Ctrl signals pan availability, otherwise crosshair
            if event.modifiers() & Qt.ControlModifier:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.CrossCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        px, py = event.x(), event.y()

        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor if self.tool_mode == "select" else Qt.CrossCursor)
            return

        if event.button() != Qt.LeftButton:
            return

        if self._drawing and self._draw_start and self.tool_mode == "sam_prompt":
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            sx, sy = self._draw_start
            self._drawing = False
            self._draw_start = None
            self._draw_current = None
            self.update()
            # Same 1% minimum as bbox drawing separates click from drag.
            if abs(nx - sx) > 0.01 and abs(ny - sy) > 0.01:
                self.sam_box_requested.emit(
                    min(sx, nx), min(sy, ny), max(sx, nx), max(sy, ny),
                )
            else:
                self.sam_point_requested.emit(sx, sy)
            return

        if self._drawing and self._draw_start and self.tool_mode == "draw_bbox":
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._draw_current = (nx, ny)
            sx, sy = self._draw_start
            w = abs(nx - sx)
            h = abs(ny - sy)
            if w > 0.01 and h > 0.01:
                self.class_requested.emit(px, py)
            else:
                self._draw_start = None
                self._draw_current = None
            self._drawing = False
            self.update()
            return

        if self.tool_mode == "draw_keypoint" and self._draw_start:
            # Check if inside an existing bbox — attach to it
            hit_id = self.hit_test(px, py)
            if hit_id:
                ann = next((a for a in self._annotations if a.id == hit_id), None)
                if ann and ann.bbox:
                    self.keypoint_attach_requested.emit(hit_id, px, py)
                    return
            # Outside any bbox — standalone keypoint (show class picker)
            self.class_requested.emit(px, py)
            return

        if self._dragging:
            if self._drag_type in (
                "move", "resize_tl", "resize_tr", "resize_bl", "resize_br",
                "move_kp", "move_poly_vertex",
            ):
                self.annotation_modified.emit(self._drag_ann_id)
            self._dragging = False
            self._drag_type = ""
            self._drag_ann_id = None
            self._drag_kp_idx = -1
            self._drag_vertex_idx = -1
            self._drag_ann_snapshot = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double-click closes an in-progress polygon.

        Qt delivers double-click as press → release → dblClick → release, so
        the first press already appended a vertex. Drop the duplicate final
        vertex left by that press before closing.
        """
        if (
            event.button() == Qt.LeftButton
            and self.tool_mode == "draw_polygon"
            and self._polygon_points
        ):
            self._dedupe_last_polygon_vertex()
            self.finish_polygon(event.x(), event.y())
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event) -> None:
        """Cursor left the canvas — drop the crosshair so it doesn't stick.

        Without this, the last known cursor position keeps rendering the
        crosshair + coord badge forever (until a new event arrives).
        """
        self._mouse_pos_widget = None
        self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.ControlModifier:
            factor = ZOOM_FACTOR if event.angleDelta().y() > 0 else 1.0 / ZOOM_FACTOR
            self._apply_zoom(event.x(), event.y(), factor)

    def _apply_zoom(self, center_px: float, center_py: float, factor: float) -> None:
        """Apply zoom by factor around a pixel center point."""
        old_nx, old_ny = self.pixel_to_norm(center_px, center_py)
        self._scale = max(MIN_SCALE, min(self._scale * factor, MAX_SCALE))
        new_px = old_nx * self._image_w * self._scale + self._offset_x
        new_py = old_ny * self._image_h * self._scale + self._offset_y
        self._offset_x += center_px - new_px
        self._offset_y += center_py - new_py
        self.zoom_changed.emit(self._scale)
        self.update()

    def zoom_in(self) -> None:
        """Zoom in by one step."""
        if self._image is None:
            return
        self._apply_zoom(self.width() / 2, self.height() / 2, ZOOM_FACTOR)

    def zoom_out(self) -> None:
        """Zoom out by one step."""
        if self._image is None:
            return
        self._apply_zoom(self.width() / 2, self.height() / 2, 1.0 / ZOOM_FACTOR)

    def zoom_fit(self) -> None:
        """Reset zoom to fit image in window."""
        if self._image is None:
            return
        self._fit_to_window()
        self.zoom_changed.emit(self._scale)
        self.update()

    def contextMenuEvent(self, event) -> None:
        """Show right-click context menu."""
        px, py = event.x(), event.y()

        # Check if right-clicking a keypoint specifically
        kp_hit = self._hit_test_keypoint(px, py)

        hit_id = self.hit_test(px, py)
        if not hit_id:
            return

        self.select_annotation(hit_id)
        ann = self.get_selected_annotation()
        if not ann:
            return

        menu = QMenu(self)

        # Keypoint-specific actions (when right-clicking directly on a keypoint)
        if kp_hit and kp_hit[0] == hit_id:
            kp_idx = kp_hit[1]
            kp = ann.keypoints[kp_idx] if kp_idx < len(ann.keypoints) else None
            if kp:
                vis_names = ["不可见", "被遮挡", "可见"]
                vis_label = vis_names[kp.visible] if kp.visible < 3 else "?"
                kp_header = menu.addAction(f"关键点: {kp.label} ({vis_label})")
                kp_header.setEnabled(False)

                rename_kp = menu.addAction("重命名关键点")
                rename_kp.triggered.connect(
                    lambda _, aid=ann.id, ki=kp_idx: self._request_rename_keypoint(aid, ki))

                cycle_vis = menu.addAction("切换可见性")
                cycle_vis.triggered.connect(
                    lambda _, aid=ann.id, ki=kp_idx: self.cycle_keypoint_visibility(aid, ki))

                del_kp = menu.addAction("删除关键点")
                del_kp.triggered.connect(
                    lambda _, aid=ann.id, ki=kp_idx: self.remove_keypoint(aid, ki))

                menu.addSeparator()

        # Polygon vertex / edge actions (annotation already selected above, so
        # the vertex/edge hit-tests target it).
        if ann.polygon:
            vtx_hit = self._hit_test_polygon_vertex(px, py)
            if vtx_hit and vtx_hit[0] == hit_id:
                vtx_idx = vtx_hit[1]
                poly_header = menu.addAction(f"多边形顶点 {vtx_idx}")
                poly_header.setEnabled(False)
                del_vtx = menu.addAction("删除顶点")
                del_vtx.setEnabled(len(ann.polygon) > 3)
                del_vtx.triggered.connect(
                    lambda _, aid=ann.id, vi=vtx_idx: self.remove_polygon_vertex(aid, vi))
                menu.addSeparator()
            else:
                edge_idx = self._hit_test_polygon_edge(px, py)
                if edge_idx is not None:
                    nx, ny = self.pixel_to_norm(px, py)
                    ins_vtx = menu.addAction("在此处插入顶点")
                    ins_vtx.triggered.connect(
                        lambda _, aid=ann.id, ei=edge_idx, ix=nx, iy=ny:
                        self.insert_polygon_vertex(aid, ei, ix, iy))
                    menu.addSeparator()

        # Conflict resolution options
        paired_id = self._conflict_pairs.get(ann.id)
        if paired_id:
            paired_ann = next((a for a in self._annotations if a.id == paired_id), None)
            if paired_ann:
                # Determine which is existing (confirmed) and which is prediction
                if ann.confirmed:
                    existing_ann, pred_ann = ann, paired_ann
                else:
                    existing_ann, pred_ann = paired_ann, ann
                keep_existing = menu.addAction(
                    f"保留确认框 (conf={existing_ann.confidence:.2f})")
                keep_existing.triggered.connect(
                    lambda: self.resolve_conflict(existing_ann.id))
                keep_pred = menu.addAction(
                    f"保留预测框 (conf={pred_ann.confidence:.2f})")
                keep_pred.triggered.connect(
                    lambda: self.resolve_conflict(pred_ann.id))
                menu.addSeparator()

        # Modify class
        change_cls = menu.addAction("修改类别")
        change_cls.triggered.connect(lambda: self.class_change_requested.emit(ann.id, px, py))

        if ann.confirmed:
            unconfirm = menu.addAction("取消确认")
            unconfirm.triggered.connect(lambda: self._toggle_confirm(ann, False))
        else:
            confirm = menu.addAction("确认")
            confirm.triggered.connect(lambda: self._toggle_confirm(ann, True))

        menu.addSeparator()

        copy_ann = menu.addAction("复制标注 (Ctrl+C)")
        copy_ann.triggered.connect(lambda: self.annotation_copied.emit(ann.id))

        delete = menu.addAction("删除")
        delete.triggered.connect(lambda: self.annotation_deleted.emit(ann.id))

        menu.exec_(event.globalPos())

    def resizeEvent(self, event: QResizeEvent) -> None:
        if self._image and not self._draw_start and not self._polygon_points:
            self._fit_to_window()
        super().resizeEvent(event)

    def keyPressEvent(self, event) -> None:
        """Let key events propagate to parent (LabelPanel) for handling."""
        event.ignore()

    # ── Drag helpers ───────────────────────────────────────────

    def _handle_drag(self, nx: float, ny: float) -> None:
        """Handle ongoing drag operation."""
        if not self._drag_ann_id or not self._drag_start_norm:
            return

        ann = None
        for a in self._annotations:
            if a.id == self._drag_ann_id:
                ann = a
                break
        if ann is None:
            return

        dx = nx - self._drag_start_norm[0]
        dy = ny - self._drag_start_norm[1]

        if self._drag_type == "move" and ann.bbox and self._drag_ann_snapshot:
            if ann.polygon and self._drag_ann_snapshot.get("polygon"):
                # Polygon move: translate every vertex from the snapshot,
                # clamping the delta so the whole polygon stays in bounds, then
                # rederive the bbox (single source of truth: polygon → bbox).
                orig_poly = self._drag_ann_snapshot["polygon"]
                xs = [p[0] for p in orig_poly]
                ys = [p[1] for p in orig_poly]
                cdx = max(-min(xs), min(1.0 - max(xs), dx))
                cdy = max(-min(ys), min(1.0 - max(ys), dy))
                ann.polygon = [[p[0] + cdx, p[1] + cdy] for p in orig_poly]
                ann.bbox = bbox_from_polygon(ann.polygon)
                if not ann.confirmed:
                    ann.confirmed = True
                self.annotations_changed.emit()
                return
            orig_bbox = self._drag_ann_snapshot["bbox"]
            new_cx = orig_bbox[0] + dx
            new_cy = orig_bbox[1] + dy
            w, h = orig_bbox[2], orig_bbox[3]
            new_cx = max(w / 2, min(1.0 - w / 2, new_cx))
            new_cy = max(h / 2, min(1.0 - h / 2, new_cy))
            ann.bbox = (new_cx, new_cy, w, h)
            # Move keypoints by same offset
            if "keypoints" in self._drag_ann_snapshot:
                for i, kp_dict in enumerate(self._drag_ann_snapshot["keypoints"]):
                    if i < len(ann.keypoints):
                        ann.keypoints[i].x = max(0, min(1, kp_dict["x"] + dx))
                        ann.keypoints[i].y = max(0, min(1, kp_dict["y"] + dy))
            if not ann.confirmed:
                ann.confirmed = True
            self.annotations_changed.emit()

        elif self._drag_type == "move_poly_vertex" and ann.polygon:
            if 0 <= self._drag_vertex_idx < len(ann.polygon):
                cx, cy = self._clamp_norm(nx, ny)
                ann.polygon[self._drag_vertex_idx] = [cx, cy]
                ann.bbox = bbox_from_polygon(ann.polygon)
                if not ann.confirmed:
                    ann.confirmed = True
                self.annotations_changed.emit()

        elif self._drag_type == "move_kp":
            if 0 <= self._drag_kp_idx < len(ann.keypoints):
                ann.keypoints[self._drag_kp_idx].x = max(0.0, min(1.0, nx))
                ann.keypoints[self._drag_kp_idx].y = max(0.0, min(1.0, ny))
                if not ann.confirmed:
                    ann.confirmed = True
                self.annotations_changed.emit()

        elif self._drag_type.startswith("resize_") and ann.bbox and self._drag_ann_snapshot:
            orig_bbox = self._drag_ann_snapshot["bbox"]
            ocx, ocy, ow, oh = orig_bbox
            ox1, oy1 = ocx - ow / 2, ocy - oh / 2
            ox2, oy2 = ocx + ow / 2, ocy + oh / 2

            if "tl" in self._drag_type:
                ox1 = max(0.0, min(ox2 - 0.01, ox1 + dx))
                oy1 = max(0.0, min(oy2 - 0.01, oy1 + dy))
            elif "tr" in self._drag_type:
                ox2 = max(ox1 + 0.01, min(1.0, ox2 + dx))
                oy1 = max(0.0, min(oy2 - 0.01, oy1 + dy))
            elif "bl" in self._drag_type:
                ox1 = max(0.0, min(ox2 - 0.01, ox1 + dx))
                oy2 = max(oy1 + 0.01, min(1.0, oy2 + dy))
            elif "br" in self._drag_type:
                ox2 = max(ox1 + 0.01, min(1.0, ox2 + dx))
                oy2 = max(oy1 + 0.01, min(1.0, oy2 + dy))

            ann.bbox = ((ox1 + ox2) / 2, (oy1 + oy2) / 2, ox2 - ox1, oy2 - oy1)
            if not ann.confirmed:
                ann.confirmed = True
            self.annotations_changed.emit()

    def _hit_test_handle(self, px: float, py: float) -> str | None:
        """Check if pixel pos hits a resize handle on the selected bbox."""
        if not self._selected_id:
            return None
        ann = self.get_selected_annotation()
        if not ann or not ann.bbox:
            return None
        # A polygon's bbox is derived — it exposes vertex handles, not corner
        # resize handles (resizing the box would desync the polygon).
        if ann.polygon:
            return None

        cx, cy, w, h = ann.bbox
        corners = {
            "resize_tl": (cx - w / 2, cy - h / 2),
            "resize_tr": (cx + w / 2, cy - h / 2),
            "resize_bl": (cx - w / 2, cy + h / 2),
            "resize_br": (cx + w / 2, cy + h / 2),
        }
        for handle_name, (nx, ny) in corners.items():
            hpx, hpy = self.norm_to_pixel(nx, ny)
            if abs(px - hpx) <= HANDLE_SIZE + 2 and abs(py - hpy) <= HANDLE_SIZE + 2:
                return handle_name
        return None

    def _hit_test_polygon_vertex(self, px: float, py: float) -> tuple[str, int] | None:
        """Check if pixel pos hits a vertex of the selected polygon.

        Only the selected polygon exposes vertex handles (mirrors the
        bbox-handle rule: interaction targets belong to the active selection).
        Returns (ann_id, vertex_index) or None.
        """
        if not self._selected_id:
            return None
        ann = self.get_selected_annotation()
        if not ann or not ann.polygon:
            return None
        nx, ny = self.pixel_to_norm(px, py)
        tol = (HANDLE_SIZE + 2) / (self._image_w * self._scale) if self._image_w * self._scale > 0 else 0.0
        idx = nearest_vertex(nx, ny, ann.polygon, tol)
        if idx is None:
            return None
        return ann.id, idx

    def _hit_test_polygon_edge(self, px: float, py: float) -> int | None:
        """Return the edge index of the selected polygon under the cursor, else None.

        The new vertex for edge ``i`` is inserted at position ``i + 1``.
        """
        if not self._selected_id:
            return None
        ann = self.get_selected_annotation()
        if not ann or not ann.polygon:
            return None
        nx, ny = self.pixel_to_norm(px, py)
        tol = (HANDLE_SIZE + 4) / (self._image_w * self._scale) if self._image_w * self._scale > 0 else 0.0
        return nearest_edge(nx, ny, ann.polygon, tol)

    def _hit_test_keypoint(self, px: float, py: float) -> tuple[str, int] | None:
        """Check if pixel pos hits a keypoint. Returns (ann_id, kp_index) or None."""
        for ann in reversed(self._annotations):
            for i, kp in enumerate(ann.keypoints):
                kpx, kpy = self.norm_to_pixel(kp.x, kp.y)
                if abs(px - kpx) <= KEYPOINT_RADIUS + 4 and abs(py - kpy) <= KEYPOINT_RADIUS + 4:
                    return ann.id, i
        return None

    def _toggle_confirm(self, ann: Annotation, confirmed: bool) -> None:
        ann.confirmed = confirmed
        self.annotations_changed.emit()
        self.update()

    def _request_rename_keypoint(self, ann_id: str, kp_idx: int) -> None:
        """Show input dialog to rename a keypoint."""
        ann = next((a for a in self._annotations if a.id == ann_id), None)
        if not ann or kp_idx >= len(ann.keypoints):
            return
        old_label = ann.keypoints[kp_idx].label
        new_label, ok = QInputDialog.getText(
            self, "重命名关键点", "标签:", text=old_label)
        if ok and new_label.strip():
            self.rename_keypoint(ann_id, kp_idx, new_label.strip())

    # ── Public helpers for external bbox/kp creation ───────────

    def create_bbox_from_draw(self, class_name: str, class_id: int) -> Annotation | None:
        """Create a bbox annotation from the last draw operation."""
        if not self._draw_start or not self._draw_current:
            return None
        sx, sy = self._draw_start
        ex, ey = self._draw_current
        w = abs(ex - sx)
        h = abs(ey - sy)
        if w < 0.01 or h < 0.01:
            return None
        cx = (sx + ex) / 2
        cy = (sy + ey) / 2
        ann = Annotation(
            class_name=class_name,
            class_id=class_id,
            bbox=(cx, cy, w, h),
            confirmed=True,
            source="manual",
        )
        self._annotations.append(ann)
        self.select_annotation(ann.id)
        self.annotation_created.emit(ann)
        self.annotations_changed.emit()
        self._draw_start = None
        self._draw_current = None
        self.update()
        return ann

    def create_keypoint_at(
        self, class_name: str, class_id: int, label: str = "point"
    ) -> Annotation | None:
        """Create a keypoint annotation at the stored draw position."""
        if not self._draw_start:
            return None
        nx, ny = self._draw_start
        kp = Keypoint(x=nx, y=ny, visible=2, label=label)
        ann = Annotation(
            class_name=class_name,
            class_id=class_id,
            keypoints=[kp],
            confirmed=True,
            source="manual",
        )
        self._annotations.append(ann)
        self.select_annotation(ann.id)
        self.annotation_created.emit(ann)
        self.annotations_changed.emit()
        self._draw_start = None
        self.update()
        return ann

    # ── Polygon (segment) drawing + editing ────────────────────

    @property
    def has_polygon_draft(self) -> bool:
        """True while a draw_polygon session has at least one placed vertex."""
        return bool(self._polygon_points)

    def _dedupe_last_polygon_vertex(self) -> None:
        """Drop a trailing near-duplicate vertex.

        A double-click's first press appends a vertex before
        ``mouseDoubleClickEvent`` fires; when the user double-clicks on their
        last placed vertex that appended point duplicates it, so remove it.
        """
        pts = self._polygon_points
        if len(pts) < 2:
            return
        ax, ay = pts[-1]
        bx, by = pts[-2]
        tol_x = HANDLE_SIZE / (self._image_w * self._scale) if self._image_w * self._scale > 0 else 0.0
        tol_y = HANDLE_SIZE / (self._image_h * self._scale) if self._image_h * self._scale > 0 else 0.0
        if abs(ax - bx) <= tol_x and abs(ay - by) <= tol_y:
            pts.pop()

    def finish_polygon(self, px: float | None = None, py: float | None = None) -> bool:
        """Close the in-progress polygon and request a class if it has ≥3 vertices.

        Returns True when a class picker was requested (the draft is kept so
        ``create_polygon_from_draw`` can read the vertices after selection);
        False when there aren't enough vertices yet (draft kept, keep drawing).
        ``px``/``py`` anchor the class picker; default to the last vertex.
        """
        if self.tool_mode != "draw_polygon" or len(self._polygon_points) < 3:
            return False
        if px is None or py is None:
            lx, ly = self._polygon_points[-1]
            px, py = self.norm_to_pixel(lx, ly)
        self.class_requested.emit(px, py)
        return True

    def begin_polygon_from_points(self, points: list[list[float]]) -> bool:
        """Stage an externally-computed polygon (e.g. a SAM mask) as the draft
        and request a class.

        The vertices become the pending draft consumed by
        ``create_polygon_from_draw`` after the class picker — the exact reuse
        path of a hand-drawn polygon close. Returns False (draft untouched)
        for degenerate input (<3 vertices).
        """
        if not points or len(points) < 3:
            return False
        self._polygon_points = [[float(x), float(y)] for x, y in points]
        px, py = self.norm_to_pixel(*self._polygon_points[0])
        self.class_requested.emit(px, py)
        self.update()
        return True

    def create_polygon_from_draw(
        self,
        class_name: str,
        class_id: int,
        source: str = "manual",
        confirmed: bool = True,
    ) -> Annotation | None:
        """Create a polygon annotation (+ derived bbox) from the placed vertices.

        ``source``/``confirmed`` let assisted flows (SAM) mark the result as
        an unconfirmed auto annotation while hand-drawn polygons keep the
        manual/confirmed defaults.
        """
        if len(self._polygon_points) < 3:
            self._polygon_points = []
            self._draw_start = None
            self._draw_current = None
            return None
        poly = [[float(x), float(y)] for x, y in self._polygon_points]
        ann = Annotation(
            class_name=class_name,
            class_id=class_id,
            bbox=bbox_from_polygon(poly),
            polygon=poly,
            confirmed=confirmed,
            source=source,
        )
        self._annotations.append(ann)
        self.select_annotation(ann.id)
        self.annotation_created.emit(ann)
        self.annotations_changed.emit()
        self._polygon_points = []
        self._draw_start = None
        self._draw_current = None
        self.update()
        return ann

    def insert_polygon_vertex(self, ann_id: str, edge_idx: int, x: float, y: float) -> None:
        """Insert a vertex after ``edge_idx`` (between edge_idx and the next).

        Recomputes the derived bbox and auto-confirms (any manual edit
        confirms, matching bbox/keypoint editing).
        """
        for ann in self._annotations:
            if ann.id == ann_id and ann.polygon:
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                ann.polygon.insert(edge_idx + 1, [x, y])
                ann.bbox = bbox_from_polygon(ann.polygon)
                if not ann.confirmed:
                    ann.confirmed = True
                self.annotation_modified.emit(ann_id)
                self.annotations_changed.emit()
                self.update()
                return

    def remove_polygon_vertex(self, ann_id: str, vtx_idx: int) -> None:
        """Remove a polygon vertex; refuses to drop below a triangle (3 pts)."""
        for ann in self._annotations:
            if ann.id == ann_id and ann.polygon:
                if len(ann.polygon) <= 3:
                    return
                if 0 <= vtx_idx < len(ann.polygon):
                    ann.polygon.pop(vtx_idx)
                    ann.bbox = bbox_from_polygon(ann.polygon)
                    if not ann.confirmed:
                        ann.confirmed = True
                    self.annotation_modified.emit(ann_id)
                    self.annotations_changed.emit()
                    self.update()
                return

    def select_keypoint(self, ann_id: str, kp_idx: int) -> None:
        """Select a specific keypoint within an annotation."""
        self._selected_id = ann_id
        self._selected_kp_idx = kp_idx
        self.annotation_selected.emit(ann_id)
        self.keypoint_selected.emit(ann_id, kp_idx)
        self.update()

    def add_keypoint_to_annotation(self, ann_id: str, kp: Keypoint) -> None:
        """Append a keypoint to an existing annotation."""
        for ann in self._annotations:
            if ann.id == ann_id:
                ann.keypoints.append(kp)
                if not ann.confirmed:
                    ann.confirmed = True
                self.annotation_modified.emit(ann_id)
                self.annotations_changed.emit()
                self.update()
                return

    def remove_keypoint(self, ann_id: str, kp_idx: int) -> None:
        """Remove a single keypoint from an annotation.

        If the annotation has no bbox and this is the last keypoint, remove the annotation.
        """
        for ann in self._annotations:
            if ann.id == ann_id:
                if 0 <= kp_idx < len(ann.keypoints):
                    ann.keypoints.pop(kp_idx)
                    if not ann.bbox and not ann.keypoints:
                        self._annotations = [a for a in self._annotations if a.id != ann_id]
                        self.annotation_deleted.emit(ann_id)
                    else:
                        self.annotation_modified.emit(ann_id)
                    if self._selected_kp_idx is not None and self._selected_kp_idx >= len(ann.keypoints):
                        self._selected_kp_idx = None
                    self.annotations_changed.emit()
                    self.update()
                return

    def rename_keypoint(self, ann_id: str, kp_idx: int, new_label: str) -> None:
        """Rename a keypoint's label."""
        for ann in self._annotations:
            if ann.id == ann_id:
                if 0 <= kp_idx < len(ann.keypoints):
                    ann.keypoints[kp_idx].label = new_label
                    self.annotation_modified.emit(ann_id)
                    self.annotations_changed.emit()
                    self.update()
                return

    def cycle_keypoint_visibility(self, ann_id: str, kp_idx: int) -> None:
        """Cycle keypoint visibility: 0 → 1 → 2 → 0."""
        for ann in self._annotations:
            if ann.id == ann_id:
                if 0 <= kp_idx < len(ann.keypoints):
                    ann.keypoints[kp_idx].visible = (ann.keypoints[kp_idx].visible + 1) % 3
                    self.annotation_modified.emit(ann_id)
                    self.annotations_changed.emit()
                    self.update()
                return
