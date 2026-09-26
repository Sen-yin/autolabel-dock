"""AutoLabel core — pure decision logic for the auto-label pipeline.

Qt-free by contract (no UI imports). Two concerns live here:

- ``merge_predictions``: the single implementation point for merging incoming
  predictions into an image's existing annotations. It wraps the canonical
  ``find_conflicts`` matcher (per CLAUDE.md, overlap checks must route through
  it — never re-implement IoU matching). Callers choose the *surface* for the
  conflict pairs: the interactive detect/pose view adds them and highlights
  them red; the batch path drops them and counts.
- ``decide_classify_apply``: the pure decision table for applying a single
  classify prediction after ``ProjectController.register_auto_class`` ran.
  The Chinese status-bar texts moved here verbatim from ``app.py`` so the
  message logic has exactly one home.

``InferenceParams`` carries the model-panel thresholds across the
controller boundary; its defaults mirror the historical fallbacks used when
no ModelPanel exists yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.core.annotation import Annotation, find_conflicts

# Status text used when the classify view refuses to overwrite an existing
# confirmed tag (``add_auto_class_prediction`` returned False). Kept here so
# every classify status string lives in one module.
STATUS_SKIPPED_CONFIRMED = "自动标注: 已存在确认标签，跳过"
# Status text for a classify prediction the model could not produce.
STATUS_UNRECOGNIZED = "自动标注: 未识别"


@dataclass(frozen=True)
class InferenceParams:
    """Inference thresholds snapshot (defaults = historical no-panel fallbacks)."""

    conf: float = 0.5
    iou: float = 0.45
    overlap_iou: float = 0.5
    class_match_mode: str = "class_id"


@dataclass
class MergeOutcome:
    """Partition of incoming predictions against existing annotations.

    ``accepted``: predictions with no conflict — safe to add/save as-is.
    ``conflict_pairs``: ``(existing confirmed annotation, prediction)`` pairs
    matched by ``find_conflicts``. Interactive surfaces highlight them;
    bulk surfaces drop them and count.
    """

    accepted: list[Annotation] = field(default_factory=list)
    conflict_pairs: list[tuple[Annotation, Annotation]] = field(default_factory=list)

    @property
    def conflict_predictions(self) -> list[Annotation]:
        """Just the prediction half of each conflict pair."""
        return [pred for _, pred in self.conflict_pairs]


def merge_predictions(
    existing: list[Annotation],
    predictions: list[Annotation],
    iou_threshold: float = 0.5,
) -> MergeOutcome:
    """Single entry point for prediction merging (wraps ``find_conflicts``).

    A prediction conflicts when it overlaps a *confirmed same-class* existing
    annotation at ``iou >= iou_threshold`` (greedy, one prediction per
    existing annotation — see ``find_conflicts``). The caller decides what to
    do with ``conflict_pairs`` (highlight vs drop+count).
    """
    conflicts, clean = find_conflicts(existing, predictions, iou_threshold)
    return MergeOutcome(accepted=clean, conflict_pairs=conflicts)


@dataclass
class ClassifyApplyDecision:
    """What to do with a single classify prediction after registration.

    ``apply``: call the view's ``add_auto_class_prediction`` with
    ``class_name``. ``rebuild_classes``: a brand-new class was registered —
    the class widgets must be rebuilt (Supersede reload, see CONTEXT.md).
    ``status_text``: the Chinese status-bar message for the happy path of
    this decision (the view may still veto the apply — then the caller shows
    ``STATUS_SKIPPED_CONFIRMED`` instead).
    """

    apply: bool
    class_name: str | None
    rebuild_classes: bool
    status_text: str


def decide_classify_apply(
    reg_action: str,
    applied_name: str | None,
    raw_name: str,
    conf: float,
    project_classes: list[str],
) -> ClassifyApplyDecision:
    """Decision table for the single-image classify flow.

    ``reg_action`` / ``applied_name`` come from
    ``ProjectController.register_auto_class(raw_name)``; ``project_classes``
    is the live class list (used by the ``rejected_disabled`` branch to allow
    predictions whose class already exists). Status texts moved verbatim from
    the pre-controller ``app.py`` implementation.
    """
    if reg_action in ("registered", "existing"):
        suffix = (
            f" (新增类别 '{applied_name}')" if reg_action == "registered" else ""
        )
        return ClassifyApplyDecision(
            apply=True,
            class_name=applied_name,
            rebuild_classes=(reg_action == "registered"),
            status_text=f"自动标注: {applied_name} ({conf:.2f}){suffix}",
        )
    if reg_action == "rejected_disabled":
        if raw_name in project_classes:
            return ClassifyApplyDecision(
                apply=True,
                class_name=raw_name,
                rebuild_classes=False,
                status_text=f"自动标注: {raw_name} ({conf:.2f})",
            )
        return ClassifyApplyDecision(
            apply=False,
            class_name=None,
            rebuild_classes=False,
            status_text=(
                f"自动标注: 类别 '{raw_name}' 不在项目中，已跳过（未开启自动登记）"
            ),
        )
    if reg_action == "rejected_blacklist":
        return ClassifyApplyDecision(
            apply=False,
            class_name=None,
            rebuild_classes=False,
            status_text=(
                f"自动标注: 模型类名 '{raw_name}' 不可用（疑似 ImageNet ID），已跳过"
            ),
        )
    # rejected_invalid (and any unknown action) — skip with the generic text.
    return ClassifyApplyDecision(
        apply=False,
        class_name=None,
        rebuild_classes=False,
        status_text="自动标注: 模型类名无效，已跳过",
    )
