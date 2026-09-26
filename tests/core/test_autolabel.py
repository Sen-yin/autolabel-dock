"""Tests for src/core/autolabel.py — pure logic, no Qt / no qapp."""
from __future__ import annotations

import dataclasses

import pytest

from src.core.annotation import Annotation, find_conflicts
from src.core.autolabel import (
    STATUS_SKIPPED_CONFIRMED,
    STATUS_UNRECOGNIZED,
    ClassifyApplyDecision,
    InferenceParams,
    MergeOutcome,
    decide_classify_apply,
    merge_predictions,
)


def _ann(cls="cat", bbox=(0.5, 0.5, 0.2, 0.2), confirmed=True, cid=0):
    return Annotation(
        class_name=cls, class_id=cid, bbox=bbox,
        confirmed=confirmed, source="manual",
    )


def _pred(cls="cat", bbox=(0.5, 0.5, 0.2, 0.2), cid=0):
    return Annotation(
        class_name=cls, class_id=cid, bbox=bbox,
        confidence=0.9, confirmed=False, source="auto",
    )


class TestInferenceParams:
    def test_defaults_match_historical_fallbacks(self):
        p = InferenceParams()
        assert p.conf == 0.5
        assert p.iou == 0.45
        assert p.overlap_iou == 0.5
        assert p.class_match_mode == "class_id"

    def test_frozen(self):
        p = InferenceParams()
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.conf = 0.9


class TestMergePredictions:
    def test_empty_existing_accepts_all(self):
        preds = [_pred(), _pred(bbox=(0.2, 0.2, 0.1, 0.1))]
        outcome = merge_predictions([], preds)
        assert outcome.accepted == preds
        assert outcome.conflict_pairs == []
        assert outcome.conflict_predictions == []

    def test_overlapping_confirmed_same_class_is_conflict(self):
        existing = [_ann(confirmed=True)]
        pred = _pred(bbox=(0.5, 0.5, 0.2, 0.2))  # identical box → IoU 1.0
        outcome = merge_predictions(existing, [pred], 0.5)
        assert outcome.accepted == []
        assert outcome.conflict_pairs == [(existing[0], pred)]
        assert outcome.conflict_predictions == [pred]

    def test_unconfirmed_existing_never_conflicts(self):
        existing = [_ann(confirmed=False)]
        pred = _pred()
        outcome = merge_predictions(existing, [pred], 0.5)
        assert outcome.accepted == [pred]
        assert outcome.conflict_pairs == []

    def test_different_class_never_conflicts(self):
        existing = [_ann(cls="dog", cid=1)]
        pred = _pred(cls="cat", cid=0)
        outcome = merge_predictions(existing, [pred], 0.5)
        assert outcome.accepted == [pred]
        assert outcome.conflict_pairs == []

    def test_below_threshold_is_accepted(self):
        existing = [_ann(bbox=(0.3, 0.3, 0.2, 0.2))]
        pred = _pred(bbox=(0.7, 0.7, 0.2, 0.2))  # disjoint
        outcome = merge_predictions(existing, [pred], 0.5)
        assert outcome.accepted == [pred]

    def test_matches_find_conflicts_exactly(self):
        """merge_predictions is a thin partition over the canonical matcher —
        outputs must be bit-identical to a direct find_conflicts call."""
        existing = [
            _ann(bbox=(0.5, 0.5, 0.2, 0.2), confirmed=True),
            _ann(cls="dog", cid=1, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=True),
            _ann(bbox=(0.8, 0.8, 0.1, 0.1), confirmed=False),
        ]
        preds = [
            _pred(bbox=(0.5, 0.5, 0.2, 0.2)),          # conflicts with #0
            _pred(cls="dog", cid=1, bbox=(0.9, 0.1, 0.05, 0.05)),  # clean
            _pred(bbox=(0.8, 0.8, 0.1, 0.1)),           # existing unconfirmed → clean
        ]
        conflicts, clean = find_conflicts(existing, preds, 0.5)
        outcome = merge_predictions(existing, preds, 0.5)
        assert outcome.accepted == clean
        assert outcome.conflict_pairs == conflicts

    def test_default_outcome_lists_are_independent(self):
        a, b = MergeOutcome(), MergeOutcome()
        a.accepted.append(_pred())
        assert b.accepted == []


class TestDecideClassifyApply:
    CLASSES = ["cat", "dog"]

    def test_registered_applies_rebuilds_and_notes_new_class(self):
        d = decide_classify_apply("registered", "bird", "bird", 0.87, self.CLASSES)
        assert d == ClassifyApplyDecision(
            apply=True,
            class_name="bird",
            rebuild_classes=True,
            status_text="自动标注: bird (0.87) (新增类别 'bird')",
        )

    def test_existing_applies_without_rebuild(self):
        d = decide_classify_apply("existing", "cat", "cat", 0.5, self.CLASSES)
        assert d.apply is True
        assert d.class_name == "cat"
        assert d.rebuild_classes is False
        assert d.status_text == "自动标注: cat (0.50)"

    def test_rejected_disabled_but_class_in_project_applies_raw_name(self):
        d = decide_classify_apply("rejected_disabled", None, "dog", 0.75, self.CLASSES)
        assert d.apply is True
        assert d.class_name == "dog"
        assert d.rebuild_classes is False
        assert d.status_text == "自动标注: dog (0.75)"

    def test_rejected_disabled_unknown_class_skips(self):
        d = decide_classify_apply("rejected_disabled", None, "bird", 0.75, self.CLASSES)
        assert d.apply is False
        assert d.class_name is None
        assert d.rebuild_classes is False
        assert d.status_text == (
            "自动标注: 类别 'bird' 不在项目中，已跳过（未开启自动登记）"
        )

    def test_rejected_blacklist_skips_with_imagenet_hint(self):
        d = decide_classify_apply(
            "rejected_blacklist", None, "n01440764", 0.9, self.CLASSES,
        )
        assert d.apply is False
        assert d.status_text == (
            "自动标注: 模型类名 'n01440764' 不可用（疑似 ImageNet ID），已跳过"
        )

    def test_rejected_invalid_skips_with_generic_text(self):
        d = decide_classify_apply("rejected_invalid", None, "   ", 0.9, self.CLASSES)
        assert d.apply is False
        assert d.status_text == "自动标注: 模型类名无效，已跳过"

    def test_status_constants(self):
        assert STATUS_SKIPPED_CONFIRMED == "自动标注: 已存在确认标签，跳过"
        assert STATUS_UNRECOGNIZED == "自动标注: 未识别"


def test_module_is_qt_free():
    """core/autolabel must never grow a Qt/UI dependency."""
    import sys

    import src.core.autolabel as mod

    src_text = open(mod.__file__, encoding="utf-8").read()
    assert "PyQt5" not in src_text
    assert "src.ui" not in src_text
    # And importing it must not pull Qt in a fresh interpreter — approximate
    # here by checking the module's own imports don't reference QtWidgets.
    assert not any(
        name.startswith("PyQt5") for name in getattr(mod, "__dict__", {})
    )
    assert "src.core.autolabel" in sys.modules
