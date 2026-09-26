"""Tests for src/ui/train_form_binding.py — no Qt, stub duck-typed form."""
import logging

from src.ui.train_form_binding import (
    BOOL_FIELD_MAP,
    COMBO_FIELD_MAP,
    NUMERIC_FIELD_MAP,
    apply_template_params,
)


class _StubSpin:
    def __init__(self):
        self.value = None

    def setValue(self, v):
        self.value = v


class _StubCheck:
    def __init__(self):
        self.checked = None

    def setChecked(self, v):
        self.checked = v


class _StubCombo:
    """Editable-combo stub: findText mimics a fixed item list."""

    def __init__(self, items=None):
        self._items = list(items or [])
        self.text = None
        self.index = None

    def setCurrentText(self, text):
        self.text = text

    def findText(self, text):
        return self._items.index(text) if text in self._items else -1

    def setCurrentIndex(self, idx):
        self.index = idx
        if 0 <= idx < len(self._items):
            self.text = self._items[idx]


class _StubForm:
    """Duck-typed stand-in for TrainPanel — only the attributes the binding
    touches, no QApplication needed."""

    def __init__(self):
        # Numeric + bool + combo widgets, keyed by attribute name in the maps.
        for attr in NUMERIC_FIELD_MAP.values():
            setattr(self, attr, _StubSpin())
        for attr in BOOL_FIELD_MAP.values():
            setattr(self, attr, _StubCheck())
        self._optimizer_combo = _StubCombo(["auto", "SGD", "Adam", "AdamW"])
        self._device_combo = _StubCombo(["auto", "0", "cpu"])
        self._auto_augment_combo = _StubCombo(
            ["randaugment", "augmix", "autoaugment", "none"]
        )
        self._model_combo = _StubCombo()
        self._kpt_num_spin = _StubSpin()
        self._kpt_dim_spin = _StubSpin()
        self.freeze_calls = []

    def _set_freeze_value(self, value):
        self.freeze_calls.append(value)


def test_numeric_fields_mapped():
    form = _StubForm()
    apply_template_params(form, {"epochs": 200, "lr0": 0.005})
    assert form._epochs_spin.value == 200
    assert form._lr0_spin.value == 0.005


def test_missing_keys_leave_form_alone():
    form = _StubForm()
    apply_template_params(form, {"epochs": 50})
    assert form._epochs_spin.value == 50
    assert form._batch_spin.value is None  # untouched


def test_bool_fields_coerced():
    form = _StubForm()
    apply_template_params(form, {"include_detect_params": 1})
    assert form._include_detect_params_check.checked is True


def test_combo_field_mapped():
    form = _StubForm()
    apply_template_params(form, {"optimizer": "AdamW"})
    assert form._optimizer_combo.text == "AdamW"


def test_device_empty_string_becomes_auto():
    form = _StubForm()
    apply_template_params(form, {"device": ""})
    assert form._device_combo.text == "auto"


def test_device_none_becomes_auto():
    form = _StubForm()
    apply_template_params(form, {"device": None})
    assert form._device_combo.text == "auto"


def test_device_explicit_value_passthrough():
    form = _StubForm()
    apply_template_params(form, {"device": "0"})
    assert form._device_combo.text == "0"


def test_freeze_delegates_to_hook_int():
    form = _StubForm()
    apply_template_params(form, {"freeze": 7})
    assert form.freeze_calls == [7]


def test_freeze_delegates_to_hook_none():
    form = _StubForm()
    apply_template_params(form, {"freeze": None})
    assert form.freeze_calls == [None]


def test_auto_augment_string_selects_index():
    form = _StubForm()
    apply_template_params(form, {"auto_augment": "augmix"})
    assert form._auto_augment_combo.text == "augmix"


def test_auto_augment_falsy_becomes_none():
    form = _StubForm()
    apply_template_params(form, {"auto_augment": ""})
    assert form._auto_augment_combo.text == "none"


def test_auto_augment_unknown_value_ignored():
    form = _StubForm()
    apply_template_params(form, {"auto_augment": "does-not-exist"})
    # findText returns -1 → index untouched, text stays None
    assert form._auto_augment_combo.text is None


def test_model_arbitrary_string():
    form = _StubForm()
    apply_template_params(form, {"model": "yolov8s.pt"})
    assert form._model_combo.text == "yolov8s.pt"


def test_kpt_shape_two_values():
    form = _StubForm()
    apply_template_params(form, {"kpt_shape": [21, 2]})
    assert form._kpt_num_spin.value == 21
    assert form._kpt_dim_spin.value == 2


def test_kpt_shape_invalid_shape_ignored():
    form = _StubForm()
    apply_template_params(form, {"kpt_shape": [21]})
    assert form._kpt_num_spin.value is None
    assert form._kpt_dim_spin.value is None


def test_unknown_key_logs_warning(caplog):
    form = _StubForm()
    with caplog.at_level(logging.WARNING):
        apply_template_params(form, {"epochs": 10, "totally_made_up_key": 999})
    assert form._epochs_spin.value == 10
    assert any("totally_made_up_key" in rec.getMessage() for rec in caplog.records)
