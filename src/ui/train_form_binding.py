"""Template-parameter → form-widget binding for TrainPanel. No Qt imports.

The field maps translate ``TrainConfig`` field names into widget attribute
names on the training form, which makes them a UI concern — but the module
itself is duck-typed (``getattr`` on the form object) and imports no Qt, so
the mapping and its special cases stay unit-testable with a plain stub form
(no QApplication needed).

Special-cased fields (``freeze``, ``auto_augment``, ``model``, ``kpt_shape``)
are handled inline in :func:`apply_template_params`; ``freeze`` routes through
the form's ``_set_freeze_value`` hook because it toggles the paired
"use default" checkbox.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Mapping from TrainConfig field names to spinbox/checkbox attribute names on the form.
NUMERIC_FIELD_MAP: dict[str, str] = {
    "epochs": "_epochs_spin",
    "batch": "_batch_spin",
    "imgsz": "_imgsz_spin",
    "workers": "_workers_spin",
    "patience": "_patience_spin",
    "lr0": "_lr0_spin",
    "lrf": "_lrf_spin",
    "momentum": "_momentum_spin",
    "weight_decay": "_weight_decay_spin",
    "warmup_epochs": "_warmup_epochs_spin",
    "warmup_momentum": "_warmup_momentum_spin",
    "warmup_bias_lr": "_warmup_bias_lr_spin",
    "hsv_h": "_hsv_h_spin",
    "hsv_s": "_hsv_s_spin",
    "hsv_v": "_hsv_v_spin",
    "degrees": "_degrees_spin",
    "translate": "_translate_spin",
    "scale": "_scale_spin",
    "shear": "_shear_spin",
    "perspective": "_perspective_spin",
    "flipud": "_flipud_spin",
    "fliplr": "_fliplr_spin",
    "mosaic": "_mosaic_spin",
    "mixup": "_mixup_spin",
    "copy_paste": "_copy_paste_spin",
    "erasing": "_erasing_spin",
    "dropout": "_dropout_spin",
    "pose": "_pose_weight_spin",
    "kobj": "_kobj_spin",
}

BOOL_FIELD_MAP: dict[str, str] = {
    "include_detect_params": "_include_detect_params_check",
    "include_classify_params": "_include_classify_params_check",
    "include_pose_params": "_include_pose_params_check",
}

COMBO_FIELD_MAP: dict[str, str] = {
    "optimizer": "_optimizer_combo",
    "device": "_device_combo",
}


def apply_template_params(form, params: dict) -> None:
    """Set spin/check/combo values on ``form`` for keys present in ``params``.

    Missing keys leave the form alone; unknown keys are skipped with a
    warning log. ``form`` is duck-typed: any object exposing the widget
    attributes named in the maps (plus ``_auto_augment_combo``,
    ``_model_combo``, ``_kpt_num_spin``/``_kpt_dim_spin`` and the
    ``_set_freeze_value`` hook) works — ``TrainPanel`` in production, a stub
    in tests.
    """
    for key, value in params.items():
        if key in NUMERIC_FIELD_MAP:
            getattr(form, NUMERIC_FIELD_MAP[key]).setValue(value)
        elif key in BOOL_FIELD_MAP:
            getattr(form, BOOL_FIELD_MAP[key]).setChecked(bool(value))
        elif key in COMBO_FIELD_MAP:
            combo = getattr(form, COMBO_FIELD_MAP[key])
            text = str(value) if value is not None else ""
            if key == "device" and text == "":
                text = "auto"
            combo.setCurrentText(text)
        elif key == "freeze":
            form._set_freeze_value(value)
        elif key == "auto_augment":
            text = value if value else "none"
            idx = form._auto_augment_combo.findText(text)
            if idx >= 0:
                form._auto_augment_combo.setCurrentIndex(idx)
        elif key == "model":
            # Editable combo — accept arbitrary string
            form._model_combo.setCurrentText(str(value))
        elif key == "kpt_shape":
            if isinstance(value, (list, tuple)) and len(value) == 2:
                form._kpt_num_spin.setValue(int(value[0]))
                form._kpt_dim_spin.setValue(int(value[1]))
        else:
            logger.warning("Skipping unknown template key: %s", key)
