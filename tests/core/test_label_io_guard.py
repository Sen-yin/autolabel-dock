"""Guard: session code must not bypass LabelStore for label record IO.

The "flush before reading a label record" invariant is owned by
``src/core/label_store.py``. Any *new* direct use of
``label_io.load_annotation`` / ``label_io.save_annotation`` in ``src/``
reintroduces the class of bug the store exists to kill (8d769b4, the
tag-manager cascade), so referencing those names outside the allowlist
fails this test.

Allowlisted by design (Qt-free consumers that run strictly after an
explicit flush, plus the store/IO modules themselves):
- src/core/label_store.py   (the store delegates to label_io)
- src/core/label_io.py      (the definitions)
- src/engine/dataset.py     (training prep; runs after app.py's explicit flush)
- src/core/formats/*        (import/export codecs)
"""
import re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

ALLOWLIST = {
    "core/label_store.py",
    "core/label_io.py",
    "engine/dataset.py",
}
ALLOWLIST_DIRS = ("core/formats/",)

_PATTERN = re.compile(r"\b(load_annotation|save_annotation)\b")


def _is_allowlisted(rel: str) -> bool:
    return rel in ALLOWLIST or rel.startswith(ALLOWLIST_DIRS)


def test_no_direct_label_io_outside_allowlist():
    assert SRC_ROOT.is_dir(), f"src root not found: {SRC_ROOT}"
    offenders: list[str] = []
    for py in sorted(SRC_ROOT.rglob("*.py")):
        rel = py.relative_to(SRC_ROOT).as_posix()
        if _is_allowlisted(rel):
            continue
        text = py.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _PATTERN.search(line):
                offenders.append(f"src/{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Direct label_io usage outside the LabelStore allowlist "
        "(route it through LabelStore instead — reads must flush first):\n"
        + "\n".join(offenders)
    )


def test_allowlist_entries_still_exist():
    """Keep the allowlist honest — stale entries hide real regressions."""
    for rel in ALLOWLIST:
        assert (SRC_ROOT / rel).is_file(), f"stale allowlist entry: src/{rel}"
    for d in ALLOWLIST_DIRS:
        assert (SRC_ROOT / d).is_dir(), f"stale allowlist dir: src/{d}"
