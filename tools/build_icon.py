"""Generate the application icon (.ico) from resources/icons/app_icon.png.

Windows reads .ico for the taskbar, window titlebar and the .exe resource.
PyInstaller's ``--icon`` flag points at a .ico (or .icns on macOS). This script
embeds the PNG at multiple resolutions so Windows picks the right bitmap per-DPI
instead of stretching a single 256x256.

Why hand-rolled instead of ``Pillow.Image.save(..., format='ICO',
append_images=...)``:
    Pillow 12's ICO writer silently drops everything but the first frame when
    ``append_images`` is used (verified on 12.2.0). The format itself is trivial
    (6-byte header + 16-byte directory + raw PNG/BMP blobs), so we write the
    container ourselves. Modern Windows reads PNG-in-ICO entries natively since
    Vista, so we encode each frame as PNG and avoid the alpha-channel
    limitations of BMP-encoded entries.

Run from repo root:
    python tools/build_icon.py

Inputs / outputs:
    in:  resources/icons/app_icon.png  (committed PNG, RGBA, square recommended)
    out: resources/icons/app.ico       (16/24/32/48/64/128/256 PNG-in-ICO)
"""
from __future__ import annotations

import io
import struct
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PNG = REPO_ROOT / "resources" / "icons" / "app_icon.png"
OUT_ICO = REPO_ROOT / "resources" / "icons" / "app.ico"

# Sizes Windows actually uses. 16 is required for the title bar; 32 covers
# taskbar at 100% DPI; 48/64 for medium DPI; 128/256 for shortcuts and the
# Alt-Tab thumbnail at high DPI.
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _encode_frame_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    # optimize=True keeps the small-size entries (16/24) under a kilobyte.
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _build_ico(frames: list[Image.Image]) -> bytes:
    blobs = [_encode_frame_png(frame) for frame in frames]
    # ICONDIR: reserved(2)=0, type(2)=1 (icon), count(2)=N
    header = struct.pack("<HHH", 0, 1, len(frames))
    entries = bytearray()
    body = bytearray()
    data_offset = 6 + 16 * len(frames)
    for frame, blob in zip(frames, blobs):
        w, h = frame.size
        # ICONDIRENTRY width/height: 0 means 256 (the field is 1 byte).
        entries += struct.pack(
            "<BBBBHHII",
            w if w < 256 else 0,
            h if h < 256 else 0,
            0,            # palette colors (0 for RGBA)
            0,            # reserved
            1,            # color planes
            32,           # bits per pixel
            len(blob),
            data_offset + len(body),
        )
        body += blob
    return bytes(header + entries + body)


def main() -> int:
    if not SRC_PNG.exists():
        raise SystemExit(
            f"missing source icon: {SRC_PNG}\n"
            "Drop the application icon PNG at resources/icons/app_icon.png and re-run."
        )

    src = Image.open(SRC_PNG)
    # ICO PNG entries must be RGBA (no indexed palette support).
    if src.mode != "RGBA":
        src = src.convert("RGBA")

    # LANCZOS keeps edges crisp on the smaller sizes (16/24) which is what
    # Windows uses most often for the title bar.
    frames = [
        src.resize((size, size), Image.Resampling.LANCZOS) for size in ICO_SIZES
    ]

    OUT_ICO.parent.mkdir(parents=True, exist_ok=True)
    OUT_ICO.write_bytes(_build_ico(frames))

    # Sanity check: re-read the file and confirm every requested size is
    # actually embedded (a 16-only file would silently degrade the title bar).
    with Image.open(OUT_ICO) as check:
        embedded = sorted(check.info.get("sizes", set()))
    expected = sorted({(s, s) for s in ICO_SIZES})
    if embedded != expected:
        raise SystemExit(
            f"ico size set mismatch: wrote {embedded}, expected {expected}"
        )

    print(f"wrote {OUT_ICO.relative_to(REPO_ROOT)} ({OUT_ICO.stat().st_size} bytes)")
    print(f"embedded sizes: {embedded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
