#!/usr/bin/env python3
# Session Prep — audio toolkit.
# Copyright (C) 2026 Sandro Giacometti
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import struct
import sys
import zlib
from pathlib import Path

APP_NAME = "Session Prep"
BUNDLE_ID = "local.sg.audioconverter"
VERSION = "1.0.0"
MIN_MACOS = "11.0"

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "sessionprep"
REQUIRED = ["PySide6", "numpy", "soundfile", "soxr", "scipy", "yaml"]

BG = (24, 26, 30, 255)
WAVE = (104, 176, 216, 255)
ACCENT = (96, 168, 120, 255)

def _png(pixels) -> bytes:
    height, width, _ = pixels.shape
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        raw += row.tobytes()

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b""))


def _render(size: int, np):
    ss = 4
    n = size * ss
    img = np.zeros((n, n, 4), dtype=np.float64)

    inset = n * 0.06
    radius = n * 0.22
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float64) + 0.5
    lo, hi = inset, n - inset
    dx = np.maximum(np.maximum(lo + radius - xs, xs - (hi - radius)), 0.0)
    dy = np.maximum(np.maximum(lo + radius - ys, ys - (hi - radius)), 0.0)
    inside = (np.hypot(dx, dy) <= radius) & (xs >= lo) & (xs <= hi) & (ys >= lo) & (ys <= hi)
    img[inside] = np.array(BG) / 255.0

    rng = np.random.default_rng(7)
    bars = 9
    span = hi - lo
    bar_w = span / (bars * 1.75)
    gap = span / bars
    envelope = np.array([0.34, 0.62, 0.92, 0.55, 1.00, 0.70, 0.88, 0.48, 0.30])
    envelope = envelope * (0.85 + 0.15 * rng.random(bars))

    def draw_bars(centre_y: float, height: float, colour, amps) -> None:
        rgba = np.array(colour) / 255.0
        for i, amp in enumerate(amps):
            cx = lo + gap * (i + 0.5)
            half = max(height * amp * 0.5, n * 0.006)
            x0, x1 = cx - bar_w / 2, cx + bar_w / 2
            y0, y1 = centre_y - half, centre_y + half
            mask = ((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1) & inside)
            img[mask] = rgba

    draw_bars(lo + span * 0.38, span * 0.52, WAVE, envelope)
    draw_bars(lo + span * 0.82, span * 0.15, ACCENT, envelope * 0.6)

    out = img.reshape(size, ss, size, ss, 4).mean(axis=(1, 3))
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


ICON_VARIANTS = [(b"ic11", 32), (b"ic12", 64), (b"ic07", 128),
                 (b"ic08", 256), (b"ic09", 512), (b"ic10", 1024)]


def icns_from_png(source: Path, destination: Path) -> str:
    if not shutil.which("sips") or not shutil.which("iconutil"):
        return ("sips and iconutil weren't found — those come with macOS, so "
                "this looks like another platform. Supply icon.icns instead.")

    iconset = destination.parent / "icon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)
    try:
        for points in (16, 32, 128, 256, 512):
            for scale, suffix in ((1, ""), (2, "@2x")):
                pixels = points * scale
                target = iconset / f"icon_{points}x{points}{suffix}.png"
                result = subprocess.run(
                    ["sips", "-z", str(pixels), str(pixels),
                     str(source), "--out", str(target)],
                    capture_output=True, text=True)
                if result.returncode != 0:
                    return f"sips could not resize {source.name}: {result.stderr.strip()}"

        result = subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(destination)],
            capture_output=True, text=True)
        if result.returncode != 0:
            return f"iconutil failed: {result.stderr.strip()}"
    finally:
        shutil.rmtree(iconset, ignore_errors=True)
    return ""


def build_icns(path: Path) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False

    variants = [(b"ic11", 32), (b"ic12", 64), (b"ic07", 128),
                (b"ic08", 256), (b"ic09", 512), (b"ic10", 1024)]
    entries = bytearray()
    for ostype, size in variants:
        payload = _png(_render(size, np))
        entries += ostype + struct.pack(">I", len(payload) + 8) + payload

    path.write_bytes(b"icns" + struct.pack(">I", len(entries) + 8) + bytes(entries))
    return True


LAUNCHER = r'''#!/bin/bash
set -uo pipefail

BUNDLE="$(cd "$(dirname "$0")/../.." && pwd)"
PARENT="$(dirname "$BUNDLE")"
BAKED_PROJECT="__PROJECT__"
BAKED_PYTHON="__PYTHON__"

if [ -f "$PARENT/sessionprep/app.py" ]; then
    PROJECT="$PARENT"
elif [ -f "$BAKED_PROJECT/sessionprep/app.py" ]; then
    PROJECT="$BAKED_PROJECT"
else
    osascript -e 'display dialog "Can'"'"'t find the sessionprep folder.\n\nThe app expects it either beside this bundle or at:\n__PROJECT__\n\nRebuild with: python3 make_app.py" with title "__APPNAME__" buttons {"OK"} default button 1 with icon stop' >/dev/null 2>&1
    exit 1
fi

CANDIDATES=(
    "$PROJECT/.venv/bin/python3"
    "$PROJECT/venv/bin/python3"
    "$BAKED_PYTHON"
    "/opt/homebrew/bin/python3"
    "/usr/local/bin/python3"
    "/usr/bin/python3"
)

PYTHON=""
for candidate in "${CANDIDATES[@]}"; do
    [ -x "$candidate" ] || continue
    if "$candidate" -c 'import PySide6, numpy, soundfile, soxr, scipy, yaml' >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
    [ -n "$FIRST_PY" ] || FIRST_PY="$candidate"
done

if [ -z "$PYTHON" ]; then
    osascript -e 'display dialog "The dependencies are not installed.\n\nOpen Terminal and run:\n\n    cd __PROJECT__\n    python3 -m venv .venv\n    source .venv/bin/activate\n    pip install -r requirements.txt\n\nThen try again." with title "__APPNAME__" buttons {"OK"} default button 1 with icon caution' >/dev/null 2>&1
    exit 1
fi

cd "$PROJECT" || exit 1
exec "$PYTHON" -m sessionprep
'''


def build_launcher(project: Path) -> str:
    text = LAUNCHER.replace("__PROJECT__", str(project))
    text = text.replace("__PYTHON__", sys.executable)
    text = text.replace("__APPNAME__", APP_NAME)
    return text.replace('PYTHON=""\nfor candidate', 'PYTHON=""\nFIRST_PY=""\nfor candidate')

def main() -> int:
    if not (PACKAGE / "app.py").exists():
        print(f"error: no sessionprep package next to {Path(__file__).name}",
              file=sys.stderr)
        print(f"       looked in {HERE}", file=sys.stderr)
        return 1

    bundle = HERE / f"{APP_NAME}.app"
    contents = bundle / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"

    if bundle.exists():
        shutil.rmtree(bundle)
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)

    info = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleVersion": VERSION,
        "CFBundleShortVersionString": VERSION,
        "CFBundleExecutable": "launcher",
        "CFBundlePackageType": "APPL",
        "CFBundleSignature": "????",
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSMinimumSystemVersion": MIN_MACOS,
        "NSHighResolutionCapable": True,
        "CFBundleIconFile": "appicon",
    }

    icon = resources / "appicon.icns"
    custom_icns = HERE / "icon.icns"
    custom_png = HERE / "icon.png"

    if custom_icns.exists():
        shutil.copy2(custom_icns, icon)
        icon_note = f"from {custom_icns.name}"
    elif custom_png.exists():
        problem = icns_from_png(custom_png, icon)
        icon_note = f"from {custom_png.name}" if not problem else f"FAILED: {problem}"
        if problem:
            print(f"\nCould not use {custom_png.name}: {problem}\n")
            build_icns(icon)
            icon_note = "generated (fell back)"
    elif build_icns(icon):
        icon_note = "generated — drop an icon.png here to use your own"
    else:
        info.pop("CFBundleIconFile")
        icon_note = "skipped (numpy not importable by this interpreter)"

    with open(contents / "Info.plist", "wb") as handle:
        plistlib.dump(info, handle)
    (contents / "PkgInfo").write_text("APPL????")

    launcher = macos / "launcher"
    launcher.write_text(build_launcher(HERE))
    launcher.chmod(0o755)

    print(f"Built {bundle}")
    print(f"  project   {HERE}")
    print(f"  python    {sys.executable}")
    print(f"  icon      {icon_note}")

    missing = []
    for module in REQUIRED:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        venv = HERE / ".venv" / "bin" / "python3"
        if venv.exists():
            print(f"\nNote: {', '.join(missing)} missing from the interpreter that "
                  f"built this,\n      but .venv exists and the launcher prefers it.")
        else:
            print(f"\nWarning: {', '.join(missing)} not importable and no .venv found.")
            print("         The app will show a setup dialog until you run:")
            print(f"           cd {HERE}")
            print("           python3 -m venv .venv && source .venv/bin/activate")
            print("           pip install -r requirements.txt")

    print("\nDouble-click it, or drag it to the Dock. Keep it in this folder;")
    print("for the Dock or /Applications use an alias rather than moving it.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(0)
