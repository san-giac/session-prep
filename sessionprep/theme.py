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

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette

UI_FONT_FAMILY = ""
UI_FONT_SIZE = 11
ROW_HEIGHTS = {"compact": 18, "normal": 22, "roomy": 28}
DEFAULT_DENSITY = "normal"

THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "window": "#30302E",
        "base": "#30302E",
        "alternate_base": "#30302E",
        "text": "#FEFBF6",
        "button": "#30302E",
        "highlight": "#FEFBF6",
        "highlight_text": "#30302E",

        "muted": "#6E6E69",
        "warning": "#F15A24",
        "tag": "#F15A24",
        "mono": "#F15A24",
        "override": "#F7B900",

        "loudest": "#F15A24",
        "quietest": "#4FD18B",

        "wave_bg": "#30302E",
        "wave_grid": "#FEFBF6",
        "wave": "#FEFBF6",
        "wave_alt": "#FEFBF6",
        "wave_side": "#FFC4EB",
        "playhead": "#FEFBF6",
        "peak": "#F15A24",
        "wave_clip": "#F15A24",
    },
    "light": {
        "window": "#FEFBF6",
        "base": "#FEFBF6",
        "alternate_base": "#FEFBF6",
        "text": "#30302E",
        "button": "#FEFBF6",
        "highlight": "#30302E",
        "highlight_text": "#FEFBF6",

        "muted": "#6E6E69",
        "warning": "#F15A24",
        "tag": "#F15A24",
        "mono": "#F15A24",
        "override": "#F7B900",

        "loudest": "#F15A24",
        "quietest": "#1E7A4C",

        "wave_bg": "#FEFBF6",
        "wave_grid": "#30302E",
        "wave": "#30302E",
        "wave_alt": "#30302E",
        "wave_side": "#FFC4EB",
        "playhead": "#30302E",
        "peak": "#F15A24",
        "wave_clip": "#F15A24",
    },
}

MODES = ("auto", "dark", "light")
DEFAULT_MODE = "auto"

_active = "dark"
_issued: dict[str, QColor] = {}


def colour(key: str) -> QColor:
    existing = _issued.get(key)
    if existing is None:
        existing = QColor(THEMES[_active][key])
        _issued[key] = existing
    return existing


def active() -> str:
    return _active


def resolve(mode: str, app=None) -> str:
    if mode in ("dark", "light"):
        return mode
    if app is not None:
        try:
            scheme = app.styleHints().colorScheme()
            if scheme == Qt.ColorScheme.Light:
                return "light"
            if scheme == Qt.ColorScheme.Dark:
                return "dark"
        except (AttributeError, TypeError):
            pass
    return "dark"


def apply(app, mode: str) -> str:
    global _active
    _active = resolve(mode, app)
    values = THEMES[_active]

    for key, existing in _issued.items():
        existing.setRgb(*QColor(values[key]).getRgb())

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(values["window"]))
    palette.setColor(QPalette.WindowText, QColor(values["text"]))
    palette.setColor(QPalette.Base, QColor(values["base"]))
    palette.setColor(QPalette.AlternateBase, QColor(values["alternate_base"]))
    palette.setColor(QPalette.ToolTipBase, QColor(values["window"]))
    palette.setColor(QPalette.ToolTipText, QColor(values["text"]))
    palette.setColor(QPalette.Text, QColor(values["text"]))
    palette.setColor(QPalette.Button, QColor(values["button"]))
    palette.setColor(QPalette.ButtonText, QColor(values["text"]))
    palette.setColor(QPalette.BrightText, QColor(values["highlight_text"]))
    palette.setColor(QPalette.Highlight, QColor(values["highlight"]))
    palette.setColor(QPalette.HighlightedText, QColor(values["highlight_text"]))
    palette.setColor(QPalette.PlaceholderText, QColor(values["muted"]))

    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        palette.setColor(QPalette.Disabled, role, QColor(values["muted"]))

    app.setPalette(palette)

    font = QFont(UI_FONT_FAMILY) if UI_FONT_FAMILY else QFont()
    font.setPointSize(UI_FONT_SIZE)
    app.setFont(font)
    return _active


def next_mode(mode: str) -> str:
    order = ("auto", "light", "dark")
    try:
        return order[(order.index(mode) + 1) % len(order)]
    except ValueError:
        return "auto"


def next_density(density: str) -> str:
    order = ("compact", "normal", "roomy")
    try:
        return order[(order.index(density) + 1) % len(order)]
    except ValueError:
        return DEFAULT_DENSITY


def row_height(density: str) -> int:
    return ROW_HEIGHTS.get(density, ROW_HEIGHTS[DEFAULT_DENSITY])
