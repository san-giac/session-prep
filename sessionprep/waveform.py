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

import math

import numpy as np
from PySide6.QtCore import QRect, QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import theme
from .analysis import rebucket

BG = theme.colour("wave_bg")
GRID = theme.colour("wave_grid")
WAVE = theme.colour("wave")
WAVE_ALT = theme.colour("wave_alt")
SIDE = theme.colour("wave_side")
PLAYHEAD = theme.colour("playhead")
PEAK = theme.colour("peak")
CLIP = theme.colour("wave_clip")
TEXT = theme.colour("muted")

TEXT_BAND = 20
CLIP_CEILING = 0.999
CLIP_MARK = 3

ZOOM_FIT = 0.0
ZOOM_CHOICES = [(ZOOM_FIT, "Fit"), (1.0, "1x"), (2.0, "2x"),
                (4.0, "4x"), (8.0, "8x"), (16.0, "16x")]
FIT_MAX_GAIN = 64.0


class WaveformView(QWidget):
    seeked = Signal(int)
    scrubbed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAutoFillBackground(False)

        self._envelope = None
        self._provisional = False
        self._samplerate = 44100
        self._frames = 0
        self._show_side = False
        self._position = 0
        self._label = "No file selected"
        self._pixmap: QPixmap | None = None
        self._pixmap_size = (0, 0)
        self._pixmap_ratio = 0.0
        self._peak_frame = 0
        self._peak_db = -math.inf
        self._zoom = ZOOM_FIT
        self._clipped = False

    def set_envelope(self, envelope, label: str = "") -> None:
        self._envelope = envelope
        self._provisional = bool(envelope is not None and envelope.provisional)
        self._clipped = _has_clipping(envelope)
        if envelope is None:
            self._samplerate, self._frames = 44100, 0
            self._peak_frame, self._peak_db = 0, -math.inf
        else:
            self._samplerate = max(1, envelope.samplerate)
            self._frames = envelope.frames
            self._peak_frame = envelope.peak_frame
            magnitude = envelope.peak_magnitude
            self._peak_db = (20 * math.log10(magnitude) if magnitude > 0
                             else -math.inf)
        self._position = 0
        self._label = label or "No file selected"
        self._invalidate()

    def refresh_theme(self) -> None:
        self._invalidate()

    def set_zoom(self, zoom: float) -> None:
        if zoom != self._zoom:
            self._zoom = zoom
            self._invalidate()

    @property
    def scale(self) -> float:
        if self._zoom != ZOOM_FIT:
            return self._zoom
        magnitude = 10 ** (self._peak_db / 20) if self._peak_db > -math.inf else 0.0
        if magnitude <= 0:
            return 1.0
        return min(1.0 / magnitude, FIT_MAX_GAIN)

    def set_show_side(self, show: bool) -> None:
        self._show_side = show
        self._invalidate()

    def set_position(self, frame: int) -> None:
        if frame == self._position:
            return
        old = self._playhead_x(self._position)
        self._position = frame
        new = self._playhead_x(frame)
        lo, hi = (old, new) if old <= new else (new, old)
        self.update(QRect(int(lo) - 2, 0, int(hi - lo) + 5, self.height()))
        self.update(QRect(0, 0, self.width(), TEXT_BAND))

    def _invalidate(self) -> None:
        self._pixmap = None
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if (self.width(), self.height()) != self._pixmap_size:
            self._invalidate()

    def _build_pixmap(self) -> None:
        ratio = self.devicePixelRatioF()
        width, height = max(1, self.width()), max(1, self.height())
        pixmap = QPixmap(int(width * ratio), int(height * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(BG)
        self._pixmap_size = (width, height)
        self._pixmap_ratio = ratio

        if self._envelope is None or not self._frames:
            self._pixmap = pixmap
            return

        mins, maxs = rebucket(self._envelope.mins, self._envelope.maxs, width)
        painter = QPainter(pixmap)
        channels = mins.shape[0]
        lane_top = TEXT_BAND
        lane_area = max(1.0, height - TEXT_BAND)
        lane_h = lane_area / channels
        peak_x = self._playhead_x(self._peak_frame)

        for ch in range(channels):
            top = lane_top + ch * lane_h
            lane = QRectF(0, top, width, lane_h)
            self._draw_lane(painter, lane, mins[ch], maxs[ch],
                            WAVE if ch == 0 else WAVE_ALT)
            if ch:
                painter.setPen(QPen(GRID, 1))
                painter.drawLine(0, int(top), width, int(top))

        if self._show_side and self._envelope.channels >= 2:
            side_min, side_max = rebucket(self._envelope.side_min[None, :],
                                          self._envelope.side_max[None, :], width)
            self._draw_lane(painter, QRectF(0, lane_top, width, lane_area),
                            side_min[0], side_max[0], SIDE, mark_clipping=False)

        magnitude = 10 ** (self._peak_db / 20) if self._peak_db > -math.inf else 0.0
        magnitude = min(magnitude * self.scale, 1.0)
        if magnitude > 0 and not self._provisional:
            painter.setPen(QPen(PEAK, 1, Qt.DotLine))
            for ch in range(channels):
                mid = lane_top + ch * lane_h + lane_h / 2
                half = lane_h / 2 * 0.9 * magnitude
                painter.drawLine(0, int(mid - half), width, int(mid - half))
                painter.drawLine(0, int(mid + half), width, int(mid + half))
            painter.setPen(QPen(PEAK, 1, Qt.DashLine))
            painter.drawLine(int(peak_x), lane_top, int(peak_x), height)
        painter.end()
        self._pixmap = pixmap

    def _draw_lane(self, painter, lane: QRectF, mins, maxs, colour,
                   mark_clipping: bool = True):
        mid = lane.top() + lane.height() / 2
        half = lane.height() / 2 * 0.9
        painter.setPen(QPen(GRID, 1))
        painter.drawLine(int(lane.left()), int(mid), int(lane.right()), int(mid))

        scale = self.scale
        top = mid - np.clip(maxs.astype(np.float64) * scale, -1.0, 1.0) * half
        bottom = mid - np.clip(mins.astype(np.float64) * scale, -1.0, 1.0) * half
        bottom = np.where(bottom - top < 1.0, top + 1.0, bottom)

        count = min(len(mins), int(lane.width()))
        clipped = (np.zeros(count, dtype=bool) if not mark_clipping
                   else (maxs[:count] >= CLIP_CEILING)
                   | (mins[:count] <= -CLIP_CEILING))

        painter.setPen(QPen(colour, 1))
        for i in range(count):
            if not clipped[i]:
                painter.drawLine(i, int(top[i]), i, int(bottom[i]))

        if not clipped.any():
            return
        painter.setPen(QPen(CLIP, 1))
        for i in np.flatnonzero(clipped):
            painter.drawLine(int(i), int(top[i]), int(i), int(bottom[i]))
            painter.drawLine(int(i), int(lane.top()),
                             int(i), int(lane.top()) + CLIP_MARK)

    def paintEvent(self, event):
        if (self._pixmap is None
                or (self.width(), self.height()) != self._pixmap_size
                or self.devicePixelRatioF() != self._pixmap_ratio):
            self._build_pixmap()

        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)

        if self._envelope is None or not self._frames:
            painter.setPen(TEXT)
            painter.drawText(self.rect(), Qt.AlignCenter, self._label)
            return

        painter.setPen(QPen(PLAYHEAD, 1))
        x = self._playhead_x(self._position)
        painter.drawLine(int(x), TEXT_BAND, int(x), self.height())

        painter.setPen(TEXT)
        band = self.rect().adjusted(8, 4, -8, 0)
        painter.drawText(band, Qt.AlignLeft | Qt.AlignTop, self._label)
        secs = self._position / self._samplerate
        total = self._frames / self._samplerate
        painter.drawText(band, Qt.AlignRight | Qt.AlignTop,
                         f"{_mmss(secs)} / {_mmss(total)}")
        if self._peak_db > -math.inf and not self._provisional:
            painter.setPen(CLIP if self._clipped else PEAK)
            note = ""
            scale = self.scale
            if abs(scale - 1.0) > 0.01:
                note = (f"   (fit, {scale:.1f}x)" if self._zoom == ZOOM_FIT
                        else f"   ({scale:g}x)")
            if self._clipped:
                note += "   · clipping"
            painter.drawText(band, Qt.AlignHCenter | Qt.AlignTop,
                             f"peak {self._peak_db:.2f} dBFS "
                             f"@ {_mmss(self._peak_frame / self._samplerate)}{note}")

    def _playhead_x(self, frame: int) -> float:
        if not self._frames or self.width() <= 0:
            return 0.0
        return self.width() * frame / self._frames

    def _frame_at(self, x: float) -> int:
        if not self._frames or self.width() <= 0:
            return 0
        return int(np.clip(x / self.width() * self._frames, 0, self._frames - 1))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._frames:
            frame = self._frame_at(event.position().x())
            self.set_position(frame)
            self.seeked.emit(frame)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self._frames:
            frame = self._frame_at(event.position().x())
            self.set_position(frame)
            self.scrubbed.emit(frame)


def _has_clipping(envelope) -> bool:
    if envelope is None or not envelope.frames:
        return False
    return bool((envelope.maxs >= CLIP_CEILING).any()
                or (envelope.mins <= -CLIP_CEILING).any())


def _mmss(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"
