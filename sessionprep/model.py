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
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import (QAbstractTableModel, QMimeData, QModelIndex,
                            Qt, Signal)
from PySide6.QtGui import QFont

from . import classify, theme

from .analysis import (CLIP_MIN_RUN, NORMALISE_REF_DBFS, ClipFlag, FileInfo,
                       PolarityFlag, SilenceFlag, StereoAnalysis, StereoKind)
from .convert import (ConvertResult, MonoMethod, NormaliseMode,
                      NormaliseSpec, TargetSpec)
from .dither import DitherMode

KEEP = "keep"
FLOAT = "float"

RATE_CHOICES: list[tuple[object, str]] = [
    (KEEP, "Keep source"),
    (22050, "22.05 kHz"), (32000, "32 kHz"), (44100, "44.1 kHz"), (48000, "48 kHz"),
    (88200, "88.2 kHz"), (96000, "96 kHz"), (176400, "176.4 kHz"), (192000, "192 kHz"),
]

DEPTH_CHOICES: list[tuple[object, str]] = [
    (KEEP, "Keep source"),
    (16, "16-bit"), (24, "24-bit"), (32, "32-bit int"), (FLOAT, "32-bit float"),
]

DEFAULT_PATTERN = "{name}"

DEFAULT_TAGS = ["dr", "pc", "bs", "ks", "gtr", "orc", "bv", "lv"]

_ILLEGAL = str.maketrans({**{c: "_" for c in "/\\"},
                          **{c: None for c in ':*?"<>|'}})

RENAME_TOKENS = [
    ("{name}", "original file name"),
    ("{num}", "running order, padded to suit the total (01.. or 001..)"),
    ("{rate}", "output sample rate in Hz, e.g. 44100"),
    ("{depth}", "output bit depth, e.g. 24 or float"),
    ("{ch}", "mono or stereo, after any fold"),
]

def trim_characters(stem: str, front: int, back: int) -> str:
    if front:
        stem = stem[front:]
    if back:
        stem = stem[:-back] if back < len(stem) else ""
    return stem.strip()

def trim_at(stem: str, delimiter: str, keep_after: bool,
            last: bool = False, keep_delimiter: bool = False) -> str:
    if not delimiter:
        return stem
    position = stem.rfind(delimiter) if last else stem.find(delimiter)
    if position < 0:
        return stem
    if keep_after:
        start = position if keep_delimiter else position + len(delimiter)
        return stem[start:].strip()
    end = position + len(delimiter) if keep_delimiter else position
    return stem[:end].strip()

def sanitise_tag(text: str) -> str:
    return sanitise_stem(str(text).replace("_", " ")).replace(" ", "_")

def sanitise_stem(text: str) -> str:
    cleaned = text.translate(_ILLEGAL).strip().strip(".")
    return " ".join(cleaned.split())

NORM_CHOICES: list[tuple[object, str]] = [
    (NormaliseSpec(NormaliseMode.NONE), "Off"),
    (NormaliseSpec(NormaliseMode.LUFS, -14.0), "LUFS -14"),
    (NormaliseSpec(NormaliseMode.LUFS, -16.0), "LUFS -16"),
    (NormaliseSpec(NormaliseMode.LUFS, -18.0), "LUFS -18"),
    (NormaliseSpec(NormaliseMode.LUFS, -23.0), "LUFS -23"),
    (NormaliseSpec(NormaliseMode.PEAK, -0.1), "Peak -0.1"),
    (NormaliseSpec(NormaliseMode.PEAK, -0.3), "Peak -0.3"),
    (NormaliseSpec(NormaliseMode.PEAK, -1.0), "Peak -1.0"),
    (NormaliseSpec(NormaliseMode.RMS, -18.0), "RMS -18"),
    (NormaliseSpec(NormaliseMode.RMS, -20.0), "RMS -20"),
    (NormaliseSpec(NormaliseMode.VU, 0.0), "0 VU"),
    (NormaliseSpec(NormaliseMode.PPM_NORDIC, 0.0), "0 PPM Nordic"),
]

POLARITY_LABEL = {PolarityFlag.INVERTED: "inverted",
                  PolarityFlag.CANCELS: "cancels in mono"}
CLIP_LABEL = {ClipFlag.CLIPPED: "clipped",
              ClipFlag.INTERSAMPLE: "over 0 dBTP"}

SEVERE = {"silent", "clipped", "inverted"}

KIND_LABEL = {
    StereoKind.MONO: "mono",
    StereoKind.IDENTICAL: "fake stereo (identical)",
    StereoKind.NEAR_IDENTICAL: "fake stereo (below threshold)",
    StereoKind.SCALED: "fake stereo (level offset)",
    StereoKind.INVERTED: "polarity inverted",
    StereoKind.STEREO: "true stereo",
    StereoKind.SILENT: "silent",
    StereoKind.MULTICHANNEL: "multichannel",
}

@dataclass
class MasterSettings:
    samplerate: object = 44100
    bit_depth: object = 24
    mono_method: MonoMethod = MonoMethod.AVERAGE
    dither: DitherMode = DitherMode.SHAPED
    threshold_db: float = -60.0
    rename_pattern: str = DEFAULT_PATTERN
    number_names: bool = False
    phase_threshold_db: float = -40.0
    normalise: NormaliseSpec = None
    prevent_clipping: bool = True
    preserve_chunks: bool = True
    auto_mono: bool = True

    def __post_init__(self):
        if self.normalise is None:
            self.normalise = NormaliseSpec()

@dataclass
class Job:
    info: FileInfo
    analysis: StereoAnalysis | None = None
    mono_override: bool | None = None
    rate_override: object | None = None
    depth_override: object | None = None
    norm_override: object | None = None
    name_override: str | None = None
    tag: str = ""
    status: str = ""
    result: ConvertResult | None = None
    analysing: bool = True

    def auto_mono(self, master: MasterSettings) -> bool:
        if not master.auto_mono or self.analysis is None:
            return False
        return self.analysis.is_fake_stereo

    def mono(self, master: MasterSettings) -> bool:
        if self.info.channels < 2:
            return False
        if self.mono_override is not None:
            return self.mono_override
        return self.auto_mono(master)

    def is_mono_overridden(self, master: MasterSettings) -> bool:
        return (self.mono_override is not None
                and self.mono_override != self.auto_mono(master))

    def rate(self, master: MasterSettings) -> int:
        value = self.rate_override if self.rate_override is not None else master.samplerate
        return self.info.samplerate if value == KEEP else int(value)

    def depth(self, master: MasterSettings) -> object:
        value = self.depth_override if self.depth_override is not None else master.bit_depth
        if value == KEEP:
            return self.info.bit_depth or FLOAT
        return value

    def normalise(self, master: MasterSettings) -> NormaliseSpec:
        if self.norm_override is None:
            return master.normalise
        return NormaliseSpec(self.norm_override.mode, self.norm_override.target,
                             master.normalise.ceiling_dbtp)

    def base_stem(self, master: MasterSettings, index: int,
                  total: int = 1) -> str:
        depth = self.depth(master)
        mono = self.mono(master) or self.info.channels == 1
        width = max(2, len(str(max(total, 1))))
        tokens = {
            "name": self.info.path.stem,
            "num": f"{index + 1:0{width}d}",
            "index": index + 1,
            "rate": self.rate(master),
            "khz": f"{self.rate(master) / 1000:g}",
            "depth": "float" if depth == FLOAT else str(depth),
            "ch": "mono" if mono else "stereo",
            "parent": self.info.path.parent.name,
        }
        if self.name_override:
            return self.name_override
        pattern = master.rename_pattern or DEFAULT_PATTERN
        try:
            stem = sanitise_stem(pattern.format(**tokens))
        except (KeyError, IndexError, ValueError):
            stem = self.info.path.stem
        return stem or self.info.path.stem

    def output_stem(self, master: MasterSettings, index: int,
                    total: int = 1) -> str:
        stem = self.base_stem(master, index, total)
        pattern = master.rename_pattern or DEFAULT_PATTERN
        if master.number_names and "{num}" not in pattern:
            width = max(2, len(str(max(total, 1))))
            stem = f"{index + 1:0{width}d}_{stem}"
        if self.tag:
            stem = f"{stem}_{self.tag}"
        return stem

    def spec(self, master: MasterSettings) -> TargetSpec:
        depth = self.depth(master)
        method = master.mono_method
        if (method is MonoMethod.AVERAGE and self.analysis is not None
                and self.analysis.needs_single_channel_fold):
            method = MonoMethod.LOUDEST
        return TargetSpec(
            samplerate=self.rate(master),
            bit_depth=0 if depth == FLOAT else int(depth),
            to_mono=self.mono(master),
            mono_method=method,
            dither=master.dither,
            prevent_clipping=master.prevent_clipping,
            preserve_chunks=master.preserve_chunks,
            normalise=self.normalise(master),
        )

COLUMNS = ["File", "In", "Rate", "Depth", "Peak", "LUFS", "Corr", "Width",
           "Flags", "Mono", "Normalise", "Out rate", "Out depth", "Tag",
           "Name", "Out name", "Status"]
COL_FILE, COL_IN, COL_RATE, COL_DEPTH, COL_PEAK, COL_LUFS, COL_CORR, \
    COL_VERDICT, COL_FLAGS, COL_MONO, COL_NORM, COL_ORATE, COL_ODEPTH, \
    COL_TAG, COL_NAME, COL_ONAME, COL_STATUS = range(17)

def _channel_name(index: int, total: int) -> str:
    if total == 2:
        return "L" if index == 0 else "R"
    return f"ch{index + 1}"

def flag_tokens(a: StereoAnalysis) -> list[str]:
    tokens: list[str] = []
    if a.silence is SilenceFlag.SILENT:
        tokens.append("silent")
    elif a.silence is SilenceFlag.CHANNEL_SILENT:
        dead = [_channel_name(i, len(a.channel_peaks_db))
                for i, pk in enumerate(a.channel_peaks_db) if pk < -90.0]
        tokens.append("dead " + "/".join(dead) if dead else "dead channel")
    if a.clipping in CLIP_LABEL:
        tokens.append(CLIP_LABEL[a.clipping])
    if a.polarity in POLARITY_LABEL:
        tokens.append(POLARITY_LABEL[a.polarity])
    return tokens

TAG_COLOUR = theme.colour("tag")
MONO_COLOUR = theme.colour("mono")
OVERRIDE_COLOUR = theme.colour("override")
WARN_COLOUR = theme.colour("warning")
LOUDEST_COLOUR = theme.colour("loudest")
QUIETEST_COLOUR = theme.colour("quietest")
DIM = theme.colour("muted")
EXTREME_COLUMNS = (COL_FILE, COL_PEAK)

ROW_MIME = "application/x-sessionprep-rows"

class JobModel(QAbstractTableModel):
    reordered = Signal(list)

    def __init__(self, master: MasterSettings, parent=None):
        super().__init__(parent)
        self.jobs: list[Job] = []
        self.master = master
        self.tags: list[str] = []
        self._extremes: dict[int, tuple[int, int]] | None = None
        self._clashes: set[int] | None = None
        for signal in (self.modelReset, self.rowsInserted,
                       self.rowsRemoved, self.dataChanged):
            signal.connect(self._invalidate_cache)

    def _invalidate_cache(self, *_) -> None:
        self._extremes = None
        self._clashes = None

    def register_tag(self, tag: str) -> bool:
        tag = sanitise_tag(tag)
        if not tag or tag in self.tags:
            return False
        self.tags.append(tag)
        self.tags.sort(key=str.lower)
        return True

    def set_tag(self, rows: list[int], tag: str) -> None:
        tag = sanitise_tag(tag)
        for row in rows:
            if 0 <= row < len(self.jobs):
                self.jobs[row].tag = tag
        self.register_tag(tag)
        self.refresh_all()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.jobs)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return COLUMNS[section]
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsDropEnabled
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled
        col = index.column()
        job = self.jobs[index.row()]
        if col == COL_MONO:
            if job.info.channels < 2:
                return base
            return base | Qt.ItemIsUserCheckable
        if col in (COL_ORATE, COL_ODEPTH, COL_NORM, COL_TAG, COL_NAME):
            return base | Qt.ItemIsEditable
        return base

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        job = self.jobs[index.row()]
        col = index.column()
        m = self.master

        if role == Qt.CheckStateRole and col == COL_MONO:
            if job.info.channels < 2:
                return None
            return Qt.Checked if job.mono(m) else Qt.Unchecked

        if role == Qt.DisplayRole:
            return self._display(job, col, m, index.row())

        if role == Qt.ForegroundRole:
            if col in EXTREME_COLUMNS:
                marked = self.extreme_colour(index.row(), col)
                if marked is not None:
                    return marked
            if col == COL_MONO and job.mono(m):
                return OVERRIDE_COLOUR if job.is_mono_overridden(m) else MONO_COLOUR
            if col == COL_FLAGS and job.analysis and job.analysis.has_issue:
                tokens = flag_tokens(job.analysis)
                severe = any(t in SEVERE or t.startswith("dead") for t in tokens)
                return WARN_COLOUR if severe else OVERRIDE_COLOUR
            if col == COL_VERDICT and job.analysis:
                if job.analysis.kind.is_fake_stereo:
                    return MONO_COLOUR
                if job.analysis.kind is StereoKind.INVERTED:
                    return WARN_COLOUR
            if col == COL_TAG:
                return TAG_COLOUR if job.tag else DIM
            if col == COL_NAME:
                return OVERRIDE_COLOUR if job.name_override else DIM
            if col == COL_ONAME:
                if index.row() in self.clashing_rows():
                    return WARN_COLOUR
                return DIM
            if col in (COL_ORATE, COL_ODEPTH, COL_NORM):
                overridden = {COL_ORATE: job.rate_override,
                              COL_ODEPTH: job.depth_override,
                              COL_NORM: job.norm_override}[col] is not None
                return OVERRIDE_COLOUR if overridden else DIM
            if col == COL_STATUS and job.result and not job.result.ok:
                return WARN_COLOUR
            if (col == COL_CORR and job.analysis is not None
                    and job.info.channels >= 2
                    and job.analysis.kind is not StereoKind.SILENT):
                if job.analysis.correlation <= -0.999:
                    return WARN_COLOUR
                if job.analysis.correlation < 0:
                    return OVERRIDE_COLOUR
                return DIM
            if col in (COL_IN, COL_RATE, COL_DEPTH, COL_PEAK, COL_LUFS,
                       COL_CORR):
                return DIM

        if role == Qt.FontRole and col == COL_MONO and job.is_mono_overridden(m):
            f = QFont()
            f.setBold(True)
            return f

        if role == Qt.ToolTipRole:
            return self._tooltip(job, col, m)

        if role == Qt.TextAlignmentRole and col in (COL_IN, COL_RATE, COL_DEPTH,
                                                   COL_PEAK, COL_LUFS, COL_CORR):
            return int(Qt.AlignRight | Qt.AlignVCenter)

        if role == Qt.EditRole and col == COL_TAG:
            return job.tag
        if role == Qt.EditRole and col == COL_NAME:
            return job.base_stem(self.master, index.row(), len(self.jobs))

        if role == Qt.UserRole:
            if col == COL_ORATE:
                return job.rate_override
            if col == COL_ODEPTH:
                return job.depth_override
            if col == COL_NORM:
                return job.norm_override
        return None

    def _display(self, job: Job, col: int, m: MasterSettings, row: int = 0):
        info = job.info
        if col == COL_FILE:
            return info.path.name
        if col == COL_IN:
            return {1: "mono", 2: "stereo"}.get(info.channels, f"{info.channels} ch")
        if col == COL_RATE:
            return f"{info.samplerate / 1000:g} k"
        if col == COL_DEPTH:
            return f"{info.bit_depth}-bit" if info.bit_depth else info.subtype
        a = job.analysis
        if col in (COL_PEAK, COL_LUFS, COL_CORR):
            if job.analysing:
                return "…"
            if a is None:
                return "—"
            if col == COL_PEAK:
                return "—" if a.peak_dbfs <= -200 else f"{a.peak_dbfs:.1f}"
            if col == COL_LUFS:
                if a.loudness_lufs == -math.inf:
                    return "—"
                return (f"{a.loudness_lufs:.1f}"
                        + ("" if a.loudness_gated else "*"))
            if (info.channels < 2 or math.isnan(a.correlation)
                    or a.kind is StereoKind.SILENT):
                return "—"
            return f"{a.correlation:+.2f}"
        if col == COL_VERDICT:
            if job.analysing:
                return "analysing…"
            return KIND_LABEL.get(job.analysis.kind, "?") if job.analysis else "—"
        if col == COL_FLAGS:
            if job.analysing or job.analysis is None:
                return ""
            return " · ".join(flag_tokens(job.analysis))
        if col == COL_MONO:
            if info.channels < 2:
                return "already mono"
            if job.is_mono_overridden(m):
                return "manual"
            return "auto" if job.mono(m) else ""
        if col == COL_ORATE:
            rate = job.rate(m)
            return f"{rate / 1000:g} k"
        if col == COL_ODEPTH:
            depth = job.depth(m)
            return "32-bit float" if depth == FLOAT else f"{depth}-bit"
        if col == COL_NORM:
            return job.normalise(m).label
        if col == COL_TAG:
            return job.tag
        if col == COL_NAME:
            return job.base_stem(m, row, len(self.jobs))
        if col == COL_ONAME:
            return job.output_stem(m, row, len(self.jobs))
        if col == COL_STATUS:
            return job.status
        return None

    def _tooltip(self, job: Job, col: int, m: MasterSettings):
        a = job.analysis
        if col == COL_FLAGS and a is not None:
            if not a.has_issue:
                return ""
            lines: list[str] = []
            if a.silence is SilenceFlag.SILENT:
                lines += ["Silent: nothing above -90 dBFS anywhere in the file.", ""]
            elif a.silence is SilenceFlag.CHANNEL_SILENT:
                peaks = ", ".join(
                    f"{_channel_name(i, len(a.channel_peaks_db))} "
                    f"{('-inf' if pk == -math.inf else format(pk, '.1f'))} dBFS"
                    for i, pk in enumerate(a.channel_peaks_db))
                lines += [f"One channel carries no signal ({peaks}).", ""]
            if a.clipping is ClipFlag.CLIPPED:
                lines += [
                    f"Clipped: {a.clipped_samples:,} samples pinned to full scale "
                    f"in {a.clip_runs:,} runs of {CLIP_MIN_RUN}+ "
                    f"(longest {a.clip_longest_run:,}).",
                    f"True peak {a.true_peak_dbfs:+.2f} dBTP.", ""]
            elif a.clipping is ClipFlag.INTERSAMPLE:
                lines += [
                    f"No clipped samples, but the reconstructed waveform reaches "
                    f"{a.true_peak_dbfs:+.2f} dBTP against a sample peak of "
                    f"{a.peak_dbfs:.2f} dBFS.",
                    "Resampling can turn that into real clipping; the clip guard "
                    "will trim it.", ""]
            if a.has_polarity_issue:
                lines += [
                    ("The two channels are near-perfect opposites."
                     if a.polarity is PolarityFlag.INVERTED
                     else "Folding to mono all but nulls this file."),
                    f"L/R correlation: {a.correlation:.3f}",
                    f"Summing to mono would lose {abs(a.mono_loss_db):.1f} dB RMS",
                    "Forcing mono uses a single-channel fold so it isn't cancelled.",
                ]
            return "\n".join(lines).strip()
        if col == COL_CORR and a is not None:
            summed = ("n/a" if math.isnan(a.mono_sum_db)
                      else ("nulls out" if a.mono_sum_db <= -120
                            else f"{a.mono_sum_db - NORMALISE_REF_DBFS:+.2f} dBFS"))
            return "\n".join([
                f"L/R correlation: {a.correlation:.4f}",
                f"Mono sum, peak change: {summed}",
                f"Fold loss vs source: {abs(a.mono_loss_db):.1f} dB RMS",
            ])
        if col == COL_PEAK and a is not None:
            if a.peak_dbfs <= -200:
                return "Silent."
            return (f"Sample peak: {a.peak_dbfs:.2f} dBFS\n"
                    f"True peak: {a.true_peak_dbfs:+.2f} dBTP")
        if col == COL_LUFS and a is not None:
            if a.loudness_lufs == -math.inf:
                return "Too quiet to measure."
            note = ("" if a.loudness_gated else
                    "\n* shorter than the 400 ms gating block, so this is an "
                    "ungated whole-file measurement.")
            start = ("never rises above its own noise floor"
                     if a.start_seconds == math.inf
                     else f"{int(a.start_seconds // 60)}:"
                          f"{a.start_seconds % 60:05.2f}")
            return (f"BS.1770-4 integrated loudness: {a.loudness_lufs:.2f} LUFS\n"
                    f"RMS (loudest channel): {a.rms_dbfs:.2f} dBFS\n"
                    f"Starts playing at: {start}" + note)
        if col == COL_VERDICT and a is not None:
            lines = [
                f"Side/mid RMS: {a.side_rms_db:.1f} dB",
                f"Side/mid peak: {a.side_peak_db:.1f} dB",
                f"R relative to L: {a.gain_ratio_db:+.2f} dB",
            ]
            return "\n".join(lines)
        if col == COL_TAG:
            return ("")
        if col == COL_NAME:
            return ("")
        if col == COL_ONAME:
            base = ("")
            if job in [self.jobs[r] for r in self.clashing_rows()]:
                return (base + "\n\nAnother file resolves to this same name in the\n"
                        "same folder. A numeric suffix will be added so nothing\n"
                        "is overwritten.")
            return base
        if col == COL_MONO:
            return ("")
        if col == COL_FILE:
            return str(job.info.path)
        if col == COL_STATUS and job.result:
            parts = [job.result.message]
            parts += job.result.warnings
            if job.result.chunks_carried:
                parts.append(f"Carried metadata: {job.result.chunks_carried}")
            return "\n".join(p for p in parts if p)
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid():
            return False
        job = self.jobs[index.row()]
        col = index.column()

        if role == Qt.CheckStateRole and col == COL_MONO:
            want = Qt.CheckState(value) == Qt.Checked
            job.mono_override = None if want == job.auto_mono(self.master) else want
            self.dataChanged.emit(index, index)
            return True

        if role == Qt.EditRole and col == COL_NAME:
            typed = sanitise_stem(str(value))
            total = len(self.jobs)
            previous, job.name_override = job.name_override, None
            auto = job.base_stem(self.master, index.row(), total)
            job.name_override = None if (not typed or typed == auto) else typed
            if job.name_override != previous:
                self.refresh_all()
            return True

        if role == Qt.EditRole and col in (COL_ORATE, COL_ODEPTH, COL_NORM):
            if col == COL_ORATE:
                job.rate_override = value
            elif col == COL_ODEPTH:
                job.depth_override = value
            else:
                job.norm_override = value
            self.dataChanged.emit(index, index)
            return True
        return False

    def supportedDropActions(self):
        return Qt.MoveAction

    def supportedDragActions(self):
        return Qt.MoveAction

    def mimeTypes(self):
        return [ROW_MIME]

    def mimeData(self, indexes):
        rows = sorted({i.row() for i in indexes if i.isValid()})
        payload = QMimeData()
        payload.setData(ROW_MIME, ",".join(str(r) for r in rows).encode())
        return payload

    def canDropMimeData(self, data, action, row, column, parent):
        return data is not None and data.hasFormat(ROW_MIME)

    def dropMimeData(self, data, action, row, column, parent):
        if action == Qt.IgnoreAction or not self.canDropMimeData(
                data, action, row, column, parent):
            return False

        try:
            source = [int(part) for part in
                      bytes(data.data(ROW_MIME)).decode().split(",") if part]
        except ValueError:
            return False
        source = [r for r in source if 0 <= r < len(self.jobs)]
        if not source:
            return False

        if row != -1:
            target = row
        elif parent.isValid():
            target = parent.row()
        else:
            target = len(self.jobs)

        moving = [self.jobs[r] for r in source]
        target -= sum(1 for r in source if r < target)

        self.beginResetModel()
        for job in moving:
            self.jobs.remove(job)
        self.jobs[target:target] = moving
        self.endResetModel()

        self.reordered.emit(moving)
        return False

    def move_rows(self, rows: list[int], offset: int) -> list[Job]:
        rows = sorted({r for r in rows if 0 <= r < len(self.jobs)})
        if not rows or offset == 0:
            return []
        if offset < 0 and rows[0] == 0:
            return []
        if offset > 0 and rows[-1] == len(self.jobs) - 1:
            return []

        moving = [self.jobs[r] for r in rows]
        self.beginResetModel()
        for job in moving:
            self.jobs.remove(job)
        insert_at = max(0, min(rows[0] + offset, len(self.jobs)))
        self.jobs[insert_at:insert_at] = moving
        self.endResetModel()
        self.reordered.emit(moving)
        return moving

    def rows_for(self, jobs: list[Job]) -> list[int]:
        wanted = {id(job) for job in jobs}
        return [r for r, job in enumerate(self.jobs) if id(job) in wanted]

    def classify(self) -> list:
        return classify.classify_many(
            [job.info.path.stem for job in self.jobs])

    def sort_by_group(self) -> list:
        total = len(self.jobs)
        by_file = self.classify()
        by_name = classify.classify_many(
            [job.base_stem(self.master, row, total)
             for row, job in enumerate(self.jobs)])

        part_start = self._earliest_per_part(by_file, by_name)

        order = sorted(range(total),
                       key=lambda row: self._group_sort_key(
                           row, by_file, by_name, part_start))
        if order == list(range(total)):
            return []
        moved = [self.jobs[i] for i in order]
        self.beginResetModel()
        self.jobs = moved
        self.endResetModel()
        self.reordered.emit(moved)
        return moved

    def _part_of(self, row: int, by_file: list, by_name: list) -> tuple:
        chosen = by_name[row] or by_file[row]
        if chosen is None:
            return ("", "", ""), classify.CAPTURE_NEUTRAL, 0
        return ((chosen.group, chosen.role, chosen.part),
                chosen.capture_rank, chosen.capture_number)

    def _earliest_per_part(self, by_file: list, by_name: list) -> dict:
        earliest: dict = {}
        for row, job in enumerate(self.jobs):
            key, _rank, _number = self._part_of(row, by_file, by_name)
            start = job.analysis.start_seconds if job.analysis else math.inf
            if key not in earliest or start < earliest[key]:
                earliest[key] = start
        return earliest

    def _group_sort_key(self, row: int, by_file: list, by_name: list,
                        part_start: dict) -> tuple:
        job = self.jobs[row]
        named, filed = by_name[row], by_file[row]

        group = job.tag or None
        if group is None:
            group = ((named.group if named else None)
                     or (filed.group if filed else None))

        if group in classify.GROUP_ORDER:
            group_rank = classify.GROUP_ORDER.index(group)
            group_key = ""
        elif group:
            group_rank = len(classify.GROUP_ORDER)
            group_key = group.lower()
        else:
            group_rank = len(classify.GROUP_ORDER) + 1
            group_key = ""

        role_rank = classify.UNRANKED_ROLE
        for candidate in (named, filed):
            if candidate is not None and candidate.group == group:
                role_rank = candidate.rank[1]
                break

        start = math.inf
        if job.analysis is not None:
            start = job.analysis.start_seconds

        part_key, capture_rank, capture_number = self._part_of(
            row, by_file, by_name)
        block_start = part_start.get(part_key, start)

        return (group_rank, group_key, role_rank,
                block_start, part_key[2],
                capture_rank, capture_number,
                start, row)

    def sort_by_name(self) -> None:
        self.beginResetModel()
        self.jobs.sort(key=lambda j: str(j.info.path).lower())
        self.endResetModel()

    def add_jobs(self, jobs: list[Job]) -> None:
        if not jobs:
            return
        start = len(self.jobs)
        self.beginInsertRows(QModelIndex(), start, start + len(jobs) - 1)
        self.jobs.extend(jobs)
        self.endInsertRows()

    def remove_rows(self, rows: list[int]) -> None:
        for row in sorted(set(rows), reverse=True):
            if 0 <= row < len(self.jobs):
                self.beginRemoveRows(QModelIndex(), row, row)
                del self.jobs[row]
                self.endRemoveRows()

    def clear(self) -> None:
        self.beginResetModel()
        self.jobs.clear()
        self.endResetModel()

    def existing_paths(self) -> set[Path]:
        return {job.info.path for job in self.jobs}

    def row_changed(self, row: int) -> None:
        if 0 <= row < len(self.jobs):
            self.dataChanged.emit(self.index(row, 0),
                                  self.index(row, len(COLUMNS) - 1))

    def refresh_all(self) -> None:
        if self.jobs:
            self.dataChanged.emit(self.index(0, 0),
                                  self.index(len(self.jobs) - 1, len(COLUMNS) - 1))

    def apply_to_rows(self, rows: list[int], **overrides) -> None:
        for row in rows:
            if 0 <= row < len(self.jobs):
                for key, value in overrides.items():
                    setattr(self.jobs[row], key, value)
                self.row_changed(row)

    def counts(self) -> tuple[int, int, int]:
        mono = sum(1 for j in self.jobs if j.mono(self.master))
        flagged = sum(1 for j in self.jobs
                      if j.analysis is not None and j.analysis.has_issue)
        return len(self.jobs), mono, flagged

    def clashing_rows(self) -> set[int]:
        if self._clashes is None:
            seen: dict[tuple, int] = {}
            clashes: set[int] = set()
            for row, job in enumerate(self.jobs):
                key = (job.info.path.parent,
                       job.output_stem(self.master, row, len(self.jobs)).lower())
                if key in seen:
                    clashes.add(seen[key])
                    clashes.add(row)
                else:
                    seen[key] = row
            self._clashes = clashes
        return self._clashes

    def flagged_rows(self) -> list[int]:
        return [r for r, j in enumerate(self.jobs)
                if j.analysis is not None and j.analysis.has_issue]

    def _rank(self, read, floor: float) -> tuple[int, int]:
        values = []
        for row, job in enumerate(self.jobs):
            if job.analysing or job.analysis is None:
                continue
            value = read(job.analysis)
            if value > floor:
                values.append((value, row))
        if len(values) < 2:
            return -1, -1
        return max(values)[1], min(values)[1]

    def extremes(self) -> dict[int, tuple[int, int]]:
        if self._extremes is None:
            by_peak = self._rank(lambda a: a.peak_dbfs, -200.0)
            by_lufs = self._rank(lambda a: a.loudness_lufs, -math.inf)
            self._extremes = {
                COL_PEAK: by_peak,
                COL_LUFS: by_lufs,
                COL_FILE: by_peak,
            }
        return self._extremes

    def extreme_colour(self, row: int, col: int):
        loudest, quietest = self.extremes().get(col, (-1, -1))
        if row == loudest:
            return LOUDEST_COLOUR
        if row == quietest:
            return QUIETEST_COLOUR
        return None
