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

import html
import json
import math
import os
import re
import sys
import traceback
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import (QItemSelection, QItemSelectionModel, QObject,
                            QRectF, QRunnable, QSettings, QSize, Qt, QThread,
                            QThreadPool, QTimer, Signal, Slot)
from PySide6.QtGui import (QAction, QFont, QFontMetrics, QKeySequence,
                           QPainter, QPen, QPixmap)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView,
                               QInputDialog, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton,
                               QSpinBox, QSplitter, QStyledItemDelegate,
                               QTableWidget, QTableWidgetItem, QTableView,
                               QVBoxLayout, QWidget)

from . import __version__, classify
from .analysis import (analyse, build_coarse_envelope, build_envelope,
                       probe, wants_coarse_pass)
from .convert import (MonoMethod, NormaliseMode, NormaliseSpec, convert,
                      measured_level_db, plan_relative, target_level_db)
from .dither import HAVE_NUMBA, DitherMode
from .model import (COL_FILE, COL_LUFS, COL_NAME, COL_NORM, COL_ODEPTH,
                    COL_ONAME, COL_ORATE, COL_PEAK, COL_STATUS, COL_TAG,
                    DEFAULT_PATTERN, DEFAULT_TAGS, DEPTH_CHOICES, DIM,
                    NORM_CHOICES, RATE_CHOICES, RENAME_TOKENS, Job, JobModel,
                    MasterSettings, flag_tokens, sanitise_stem, sanitise_tag,
                    trim_at, trim_characters)
from . import errors, theme
from .player import Player, playable_samplerate
from .waveform import ZOOM_CHOICES, WaveformView

APP_NAME = "Session Prep"
ORG_NAME = "SG"
ORG_DOMAIN = "session-prep.local"

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".aifc", ".flac", ".ogg", ".w64",
              ".caf", ".au", ".snd", ".rf64", ".mat", ".voc"}

def selected_rows(table) -> list[int]:
    return sorted({i.row() for i in table.selectionModel().selectedRows()})



class AnalysisSignals(QObject):
    done = Signal(object, object, str)

class AnalysisTask(QRunnable):
    def __init__(self, job, threshold: float, phase_threshold: float,
                 signals: AnalysisSignals):
        super().__init__()
        self.job, self.threshold, self.signals = job, threshold, signals
        self.phase_threshold = phase_threshold

    @Slot()
    def run(self):
        try:
            result, error = analyse(self.job.info.path, self.threshold,
                                    self.phase_threshold), ""
        except Exception as exc:
            result, error = None, str(exc)
        try:
            self.signals.done.emit(self.job, result, error)
        except RuntimeError:
            pass

class ConvertWorker(QThread):
    progress = Signal(int, int, str)
    finished_all = Signal(int, int, int)

    def __init__(self, jobs, master, out_dir, root, on_noop, parent=None):
        super().__init__(parent)
        self.jobs, self.master = jobs, master
        self.out_dir, self.root, self.on_noop = out_dir, root, on_noop
        self._cancel = False
        self._used: set[str] = set()

    def cancel(self):
        self._cancel = True

    def run(self):
        ok = failed = skipped = 0
        for i, (row, job) in enumerate(self.jobs):
            if self._cancel:
                break
            try:
                dest = self._dest_for(job, row)
                result = convert(job.info, job.spec(self.master), dest, self.on_noop)
                job.result = result
                if not result.ok:
                    failed += 1
                elif result.skipped and result.dest is None:
                    skipped += 1
                else:
                    ok += 1
                note = result.message
                if result.warnings:
                    note += "  ⚠ " + result.warnings[0]
                job.status = note
            except Exception:
                failed += 1
                job.status = "Error: " + traceback.format_exc(limit=1).strip().splitlines()[-1]
            self.progress.emit(row, i + 1, job.status)
        self.finished_all.emit(ok, failed, skipped)

    def _dest_for(self, job, row: int) -> Path:
        src = job.info.path
        if self.root is not None:
            try:
                rel = src.relative_to(self.root)
            except ValueError:
                rel = Path(src.name)
        else:
            rel = Path(src.name)

        suffix = src.suffix if src.suffix.lower() in (".wav", ".aif", ".aiff",
                                                      ".flac") else ".wav"
        dest = (self.out_dir / rel).with_name(
            job.output_stem(self.master, row, len(self.jobs)) + suffix)

        if dest.resolve() == src.resolve():
            dest = dest.with_name(dest.stem + "_converted" + suffix)

        base, index = dest, 2
        while str(dest).lower() in self._used:
            dest = base.with_name(f"{base.stem}_{index}{suffix}")
            index += 1
        self._used.add(str(dest).lower())
        return dest

class FileTableView(QTableView):

    filesDropped = Signal(list)
    resized = Signal()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()

    @staticmethod
    def _urls(event):
        data = event.mimeData()
        return [Path(u.toLocalFile()) for u in data.urls() if u.isLocalFile()] \
            if data.hasUrls() else []

    def dragEnterEvent(self, event):
        if self._urls(event):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if self._urls(event):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = self._urls(event)
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def mousePressEvent(self, event):
        index = self.indexAt(event.position().toPoint())
        if (index.isValid() and index.column() == COL_TAG
                and event.button() == Qt.LeftButton):
            rows = {i.row() for i in self.selectionModel().selectedRows()}
            if len(rows) > 1 and index.row() in rows:
                self.selectionModel().setCurrentIndex(
                    index, QItemSelectionModel.NoUpdate)
                self.edit(index)
                return
        super().mousePressEvent(event)

class PreviewSignals(QObject):
    done = Signal(int, object, object, int, str)
    current: int = -1

class PreviewTask(QRunnable):
    def __init__(self, token: int, job, signals: PreviewSignals):
        super().__init__()
        self.token, self.job = token, job
        self.signals = signals

    @Slot()
    def run(self):
        try:
            path = self.job.info.path
            if wants_coarse_pass(path) and not self._superseded():
                sketch = build_coarse_envelope(path)
                self._emit(sketch, sketch.samplerate, "")
            if self._superseded():
                return
            envelope = build_envelope(path)
        except Exception as exc:
            envelope, rate, error = None, 0, str(exc)
        else:
            rate, error = envelope.samplerate, ""
        self._emit(envelope, rate, error)

    def _superseded(self) -> bool:
        return getattr(self.signals, "current", self.token) != self.token

    def _emit(self, envelope, rate: int, error: str) -> None:
        try:
            self.signals.done.emit(self.token, self.job, envelope, rate, error)
        except RuntimeError:
            pass

class TrimDialog(QDialog):

    PREVIEW_ROWS = 8

    BY_LENGTH, BY_CHARACTER = "length", "character"

    def __init__(self, parent, names: list[str]):
        super().__init__(parent)
        self.setWindowTitle("Trim")
        self.names = names
        self.setMinimumWidth(560)

        self.layout = QVBoxLayout(self)

        chooser = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("By length — count characters off the ends",
                          self.BY_LENGTH)
        self.mode.addItem("By character — cut at a character or word",
                          self.BY_CHARACTER)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        chooser.addRow("Trim", self.mode)
        self.layout.addLayout(chooser)

        self.layout.addWidget(self._build_by_length())
        self.layout.addWidget(self._build_by_character())

        self.preview = QLabel("")
        self.preview.setTextFormat(Qt.PlainText)
        self.preview.setStyleSheet(
            f"font-family: monospace; color: {theme.colour('muted').name()}; "
            f"padding: 6px;")
        self.layout.addWidget(self.preview)

        self.warning = QLabel("")
        self.warning.setStyleSheet(f"color: {theme.colour('warning').name()};")
        self.layout.addWidget(self.warning)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        self._mode_changed()

    def _build_by_length(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        self.front = QSpinBox(); self.front.setRange(0, 99)
        self.back = QSpinBox(); self.back.setRange(0, 99)
        for spin in (self.front, self.back):
            spin.setSuffix(" chars")
            spin.valueChanged.connect(self.refresh)
        form.addRow("Remove from front", self.front)
        form.addRow("Remove from end", self.back)
        self.by_length_page = page
        return page

    def _build_by_character(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)

        self.delimiter = QLineEdit("_")
        self.delimiter.setPlaceholderText("character or text to cut at")
        self.delimiter.textChanged.connect(self.refresh)

        self.side = QComboBox()
        self.side.addItem("Keep what follows it", True)
        self.side.addItem("Keep what precedes it", False)
        self.side.currentIndexChanged.connect(self.refresh)

        self.occurrence = QComboBox()
        self.occurrence.addItem("First occurrence", False)
        self.occurrence.addItem("Last occurrence", True)
        self.occurrence.currentIndexChanged.connect(self.refresh)

        self.keep = QCheckBox("Keep the character itself")
        self.keep.toggled.connect(self.refresh)

        form.addRow("Cut at", self.delimiter)
        form.addRow("Side", self.side)
        form.addRow("Which one", self.occurrence)
        form.addRow("", self.keep)
        self.by_character_page = page
        return page

    def _mode_changed(self, *_):
        by_length = self.mode.currentData() == self.BY_LENGTH
        self.by_length_page.setVisible(by_length)
        self.by_character_page.setVisible(not by_length)
        self.adjustSize()
        self.refresh()

    def transform(self, name: str) -> str:
        if self.mode.currentData() == self.BY_LENGTH:
            return trim_characters(name, self.front.value(), self.back.value())
        return trim_at(name, self.delimiter.text(), self.side.currentData(),
                       self.occurrence.currentData(), self.keep.isChecked())

    def refresh(self):
        lines, empties = [], 0
        for name in self.names:
            result = sanitise_stem(self.transform(name))
            if not result:
                empties += 1
            if len(lines) < self.PREVIEW_ROWS:
                lines.append(f"{name}   →   {result or '(empty — left as is)'}")
        if len(self.names) > self.PREVIEW_ROWS:
            lines.append(f"… and {len(self.names) - self.PREVIEW_ROWS} more")
        self.preview.setText("\n".join(lines) or "Nothing selected")
        self.warning.setText(
            f"{empties} name{'s' if empties != 1 else ''} would be left empty "
            f"and will be kept unchanged." if empties else "")

class SuggestDialog(QDialog):

    def __init__(self, parent, rows, tags):
        super().__init__(parent)
        self.setWindowTitle("Suggested names and tags")
        self.resize(860, 620)
        self.rows = rows

        layout = QVBoxLayout(self)
        options = QHBoxLayout()
        self.set_names = QCheckBox("Set names")
        self.set_names.setChecked(True)
        self.set_tags = QCheckBox("Set tags")
        self.set_tags.setChecked(True)
        self.skip_edited = QCheckBox("Leave files I have already renamed alone")
        self.skip_edited.setChecked(True)
        for box in (self.set_names, self.set_tags, self.skip_edited):
            options.addWidget(box)
        options.addStretch(1)
        layout.addLayout(options)

        layout.addWidget(QLabel(
            "Edit any suggestion below. Rows the classifier missed are blank — "
            "fill one in and it can be remembered for next time."))

        self.table = QTableWidget(len(rows), 3)
        self.table.setHorizontalHeaderLabels(["File", "Name", "Tag"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        for row, (stem, name, tag) in enumerate(rows):
            source = QTableWidgetItem(stem)
            source.setFlags(Qt.ItemIsEnabled)
            source.setForeground(DIM)
            self.table.setItem(row, 0, source)
            self.table.setItem(row, 1, QTableWidgetItem(name or ""))

            editor = QComboBox()
            editor.setEditable(True)
            editor.addItem("")
            editor.addItems(tags)
            editor.setCurrentText(tag or "")
            self.table.setCellWidget(row, 2, editor)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setColumnWidth(2, 120)
        layout.addWidget(self.table, 1)

        hits = sum(1 for _, name, _ in rows if name)
        layout.addWidget(QLabel(
            f"{hits} recognised, {len(rows) - hits} not. "
            f"Everything here is a suggestion — the cells stay editable "
            f"afterwards too."))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Apply")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def edited(self) -> list[tuple]:
        out = []
        for row, (stem, _, _) in enumerate(self.rows):
            name_item = self.table.item(row, 1)
            editor = self.table.cellWidget(row, 2)
            out.append((stem,
                        (name_item.text().strip() if name_item else ""),
                        (editor.currentText().strip() if editor else "")))
        return out

class LearnDialog(QDialog):

    COLUMNS = ["Learn", "New Word", "Maps To", "Tag"]

    def __init__(self, parent, entries, tags, existing_rules: dict[str, str], title="Words to remember"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(660, 460)
        self.tags = tags
        self.existing_rules = existing_rules
        self.rule_names = sorted(existing_rules.keys())

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "A file whose name contains one of these words will be recognised "
            "from now on.\nMap it to an existing rule name, or type a new one."))

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 56)
        self.table.setColumnWidth(3, 110)
        layout.addWidget(self.table, 1)

        for entry in entries:
            self._add_row(entry)

        buttons_row = QHBoxLayout()
        add = QPushButton("Add word")
        add.clicked.connect(lambda: self._add_row(
            {"token": "", "name": "", "group": ""}))
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_selected)
        buttons_row.addWidget(add)
        buttons_row.addWidget(remove)
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Save")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_row(self, entry):
        row = self.table.rowCount()
        self.table.insertRow(row)

        tick = QTableWidgetItem()
        tick.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        tick.setCheckState(Qt.Checked)
        self.table.setItem(row, 0, tick)

        self.table.setItem(row, 1, QTableWidgetItem(entry.get("token", "")))

        name_combo = QComboBox()
        name_combo.setEditable(True)
        name_combo.addItem("")
        name_combo.addItems(self.rule_names)
        name_combo.setCurrentText(entry.get("name", ""))

        tag_combo = QComboBox()
        tag_combo.setEditable(True)
        tag_combo.addItem("")
        tag_combo.addItems(self.tags)
        tag_combo.setCurrentText(entry.get("group", ""))

        def on_name_changed(text, t_combo=tag_combo):
            if text in self.existing_rules:
                t_combo.setCurrentText(self.existing_rules[text])

        name_combo.currentTextChanged.connect(on_name_changed)

        self.table.setCellWidget(row, 2, name_combo)
        self.table.setCellWidget(row, 3, tag_combo)

    def _remove_selected(self):
        for row in sorted({i.row() for i in self.table.selectedIndexes()},
                          reverse=True):
            self.table.removeRow(row)

    def entries(self) -> list[dict]:
        out = []
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).checkState() != Qt.Checked:
                continue
            token = (self.table.item(row, 1).text() or "").strip()
            name_combo = self.table.cellWidget(row, 2)
            name = name_combo.currentText().strip()
            tag_combo = self.table.cellWidget(row, 3)
            group = tag_combo.currentText().strip()
            if not token or not group:
                continue
            out.append({"token": token, "name": name or token.upper(),
                        "group": group})
        return out


class TagDelegate(QStyledItemDelegate):

    def __init__(self, tags_fn, apply_fn, parent=None):
        super().__init__(parent)
        self.tags_fn = tags_fn
        self.apply_fn = apply_fn

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.addItem("") 
        combo.addItems(self.tags_fn())
        combo.lineEdit().setPlaceholderText("tag or new…")
        return combo

    def setEditorData(self, editor, index):
        current = index.data(Qt.EditRole) or ""
        position = editor.findText(current)
        if position >= 0:
            editor.setCurrentIndex(position)
        else:
            editor.setCurrentText(current)
        editor.lineEdit().selectAll()

    def setModelData(self, editor, model, index):
        self.apply_fn(editor.currentText(), index.row())


class ChoiceDelegate(QStyledItemDelegate):

    def __init__(self, choices, master_label_fn, parent=None):
        super().__init__(parent)
        self.choices = choices
        self.master_label_fn = master_label_fn

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItem(f"Master ({self.master_label_fn()})", None)
        for value, label in self.choices:
            combo.addItem(label, value)
        return combo

    def setEditorData(self, editor, index):
        current = index.data(Qt.UserRole)
        pos = editor.findData(current)
        editor.setCurrentIndex(max(0, pos))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentData(), Qt.EditRole)


class LogoLabel(QLabel):

    LOGO_MAX = QSize(250, 50)

    FILENAMES = ("logo.png", "logo.jpg", "logo.jpeg", "logo.webp", "logo.bmp")

    FOLDERS = (Path(__file__).resolve().parent.parent,
               Path(__file__).resolve().parent)

    problem = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setFixedHeight(self.LOGO_MAX.height())
        self.refresh()

    def _source(self) -> Path | None:
        for folder in self.FOLDERS:
            for name in self.FILENAMES:
                candidate = folder / name
                if candidate.is_file():
                    return candidate
        return None

    def refresh(self) -> None:
        source = self._source()
        if source is not None:
            image = QPixmap(str(source))
            if image.isNull():
                self.problem.emit(f"Logo could not be read: {source.name}")
            else:
                self.setPixmap(self._fitted(image))
                self.setToolTip(str(source))
                return
        self.setPixmap(self._built_in())
        self.setToolTip(
            f"Drop a {self.FILENAMES[0]} into\n{self.FOLDERS[0]}\nto use your "
            f"own logo — it is scaled to fit "
            f"{self.LOGO_MAX.width()}x{self.LOGO_MAX.height()}.")

    def _fitted(self, image: QPixmap) -> QPixmap:
        ceiling = self.devicePixelRatioF() or 1.0
        limit = QSize(int(self.LOGO_MAX.width() * ceiling),
                      int(self.LOGO_MAX.height() * ceiling))
        if image.width() > limit.width() or image.height() > limit.height():
            image = image.scaled(limit, Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation)
        image.setDevicePixelRatio(max(
            image.width() / self.LOGO_MAX.width(),
            image.height() / self.LOGO_MAX.height(), 1.0))
        return image

    def _built_in(self) -> QPixmap:
        ratio = self.devicePixelRatioF()
        width, height = self.LOGO_MAX.width(), self.LOGO_MAX.height()
        pixmap = QPixmap(int(width * ratio), int(height * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        bars = (0.30, 0.58, 0.92, 1.00, 0.74, 0.46, 0.26)
        accent = theme.colour("peak")
        bar_w, gap = 4.0, 3.0
        x = 0.0
        for amplitude in bars:
            bar_h = max(3.0, height * 0.72 * amplitude)
            painter.fillRect(QRectF(x, (height - bar_h) / 2, bar_w, bar_h),
                             accent)
            x += bar_w + gap

        painter.setPen(QPen(theme.colour("text")))
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() * 0.95))
        font.setBold(True)
        font.setLetterSpacing(QFont.PercentageSpacing, 112)
        painter.setFont(font)
        painter.drawText(QRectF(x + 8, 0, width - x - 8, height),
                         Qt.AlignLeft | Qt.AlignVCenter, "SESSION\nPREP")
        painter.end()
        return pixmap

class LoudnessWindow(QDialog):

    VIEWS = [
        ("Peak, loudest first", "peak"),
        ("LUFS, loudest first", "lufs"),
        ("Flagged only, in list order", "flagged"),
    ]

    COLUMNS = ["#", "File", "Peak dBFS", "True peak", "LUFS", "Flags", "Tag"]
    C_ROW, C_FILE, C_PEAK, C_TP, C_LUFS, C_FLAGS, C_TAG = range(7)

    rowPicked = Signal(int)

    def __init__(self, model: JobModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Loudness")
        self.setWindowFlag(Qt.Window, True)
        self.setModal(False)
        self.resize(720, 520)

        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Show"))
        self.view_combo = QComboBox()
        for label, value in self.VIEWS:
            self.view_combo.addItem(label, value)
        self.view_combo.currentIndexChanged.connect(self.refresh)
        top.addWidget(self.view_combo)
        top.addStretch(1)
        self.count_label = QLabel("")
        top.addWidget(self.count_label)
        layout.addLayout(top)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self.C_FILE, QHeaderView.Stretch)
        header.setSectionResizeMode(self.C_FLAGS, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._picked)
        layout.addWidget(self.table, 1)

        layout.addWidget(QLabel(
            "Ranked on the figures already measured. Picking a row selects "
            "that file in the main window; the running order is left alone."))

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(200)
        self._refresh_timer.timeout.connect(self.refresh)
        for signal in (model.modelReset, model.rowsInserted,
                       model.rowsRemoved, model.dataChanged):
            signal.connect(self._queue_refresh)
        self.refresh()

    def _queue_refresh(self, *_):
        self._refresh_timer.start()

    def _rows(self) -> list[int]:
        view = self.view_combo.currentData()
        if view == "flagged":
            return self.model.flagged_rows()

        read = ((lambda a: a.peak_dbfs) if view == "peak"
                else (lambda a: a.loudness_lufs))
        floor = -200.0 if view == "peak" else -math.inf
        measured, unmeasured = [], []
        for row, job in enumerate(self.model.jobs):
            value = (read(job.analysis)
                     if job.analysis is not None and not job.analysing
                     else -math.inf)
            if value > floor:
                measured.append((value, row))
            else:
                unmeasured.append(row)
        measured.sort(key=lambda pair: (-pair[0], pair[1]))
        return [row for _, row in measured] + unmeasured

    def refresh(self, *_):
        rows = self._rows()
        metric = (COL_LUFS if self.view_combo.currentData() == "lufs"
                  else COL_PEAK)
        value_column = self.C_LUFS if metric == COL_LUFS else self.C_PEAK
        extreme_columns = {self.C_FILE: metric, value_column: metric}
        self.table.setRowCount(len(rows))
        for line, row in enumerate(rows):
            job = self.model.jobs[row]
            a = job.analysis
            fields = {
                self.C_ROW: str(row + 1),
                self.C_FILE: job.info.path.name,
                self.C_PEAK: self._db(a.peak_dbfs if a else None, -200.0),
                self.C_TP: self._db(a.true_peak_dbfs if a else None, -math.inf),
                self.C_LUFS: self._db(a.loudness_lufs if a else None, -math.inf),
                self.C_FLAGS: " · ".join(flag_tokens(a)) if a else "",
                self.C_TAG: job.tag,
            }
            for column, text in fields.items():
                item = QTableWidgetItem(text)
                if column in (self.C_ROW, self.C_PEAK, self.C_TP, self.C_LUFS):
                    item.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
                marked = self.model.extreme_colour(
                    row, extreme_columns.get(column, -1))
                if marked is not None:
                    item.setForeground(marked)
                elif column == self.C_FLAGS and text:
                    item.setForeground(theme.colour("warning"))
                item.setData(Qt.UserRole, row)
                self.table.setItem(line, column, item)
        self.table.resizeColumnToContents(self.C_ROW)
        self.count_label.setText(
            f"{len(rows)} of {len(self.model.jobs)} files"
            if self.view_combo.currentData() == "flagged"
            else f"{len(rows)} files")

    @staticmethod
    def _db(value, floor) -> str:
        if value is None or value <= floor:
            return "—"
        return f"{value:.1f}"

    def _picked(self):
        items = self.table.selectedItems()
        if items:
            self.rowPicked.emit(int(items[0].data(Qt.UserRole)))


class ColumnFitter:

    def __init__(self, table: "FileTableView") -> None:
        self.table = table
        self._fitting = False
        self._natural: dict[int, int] | None = None
        self._heading: dict[int, int] = {}

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(self.COLUMN_MIN)
        header.setSectionsMovable(True)
        header.setResizeContentsPrecision(40)
        table.resized.connect(self._table_resized)

    def remeasure(self) -> None:
        self._natural = None
        self.fit()

    FLEX_COLUMNS = (COL_FILE, COL_NAME, COL_ONAME, COL_STATUS)
    FLEX_MIN = 120              
    FLEX_NATURAL_CAP = 460      
    COLUMN_PADDING = 18
    COLUMN_MIN = 68             
    HEADING_PADDING = 16        

    @staticmethod
    def _shrink(widths: dict[int, int], floors: dict[int, int],
                deficit: int) -> int:
        while deficit > 0:
            spare = {c: widths[c] - floors[c]
                     for c in widths if widths[c] > floors[c]}
            total = sum(spare.values())
            if total <= 0:
                break
            removed = 0
            for column, room in spare.items():
                cut = min(room, int(deficit * room / total))
                widths[column] -= cut
                removed += cut
            deficit -= removed
            if removed == 0:
                for column in sorted(spare, key=lambda c: -widths[c]):
                    if deficit <= 0:
                        break
                    widths[column] -= 1
                    deficit -= 1
                break
        return max(0, deficit)

    def fit(self):
        header = self.table.horizontalHeader()
        model = self.table.model()
        if model is None:
            return
        columns = model.columnCount()
        self._fitting = True
        try:
            if self._natural is None:
                self.table.resizeColumnsToContents()
                self._natural = {
                    c: max(self.COLUMN_MIN,
                           header.sectionSize(c) + self.COLUMN_PADDING)
                    for c in range(columns)}
                metrics = QFontMetrics(header.font())
                self._heading = {
                    c: min(natural_width,
                           max(self.COLUMN_MIN,
                               metrics.horizontalAdvance(
                                   str(model.headerData(
                                       c, Qt.Horizontal, Qt.DisplayRole) or ""))
                               + self.HEADING_PADDING))
                    for c, natural_width in self._natural.items()}
            natural = self._natural
            heading = self._heading

            visible = [c for c in range(columns) if not header.isSectionHidden(c)]
            if not visible:
                return
            flex = [c for c in self.FLEX_COLUMNS if c in visible]
            available = max(1, self.table.viewport().width())

            widths = {c: natural[c] for c in visible}
            for column in flex:
                widths[column] = max(
                    self.FLEX_MIN, min(natural[column], self.FLEX_NATURAL_CAP))

            total = sum(widths.values())
            if total > available:
                self._shrink(
                    widths,
                    {c: (self.FLEX_MIN if c in flex else heading[c])
                     for c in widths},
                    total - available)
            elif flex:
                surplus = available - total
                wanted = sum(widths[c] for c in flex) or 1
                given = 0
                for column in flex[:-1]:
                    share = int(surplus * widths[column] / wanted)
                    widths[column] += share
                    given += share
                widths[flex[-1]] += surplus - given

            for column, width in widths.items():
                header.resizeSection(column, width)
        finally:
            self._fitting = False

    def _table_resized(self):
        if not self._fitting:
            self.fit()


class PresetStore:

    NONE_LABEL = "(unsaved)"

    FIELDS = (
        ("samplerate",         "master_box.rate_combo",        "data",      None),
        ("bit_depth",          "master_box.depth_combo",       "data",      None),
        ("dither",             "master_box.dither_combo",      DitherMode,  "shaped"),
        ("mono_method",        "master_box.mono_method_combo", MonoMethod,  "average"),
        ("threshold_db",       "master_box.threshold_spin",    "value",     -60.0),
        ("phase_threshold_db", "master_box.phase_spin",        "value",     -40.0),
        ("auto_mono",          "master_box.auto_mono_check",   "check",     True),
        ("prevent_clipping",   "master_box.clip_check",        "check",     True),
        ("preserve_chunks",    "master_box.chunk_check",       "check",     True),
        ("normalise_mode",     "normalise.norm_combo",    NormaliseMode, "none"),
        ("normalise_target",   "normalise.norm_target",   "value",   -18.0),
        ("ceiling_on",         "normalise.ceiling_check", "check",   True),
        ("ceiling_dbtp",       "normalise.ceiling_spin",  "value",   -1.0),
        ("normalise_relative", "normalise.relative_check","check",   False),
    )

    def __init__(self, settings: QSettings, owner, combo: QComboBox,
                 delete_button: QPushButton) -> None:
        self.settings = settings
        self.owner = owner        
        self.combo = combo
        self.delete_button = delete_button

    def _widget(self, path: str):
        target = self.owner
        for part in path.split("."):
            target = getattr(target, part)
        return target

    def _widgets(self):
        return [self._widget(attr) for _k, attr, _kind, _d in self.FIELDS]

    def collect(self) -> dict:
        data = {}
        for key, attr, kind, _default in self.FIELDS:
            widget = self._widget(attr)
            if kind == "value":
                data[key] = widget.value()
            elif kind == "check":
                data[key] = widget.isChecked()
            elif kind == "data":
                data[key] = widget.currentData()
            else:                                  
                data[key] = widget.currentData().value
        return data

    def apply(self, data: dict) -> str:
        problem = ""
        widgets = self._widgets()
        for widget in widgets:
            widget.blockSignals(True)
        try:
            for key, attr, kind, default in self.FIELDS:
                widget = self._widget(attr)
                value = data.get(key, default)
                if kind == "value":
                    widget.setValue(float(value))
                elif kind == "check":
                    widget.setChecked(bool(value))
                elif kind == "data":
                    position = widget.findData(value)
                    if position >= 0:
                        widget.setCurrentIndex(position)
                else:
                    position = widget.findData(kind(value))
                    if position >= 0:
                        widget.setCurrentIndex(position)
        except (ValueError, TypeError) as exc:
            problem = f"Preset could not be applied: {exc}"
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        return problem

    def all(self) -> dict:
        raw = self.settings.value("presets", "")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (ValueError, TypeError):
            return {}

    def refresh(self, select: str = "") -> None:
        presets = self.all()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(self.NONE_LABEL, "")
        for name in sorted(presets, key=str.lower):
            self.combo.addItem(name, name)
        position = self.combo.findData(select) if select else 0
        self.combo.setCurrentIndex(max(0, position))
        self.combo.blockSignals(False)
        self.delete_button.setEnabled(bool(presets))

    def chosen(self) -> str:
        name = self.combo.currentData()
        if not name:
            return ""
        presets = self.all()
        if name not in presets:
            return ""
        return self.apply(presets[name]) or f"Applied preset “{name}”"

    def save(self, parent) -> str:
        current = self.combo.currentData() or ""
        name, ok = QInputDialog.getText(parent, "Save preset",
                                        "Preset name:", text=current)
        name = name.strip()
        if not ok or not name or name == self.NONE_LABEL:
            return ""
        presets = self.all()
        if name in presets:
            reply = QMessageBox.question(
                parent, "Save preset", f"Replace the existing “{name}”?",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return ""
        presets[name] = self.collect()
        self.settings.setValue("presets", json.dumps(presets))
        self.refresh(select=name)
        return f"Saved preset “{name}”"

    def delete(self, parent) -> str:
        name = self.combo.currentData()
        presets = self.all()
        if not name or name not in presets:
            return ""
        reply = QMessageBox.question(parent, "Delete preset",
                                     f"Delete “{name}”?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return ""
        del presets[name]
        self.settings.setValue("presets", json.dumps(presets))
        self.refresh()
        return f"Deleted preset “{name}”"



class PreviewPane(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.player = Player()
        self.pool = QThreadPool.globalInstance()
        self.signals = PreviewSignals()
        self.signals.done.connect(self._loaded)
        self._token = 0
        self._job = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.waveform = WaveformView()
        self.waveform.seeked.connect(self._seek)
        self.waveform.scrubbed.connect(self._seek)
        layout.addWidget(self.waveform, 1)

        transport = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        self.side_check = QCheckBox("Show side (L−R)")
        self.side_check.toggled.connect(self.waveform.set_show_side)
        self.zoom_combo = QComboBox()
        for value, label in ZOOM_CHOICES:
            self.zoom_combo.addItem(label, value)
        self.zoom_combo.currentIndexChanged.connect(
            lambda: self.waveform.set_zoom(self.zoom_combo.currentData()))

        self.message = QLabel("")
        transport.addWidget(self.play_btn)
        transport.addWidget(self.stop_btn)
        transport.addWidget(self.side_check)
        transport.addWidget(QLabel("Zoom"))
        transport.addWidget(self.zoom_combo)
        transport.addWidget(self.message, 1)
        layout.addLayout(transport)

        if not self.player.available:
            self.play_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.message.setText(
                f"Playback unavailable: {self.player.unavailable_reason}")

        self.tick = QTimer(self)
        self.tick.setInterval(33)
        self.tick.timeout.connect(self._update_playhead)
        self.tick.start()

    @property
    def job(self):
        return self._job

    def show_job(self, job) -> None:
        self.stop()
        self._job = job
        self._token += 1
        self.waveform.set_envelope(None, f"Loading {job.info.path.name}…")
        self.signals.current = self._token
        self.pool.start(PreviewTask(self._token, job, self.signals))

    def clear(self) -> None:
        self.stop()
        self._job = None
        self._token += 1
        self.waveform.set_envelope(None)

    def refresh_theme(self) -> None:
        self.waveform.refresh_theme()

    def shutdown(self) -> None:
        self.tick.stop()
        self.player.shutdown()
        self._token += 1

    @Slot(int, object, object, int, str)
    def _loaded(self, token: int, job, data, rate: int, error: str) -> None:
        if token != self._token:
            return
        if error or data is None:
            self.waveform.set_envelope(None, f"Could not read: {error}")
            return
        label = (f"{job.info.path.name} — {job.info.channels} ch, "
                 f"{job.info.samplerate / 1000:g} kHz, {job.info.subtype}")
        play_rate = playable_samplerate(job.info.samplerate)
        if play_rate != job.info.samplerate:
            label += f"  (played at {play_rate / 1000:g} kHz)"
        if data.provisional:
            label += "   (scanning…)"
        self.waveform.set_envelope(data, label)
        self.side_check.setEnabled(data.channels >= 2)
        self.message.setText(self.player.load(job.info.path, play_rate))

    def toggle_play(self) -> None:
        if not self.player.available or self.player.frames == 0:
            return
        error = self.player.toggle()
        if error:
            self.message.setText(error)
        self.play_btn.setText("Pause" if self.player.playing else "Play")

    def stop(self) -> None:
        self.player.stop()
        self.player.seek(0)
        self.waveform.set_position(0)
        self.play_btn.setText("Play")

    def _seek(self, frame: int) -> None:
        was_playing = self.player.playing
        self.player.seek(frame)
        if was_playing:
            self.player.play(frame)

    def _update_playhead(self) -> None:
        if self.player.frames:
            self.waveform.set_position(self.player.position)
            if not self.player.playing and self.play_btn.text() == "Pause":
                self.play_btn.setText("Play")



class NormaliseBox(QGroupBox):

    changed = Signal()

    def __init__(self, master: MasterSettings, model: JobModel,
                 parent=None) -> None:
        super().__init__("Normalisation", parent)
        self.master = master
        self.model = model
        outer = QVBoxLayout(self)
        row = QHBoxLayout()
        outer.addLayout(row)

        self.norm_combo = QComboBox()
        for mode in NormaliseMode:
            self.norm_combo.addItem(mode.label, mode)
        self.norm_combo.currentIndexChanged.connect(self._settings_changed)
        row.addWidget(QLabel("Mode"))
        row.addWidget(self.norm_combo)

        self.norm_target = QDoubleSpinBox()
        self.norm_target.setRange(-60.0, 12.0)
        self.norm_target.setSingleStep(0.5)
        self.norm_target.setValue(-18.0)
        self.norm_target.valueChanged.connect(self._settings_changed)
        row.addWidget(QLabel("Target"))
        row.addWidget(self.norm_target)

        self.ceiling_check = QCheckBox("Max true peak")
        self.ceiling_check.setChecked(True)
        self.ceiling_check.toggled.connect(self._settings_changed)
        row.addWidget(self.ceiling_check)

        self.ceiling_spin = QDoubleSpinBox()
        self.ceiling_spin.setRange(-12.0, 0.0)
        self.ceiling_spin.setSingleStep(0.1)
        self.ceiling_spin.setValue(-1.0)
        self.ceiling_spin.setSuffix(" dBTP")
        self.ceiling_spin.valueChanged.connect(self._settings_changed)
        row.addWidget(self.ceiling_spin)

        self.relative_check = QCheckBox("To loudest track")
        self.relative_check.setToolTip(
            "References the loudest file, and applies the same gain change to all others.")
        self.relative_check.toggled.connect(self._settings_changed)
        row.addWidget(self.relative_check)

        self.norm_note = QLabel("")
        row.addWidget(self.norm_note, 1)
        self.norm_report = QLabel("")
        self.norm_report.setTextFormat(Qt.RichText)
        outer.addWidget(self.norm_report)

        self._settings_changed()

    def restyle(self) -> None:
        self.update_report()
        self.norm_report.setStyleSheet(
            f"color: {theme.colour('muted').name()};")

    def master_label(self) -> str:
        return self.master.normalise.label

    def _settings_changed(self, *_):
        mode = self.norm_combo.currentData()
        enabled = mode is not NormaliseMode.NONE
        self.norm_target.setEnabled(enabled)
        self.ceiling_check.setEnabled(enabled)
        self.ceiling_spin.setEnabled(enabled and self.ceiling_check.isChecked())
        self.relative_check.setEnabled(enabled)
        self.norm_target.setSuffix(f" {mode.unit}" if mode.unit else "")
        if mode is NormaliseMode.PPM_NORDIC:
            self.norm_note.setText("0 PPM Nordic = -18 dBFS alignment level (EBU R68)")
        else:
            self.norm_note.setText("")
        self.master.normalise = NormaliseSpec(
            mode, self.norm_target.value(),
            self.ceiling_spin.value() if self.ceiling_check.isChecked() else None,
            relative=self.relative_check.isChecked())
        self.model.refresh_all()
        self.update_report()
        self.changed.emit()

    def master_rows(self) -> list[int]:
        return [row for row, job in enumerate(self.model.jobs)
                if job.norm_override is None]

    def plan(self):
        spec = self.master.normalise
        if spec.mode is NormaliseMode.NONE or not spec.relative:
            return None, []
        rows = self.master_rows()
        if not rows:
            return None, []
        plan = plan_relative([self.model.jobs[r].analysis for r in rows],
                             spec, self.master.prevent_clipping)
        return plan, rows

    def _predicted(self) -> list[tuple[int, float, float]]:
        spec = self.master.normalise
        if spec.mode is NormaliseMode.NONE:
            return []
        plan, rows = self.plan()
        if spec.relative and plan is None:
            return []

        target = target_level_db(spec)
        out = []
        for row in (rows or self.master_rows()):
            analysis = self.model.jobs[row].analysis
            level = measured_level_db(analysis, spec.mode)
            if level == -math.inf:
                continue
            held = ""
            if spec.relative:
                gain = plan.gain_db
            else:
                gain = target - level
                if (spec.ceiling_dbtp is not None and analysis is not None
                        and analysis.true_peak_dbfs > -math.inf
                        and analysis.true_peak_dbfs + gain > spec.ceiling_dbtp):
                    gain = spec.ceiling_dbtp - analysis.true_peak_dbfs
                    held = "Max true peak"
            if analysis is None or analysis.peak_dbfs <= -200.0:
                continue
            peak_after = analysis.peak_dbfs + gain
            if self.master.prevent_clipping and peak_after > 0.0:
                peak_after, held = 0.0, "Clip guard"
            out.append((row, gain, peak_after, held))
        return out

    def update_report(self):
        spec = self.master.normalise
        if spec.mode is NormaliseMode.NONE or not self.model.jobs:
            self.norm_report.setText("")
            return

        waiting = sum(1 for job in self.model.jobs if job.analysing)
        predicted = self._predicted()
        if not predicted:
            self.norm_report.setText(
                "Loudest after normalisation: measuring…" if waiting else
                "Loudest after normalisation: nothing measurable in this list")
            return

        row, gain, peak_after, held = max(predicted, key=lambda item: item[2])
        clipping = peak_after > 0.0
        colour = theme.colour("loudest" if clipping else "quietest").name()
        figure = f"peak {peak_after:+.1f} dBFS"
        if clipping:
            figure += ", will be clipped"
        elif held:
            figure += f", held by {held}"
        text = ("Loudest after normalisation: "
                + html.escape(self.model.jobs[row].info.path.name)
                + f" — <span style='color:{colour}'>{figure}</span>")
        if spec.relative:
            plan, _ = self.plan()
            count = len(predicted)
            text += (f"   ·   {gain:+.1f} dB applied to "
                     + (f"all {count} files" if count != 1 else "the one file"))
            if plan is not None and plan.held_by:
                text += f", held by {plan.held_by}"
        else:
            text += "   ·   each file normalised on its own"
        if waiting:
            text += f"   ·   {waiting} still being analysed"
        self.norm_report.setText(text)


class NamingBox(QGroupBox):

    status = Signal(str)

    def __init__(self, master: MasterSettings, model: JobModel,
                 table: "FileTableView", settings: QSettings,
                 parent=None) -> None:
        super().__init__("Output naming", parent)
        self.master, self.model = master, model
        self.table, self.settings = table, settings
        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        self.rename_edit = QLineEdit(DEFAULT_PATTERN)
        self.rename_edit.setPlaceholderText(DEFAULT_PATTERN)
        self.rename_edit.setToolTip(
            "Renames the output files.\n"
            "Available tokens:\n\n  "
            + "\n  ".join(f"{tok:<12} {desc}" for tok, desc in RENAME_TOKENS))
        self.rename_edit.textChanged.connect(self._rename_changed)
        row.addWidget(self.rename_edit, 1)
        reset = QPushButton("Reset")
        reset.clicked.connect(lambda: self.rename_edit.setText(DEFAULT_PATTERN))
        row.addWidget(reset)
        layout.addLayout(row)

        self.number_check = QCheckBox("Number files in list order (01_name)")
        self.number_check.toggled.connect(self._rename_changed)
        layout.addWidget(self.number_check)

        self.rename_preview = QLabel("")
        self.rename_preview.setToolTip("Preview using the first file in the list.")
        layout.addWidget(self.rename_preview)

        actions = QHBoxLayout()
        for text, slot, tip in (
            ("Auto re-name", self.suggest,
             ""),
            ("Auto sort", self.sort_by_group,
             ""),
            ("Trim", self.run_trim,
             ""),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self._load_tags()

    def _rename_changed(self, *_):
        self.master.rename_pattern = self.rename_edit.text() or DEFAULT_PATTERN
        self.master.number_names = self.number_check.isChecked()
        self.model.refresh_all()
        self.update_preview()

    def update_preview(self):
        if not self.model.jobs:
            self.rename_preview.setText("")
            return
        stem = self.model.jobs[0].output_stem(self.master, 0,
                                              len(self.model.jobs))
        suffix = self.model.jobs[0].info.path.suffix
        clashes = len(self.model.clashing_rows())
        text = f"e.g. {stem}{suffix}"
        if clashes:
            text += f"   ·   {clashes} name clash{'es' if clashes != 1 else ''}"
        self.rename_preview.setText(text)

    def apply_tag(self, tag: str, row: int):
        rows = selected_rows(self.table)
        if row not in rows:
            rows = [row]
        before = set(self.model.tags)
        self.model.set_tag(rows, tag)
        if set(self.model.tags) != before:
            self._save_tags()
        self.update_preview()

    def set_tag_dialog(self):
        rows = selected_rows(self.table)
        if not rows:
            return
        existing = list(self.model.tags)
        current = self.model.jobs[rows[0]].tag
        items = [""] + existing
        tag, ok = QInputDialog.getItem(
            self, "Set tag",
            f"Tag for {len(rows)} file{'s' if len(rows) != 1 else ''} "
            f"(appended as _tag; blank clears):",
            items, items.index(current) if current in items else 0, True)
        if ok:
            self.apply_tag(tag, rows[0])

    def _load_tags(self):
        stored = self.settings.value("tags", None)
        if stored is None:
            tags = list(DEFAULT_TAGS)
        elif isinstance(stored, str):
            tags = [stored]
        else:
            tags = list(stored)
        for tag in tags:
            self.model.register_tag(tag)

    def _save_tags(self):
        self.settings.setValue("tags", list(self.model.tags))

    def _trim_target_rows(self) -> list[int]:
        return selected_rows(self.table) or list(range(len(self.model.jobs)))

    def run_trim(self):
        rows = self._trim_target_rows()
        if not rows:
            return
        total = len(self.model.jobs)
        names = [self.model.jobs[r].base_stem(self.master, r, total) for r in rows]
        dialog = TrimDialog(self, names)
        if dialog.exec() != QDialog.Accepted:
            return

        changed = skipped = 0
        for row in rows:
            job = self.model.jobs[row]
            current = job.base_stem(self.master, row, total)
            result = sanitise_stem(dialog.transform(current))
            if not result:
                skipped += 1
                continue
            if result != current:
                job.name_override = result
                changed += 1
        self.model.refresh_all()
        self.update_preview()
        note = (f"Renamed {changed} of {len(rows)} "
                f"file{'s' if len(rows) != 1 else ''}")
        if skipped:
            note += f" · {skipped} left unchanged (would have been empty)"
        note += " · Reset overrides to undo"
        self.status.emit(note)

    def suggest(self):
        if not self.model.jobs:
            return
        rows = selected_rows(self.table) or list(range(len(self.model.jobs)))
        results = self.model.classify()

        preview = [(self.model.jobs[row].info.path.stem,
                    results[row].name if results[row] else None,
                    results[row].group if results[row] else None)
                   for row in rows]

        dialog = SuggestDialog(self, preview, list(self.model.tags))
        if dialog.exec() != QDialog.Accepted:
            return
        edits = dialog.edited()

        named = tagged = 0
        for (row, (stem, name, tag)) in zip(rows, edits):
            job = self.model.jobs[row]
            if dialog.skip_edited.isChecked() and job.name_override:
                continue
            if dialog.set_names.isChecked() and name:
                job.name_override = sanitise_stem(name) or None
                named += 1
            if dialog.set_tags.isChecked() and tag:
                job.tag = sanitise_tag(tag)
                self.model.register_tag(job.tag)
                tagged += 1
        self._save_tags()
        self.model.refresh_all()
        self.update_preview()
        self.status.emit(
            f"Suggested {named} name{'s' if named != 1 else ''} and "
            f"{tagged} tag{'s' if tagged != 1 else ''} · Reset overrides to undo")

        self._offer_to_learn(rows, results, edits)

    def _offer_to_learn(self, rows, results, edits):
        prefix = classify.common_prefix(
            [classify.tokenise(job.info.path.stem) for job in self.model.jobs])

        proposals = {}
        for (row, (stem, name, tag)) in zip(rows, edits):
            if not tag:
                continue
            before = results[row]
            corrected = before is None or classify.sanitise_tag_like(tag) != before.group
            if not corrected:
                continue
            for token in classify.teachable_tokens(stem, prefix):
                key = token.upper()
                if key in proposals:
                    continue
                base = re.sub(r"\s+\d+$", "", name or token.upper()).strip()
                proposals[key] = {"token": token,
                                  "name": base or token.upper(),
                                  "group": sanitise_tag(tag),
                                  "known": False}
                break

        if not proposals:
            return

        existing_rules = {}
        for rule in classify.RULES:
            if rule.name not in existing_rules:
                existing_rules[rule.name] = rule.group

        dialog = LearnDialog(self, list(proposals.values()),
                             list(self.model.tags), existing_rules)
        if dialog.exec() != QDialog.Accepted:
            return
        self._store_learned(dialog.entries())

    def show_config_warnings(self) -> None:
        warnings = getattr(classify.TABLES, "warnings", [])
        if not warnings:
            return
        first = warnings[0]
        if len(warnings) > 1:
            first += f"  (+{len(warnings) - 1} more)"
        self.status.emit(f"config.yaml: {first}")

    def _store_learned(self, entries: list) -> None:
        classify.append_learned_rules_to_config(entries)
        self.status.emit(
            f"Saved {len(entries)} rule{'s' if len(entries) != 1 else ''} to config.yaml")
        self.show_config_warnings()

    def sort_by_group(self):
        moved = self.model.sort_by_group()
        self.update_preview()
        if moved:
            self.status.emit(
                "Sorted into instrument order — drums, percussion, bass, "
                "guitars, keys, orchestra, BVs, lead vocal")
        else:
            self.status.emit("Already in instrument order")

    def sort_by_name(self):
        self.model.sort_by_name()
        self.update_preview()



class MasterBox(QGroupBox):
    changed = Signal()

    def __init__(self, master: MasterSettings, model: JobModel,
                 parent=None) -> None:
        super().__init__("Master settings", parent)
        self.master, self.model = master, model
        outer = QVBoxLayout(self)

        presets = QHBoxLayout()
        presets.addWidget(QLabel("Preset"))
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(180)
        self.preset_combo.setToolTip(
            "Saved presets of the master and normalisation settings.\n"
            "Naming and tags are not included.")
        presets.addWidget(self.preset_combo)
        self.preset_save = QPushButton("Save")
        presets.addWidget(self.preset_save)
        self.preset_delete = QPushButton("Delete")
        presets.addWidget(self.preset_delete)
        presets.addStretch(1)
        self.preset_row = presets
        outer.addLayout(presets)

        row = QHBoxLayout()
        outer.addLayout(row)

        left = QFormLayout()
        self.rate_combo = QComboBox()
        for value, label in RATE_CHOICES:
            self.rate_combo.addItem(label, value)
        self.rate_combo.setCurrentIndex(self.rate_combo.findData(44100))
        self.rate_combo.currentIndexChanged.connect(self._changed)
        left.addRow("Sample rate", self.rate_combo)

        self.depth_combo = QComboBox()
        for value, label in DEPTH_CHOICES:
            self.depth_combo.addItem(label, value)
        self.depth_combo.setCurrentIndex(self.depth_combo.findData(24))
        self.depth_combo.currentIndexChanged.connect(self._changed)
        left.addRow("Bit depth", self.depth_combo)
        row.addLayout(left)

        mid = QFormLayout()
        self.dither_combo = QComboBox()
        self.dither_combo.addItem("Noise-shaped TPDF (best)", DitherMode.SHAPED)
        self.dither_combo.addItem("Flat TPDF", DitherMode.TPDF)
        self.dither_combo.addItem("None (truncate)", DitherMode.NONE)
        self.dither_combo.currentIndexChanged.connect(self._changed)
        mid.addRow("Dither", self.dither_combo)

        self.mono_method_combo = QComboBox()
        for method in MonoMethod:
            self.mono_method_combo.addItem(method.label, method)
        self.mono_method_combo.currentIndexChanged.connect(self._changed)
        mid.addRow("Mono fold", self.mono_method_combo)
        row.addLayout(mid)

        right = QFormLayout()
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(-140.0, -20.0)
        self.threshold_spin.setSingleStep(6.0)
        self.threshold_spin.setValue(-60.0)
        self.threshold_spin.setSuffix(" dB")
        self.threshold_spin.valueChanged.connect(self._thresholds_changed)
        right.addRow("Fake-stereo threshold", self.threshold_spin)

        self.phase_spin = QDoubleSpinBox()
        self.phase_spin.setRange(-120.0, -6.0)
        self.phase_spin.setSingleStep(4.0)
        self.phase_spin.setValue(-40.0)
        self.phase_spin.setSuffix(" dBFS")
        self.phase_spin.valueChanged.connect(self._thresholds_changed)
        right.addRow("Phase cancel threshold", self.phase_spin)

        checks = QHBoxLayout()
        self.auto_mono_check = QCheckBox("Auto mono")
        self.auto_mono_check.setChecked(True)
        self.auto_mono_check.toggled.connect(self._changed)
        self.clip_check = QCheckBox("Clip guard")
        self.clip_check.setChecked(True)
        self.clip_check.setToolTip(
            "Resampling can turn inter-sample peaks into real overs.\n"
            "Trims gain by the smallest amount needed to avoid clipping.")
        self.clip_check.toggled.connect(self._changed)
        self.chunk_check = QCheckBox("Keep loops/markers")
        self.chunk_check.setChecked(True)
        self.chunk_check.setToolTip(
            "Carry smpl/cue/inst chunks across, rescaling loop points\n"
            "to the new sample rate.")
        self.chunk_check.toggled.connect(self._changed)
        checks.addWidget(self.auto_mono_check)
        checks.addWidget(self.clip_check)
        checks.addWidget(self.chunk_check)
        right.addRow("", checks)
        row.addLayout(right)
        row.addStretch(1)


    def rate_label(self) -> str:
        return self.rate_combo.currentText()


    def depth_label(self) -> str:
        return self.depth_combo.currentText()


    def _changed(self):
        self.master.samplerate = self.rate_combo.currentData()
        self.master.bit_depth = self.depth_combo.currentData()
        self.master.dither = self.dither_combo.currentData()
        self.master.mono_method = self.mono_method_combo.currentData()
        self.master.auto_mono = self.auto_mono_check.isChecked()
        self.master.prevent_clipping = self.clip_check.isChecked()
        self.master.preserve_chunks = self.chunk_check.isChecked()
        self.model.refresh_all()
        self.changed.emit()


    def _thresholds_changed(self, _value: float = 0.0):
        self.master.threshold_db = self.threshold_spin.value()
        self.master.phase_threshold_db = self.phase_spin.value()
        for row, job in enumerate(self.model.jobs):
            if job.analysis is None:
                continue
            job.analysis.redecide(self.master.threshold_db,
                                  self.master.phase_threshold_db)
            self.model.row_changed(row)
        self.changed.emit()



class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1220, 780)
        self.setAcceptDrops(True)

        self.settings = QSettings()
        self.settings.remove("classify_learned")
        self.faults = errors.ErrorBus()
        self.faults.raised.connect(self._report_fault)
        errors.install(self.faults)
        self._density = self.settings.value("density", theme.DEFAULT_DENSITY)
        self.master = MasterSettings()
        self.model = JobModel(self.master)
        self.pool = QThreadPool.globalInstance()
        self.analysis_signals = AnalysisSignals()
        self.analysis_signals.done.connect(self._analysis_done)
        self.worker: ConvertWorker | None = None
        self.out_dir: Path | None = None
        self.drop_root: Path | None = None
        self.loudness_window: LoudnessWindow | None = None

        self._build_ui()
        self._build_actions()
        self._build_menubar()
        self.presets.refresh()
        self.naming.show_config_warnings()

        if not HAVE_NUMBA:
            self.status_label.setText(
                "numba not installed — noise-shaped dither will be slow. "
                "Install it with: pip install numba")

    def _build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        self.table = FileTableView()
        self.table.setModel(self.model)

        self.master_box = MasterBox(self.master, self.model)
        self.master_box.changed.connect(self._update_summary)
        self._build_preset_row(self.master_box.preset_row)
        outer.addWidget(self.master_box)
        lower = QHBoxLayout()
        self.normalise = NormaliseBox(self.master, self.model)
        self.normalise.changed.connect(self._update_summary)
        lower.addWidget(self.normalise, 3)
        self.naming = NamingBox(self.master, self.model, self.table,
                                self.settings)
        self.naming.status.connect(lambda text: self.status_label.setText(text))
        lower.addWidget(self.naming, 2)
        outer.addLayout(lower)

        splitter = QSplitter(Qt.Vertical)

        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setDragDropMode(QAbstractItemView.InternalMove)
        self.table.setDragDropOverwriteMode(False)
        self.table.setDefaultDropAction(Qt.MoveAction)
        self.table.setDropIndicatorShown(True)
        self.table.filesDropped.connect(self._add_paths)
        self.table.verticalHeader().setVisible(True)
        self.table.verticalHeader().setDefaultSectionSize(
            theme.row_height(self._density))
        self.table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.table.setItemDelegateForColumn(
            COL_ORATE, ChoiceDelegate(RATE_CHOICES, self.master_box.rate_label, self))
        self.table.setItemDelegateForColumn(
            COL_ODEPTH, ChoiceDelegate(DEPTH_CHOICES, self.master_box.depth_label, self))
        self.table.setItemDelegateForColumn(
            COL_NORM, ChoiceDelegate(NORM_CHOICES, self.normalise.master_label, self))
        self.table.setItemDelegateForColumn(
            COL_TAG, TagDelegate(lambda: list(self.model.tags),
                                 self.naming.apply_tag, self))
        self.columns = ColumnFitter(self.table)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        self.model.dataChanged.connect(self._model_changed)
        self.model.reordered.connect(self._after_reorder)
        splitter.addWidget(self.table)

        self.preview = PreviewPane()
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(260)
        self.status_label = QLabel("Drop audio files or folders here")
        self.summary_label = QLabel("")
        bottom.addWidget(self.status_label, 1)
        bottom.addWidget(self.progress)
        bottom.addWidget(self.summary_label)
        bottom.addSpacing(16)

        self.output_button = QPushButton("Output folder…")
        self.output_button.setMinimumHeight(34)
        self.output_button.setMinimumWidth(150)
        self.output_button.clicked.connect(self._choose_output)
        bottom.addWidget(self.output_button)

        self.convert_button = QPushButton("Convert")
        self.convert_button.setMinimumHeight(34)
        self.convert_button.setMinimumWidth(150)
        self.convert_button.setDefault(True)
        self.convert_button.setStyleSheet(
            "QPushButton { font-weight: 600; }")
        self.convert_button.clicked.connect(self._start_convert)
        bottom.addWidget(self.convert_button)
        outer.addLayout(bottom)

        self.setCentralWidget(central)

    def _build_preset_row(self, presets) -> None:
        self.presets = PresetStore(self.settings, self,
                                   self.master_box.preset_combo,
                                   self.master_box.preset_delete)
        self.master_box.preset_combo.activated.connect(
            lambda *_: self._after_preset(self.presets.chosen()))
        self.master_box.preset_save.clicked.connect(
            lambda *_: self._after_preset(self.presets.save(self)))
        self.master_box.preset_delete.clicked.connect(
            lambda *_: self._after_preset(self.presets.delete(self)))

        self.logo = LogoLabel(self)
        self.logo.problem.connect(lambda text: self.status_label.setText(text))
        presets.addWidget(self.logo, 0, Qt.AlignRight | Qt.AlignTop)

    def _after_preset(self, message: str) -> None:
        if not message:
            return
        self.master_box._changed()
        self.normalise._settings_changed()
        self.master_box._thresholds_changed()
        self.status_label.setText(message)

    def _after_reorder(self, moved):
        rows = self.model.rows_for(moved)
        selection = QItemSelection()
        for row in rows:
            selection.select(self.model.index(row, 0),
                             self.model.index(row, self.model.columnCount() - 1))
        self.table.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect)
        if rows:
            self.table.scrollTo(self.model.index(rows[0], COL_FILE))
        self.naming.update_preview()

    def _nudge(self, offset: int):
        self.model.move_rows(self._selected_rows(), offset)

    def _model_changed(self, top_left, bottom_right, _roles=None):
        if self.worker is not None and self.worker.isRunning():
            return
        if top_left.column() <= COL_NAME <= bottom_right.column():
            self.naming.update_preview()
        if top_left.column() <= COL_NORM <= bottom_right.column():
            self.normalise.update_report()

    def _build_actions(self):
        self.acts: dict[str, QAction] = {}

        def make(key, text, slot, shortcut=None):
            act = QAction(text, self)
            act.triggered.connect(slot)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            self.addAction(act)
            self.acts[key] = act
            return act

        make("open_files", "Open file", self._add_files, "Ctrl+O")
        make("open_folder", "Open folder", self._add_folder, "Ctrl+Shift+O")
        make("convert", "Convert", self._start_convert, "Ctrl+R")
        make("remove_selected", "Remove selected", self._remove_selected,
             "Backspace")
        make("remove_all", "Remove all", self._clear, "Shift+Backspace")
        make("reset", "Reset overrides", self._reset_overrides)
        make("move_up", "Move up", lambda: self._nudge(-1), "Ctrl+Up")
        make("move_down", "Move down", lambda: self._nudge(1), "Ctrl+Down")
        make("sort", "Sort by name", self.naming.sort_by_name)
        make("suggest", "Auto re-name", self.naming.suggest, "Ctrl+I")
        make("tag", "Set tag", self.naming.set_tag_dialog, "Ctrl+T")
        make("play", "Play/Stop", self.preview.toggle_play, Qt.Key_Space)
        make("loudness", "Loudness", self._show_loudness, "Ctrl+L")
        make("about", f"About {APP_NAME}", self._show_about)

        self.select_flagged_action = make(
            "select_flagged", "Select flagged", self._select_flagged)
        self.select_flagged_action.setEnabled(False)
        self.theme_action = make("theme", "Theme", self._cycle_theme)
        self.density_action = make("density", "Density", self._cycle_density)
        self._update_view_labels()

    def _build_menubar(self):
        """Every command lives here; the shortcuts show beside their entry."""
        bar = self.menuBar()

        def menu(title, keys):
            target = bar.addMenu(title)
            for key in keys:
                if key is None:
                    target.addSeparator()
                else:
                    target.addAction(self.acts[key])
            return target

        menu("&File", ["open_files", "open_folder", None, "convert"])
        menu("&Edit", ["remove_selected", "remove_all", None, "reset", None,
                       "move_up", "move_down", "sort", None, "select_flagged"])
        menu("&Naming", ["suggest", "tag"])
        menu("&View", ["play", None, "loudness", None, "theme", "density"])
        menu("&Help", ["about"])

    def _show_about(self):
        box = QMessageBox(self)
        box.setWindowTitle(f"About {APP_NAME}")
        box.setTextFormat(Qt.RichText)
        box.setTextInteractionFlags(Qt.TextBrowserInteraction)
        box.setText(
            f"<b>{APP_NAME}</b> {__version__}<br>"
            "&copy; 2026 Sandro Giacometti")
        box.setInformativeText(
            "This program comes with ABSOLUTELY NO WARRANTY. It is free "
            "software, and you are welcome to redistribute it under the terms "
            "of the GNU General Public License, version 3 or later."
            "<br><br>"
            '<a href="https://www.gnu.org/licenses/gpl-3.0.html">'
            "View the licence</a>")
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

    @Slot(str, str)
    def _report_fault(self, summary: str, detail: str):
        self.status_label.setText(summary)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Something went wrong")
        box.setText(summary)
        box.setDetailedText(detail)
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

    def _update_view_labels(self):
        mode = self.settings.value("theme", theme.DEFAULT_MODE)
        shown = f"{mode} ({theme.active()})" if mode == "auto" else mode
        self.theme_action.setText(f"Theme: {shown}")
        self.density_action.setText(
            f"Density: {self.settings.value('density', theme.DEFAULT_DENSITY)}")

    def _cycle_theme(self):
        mode = theme.next_mode(self.settings.value("theme", theme.DEFAULT_MODE))
        self.settings.setValue("theme", mode)
        self._repaint_theme()

    def _repaint_theme(self):
        mode = self.settings.value("theme", theme.DEFAULT_MODE)
        theme.apply(QApplication.instance(), mode)
        self.preview.refresh_theme()
        self.model.refresh_all()
        self.logo.refresh()
        self.normalise.restyle()
        if self.loudness_window is not None:
            self.loudness_window.refresh()
        self._update_view_labels()
        self.update()

    def _cycle_density(self):
        self._density = theme.next_density(self._density)
        self.settings.setValue("density", self._density)
        self.table.verticalHeader().setDefaultSectionSize(
            theme.row_height(self._density))
        self._update_view_labels()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()
                 if url.isLocalFile()]
        if paths:
            self._add_paths(paths)
            event.acceptProposedAction()

    def _add_files(self):
        patterns = " ".join(f"*{ext}" for ext in sorted(AUDIO_EXTS))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add audio files", "", f"Audio files ({patterns});;All files (*)")
        if files:
            self._add_paths([Path(f) for f in files])

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add folder")
        if folder:
            self._add_paths([Path(folder)])

    def _add_paths(self, paths: list[Path]):
        collected: list[Path] = []
        roots: list[Path] = []
        for path in paths:
            if path.is_dir():
                roots.append(path)
                found = [p for p in path.rglob("*")
                         if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
                collected += sorted(found, key=lambda p: str(p).lower())
            elif path.is_file() and path.suffix.lower() in AUDIO_EXTS:
                collected.append(path)

        if roots and self.drop_root is None:
            self.drop_root = (roots[0] if len(roots) == 1
                              else Path(os.path.commonpath([str(r) for r in roots])))

        existing = self.model.existing_paths()
        jobs, failed = [], 0
        for path in collected:
            if path in existing:
                continue
            try:
                jobs.append(Job(info=probe(path)))
            except Exception:
                failed += 1

        self.model.add_jobs(jobs)
        for job in jobs:
            self.pool.start(AnalysisTask(job, self.master.threshold_db,
                                         self.master.phase_threshold_db,
                                         self.analysis_signals))
        if jobs:
            self.columns.remeasure()
        note = f"Added {len(jobs)} file{'s' if len(jobs) != 1 else ''}"
        if failed:
            note += f" ({failed} unreadable and skipped)"
        self.status_label.setText(note)
        self._update_summary()

    @Slot(object, object, str)
    def _analysis_done(self, job, result, error: str):
        try:
            row = self.model.jobs.index(job)
        except ValueError:
            return
        job.analysing = False
        job.analysis = result
        if error:
            job.status = f"Analysis failed: {error}"
        self.model.row_changed(row)
        self._update_summary()

    def _selected_rows(self) -> list[int]:
        return selected_rows(self.table)

    def _select_flagged(self):
        rows = self.model.flagged_rows()
        if not rows:
            return
        selection = self.table.selectionModel()
        selection.clearSelection()
        for row in rows:
            selection.select(
                self.model.index(row, 0),
                QItemSelectionModel.Select | QItemSelectionModel.Rows)
        self.table.scrollTo(self.model.index(rows[0], 0))

    def _show_loudness(self):
        if self.loudness_window is None:
            self.loudness_window = LoudnessWindow(self.model, self)
            self.loudness_window.rowPicked.connect(self._select_row)
        self.loudness_window.show()
        self.loudness_window.raise_()
        self.loudness_window.activateWindow()

    def _select_row(self, row: int):
        if not 0 <= row < len(self.model.jobs):
            return
        self.table.selectionModel().select(
            self.model.index(row, 0),
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        self.table.scrollTo(self.model.index(row, COL_FILE))

    def _reset_overrides(self):
        rows = self._selected_rows() or range(len(self.model.jobs))
        self.model.apply_to_rows(list(rows), mono_override=None,
                                 rate_override=None, depth_override=None,
                                 norm_override=None, name_override=None)
        self._update_summary()

    def _remove_selected(self):
        self.model.remove_rows(self._selected_rows())
        self._update_summary()

    def _clear(self):
        self.preview.clear()
        self.model.clear()
        self.drop_root = None
        self._update_summary()

    def _update_summary(self):
        total, mono, flagged = self.model.counts()
        text = f"{total} files · {mono} → mono"
        if flagged:
            text += f" · {flagged} flagged"
        self.summary_label.setText(text)
        self.select_flagged_action.setEnabled(bool(flagged))
        self.naming.update_preview()
        self.normalise.update_report()

    def _selection_changed(self, *_):
        rows = self._selected_rows()
        if not rows:
            return
        if self.model.jobs[rows[0]] is self.preview.job:
            return
        self.preview.show_job(self.model.jobs[rows[0]])

    def _choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if folder:
            self.out_dir = Path(folder)
            self.status_label.setText(f"Output: {self.out_dir}")

    def _start_convert(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            return
        if not self.model.jobs:
            return
        if self.out_dir is None:
            self._choose_output()
            if self.out_dir is None:
                return

        sources = {j.info.path.resolve() for j in self.model.jobs}
        if any(self.out_dir.resolve() == p.parent for p in sources):
            reply = QMessageBox.warning(
                self, "Output folder",
                "The output folder contains some of the source files. Files that "
                "would land on top of a source get '_converted' appended instead. "
                "Continue?",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        master = replace(self.master)

        if (master.normalise.mode is not NormaliseMode.NONE
                and master.normalise.relative):
            waiting = sum(1 for job in self.model.jobs if job.analysing)
            if waiting:
                QMessageBox.information(
                    self, "Normalise to loudest track",
                    f"{waiting} file{'s are' if waiting != 1 else ' is'} still "
                    f"being analysed.\n\nNormalising to the loudest track needs "
                    f"every file measured first. "
                    f"Try again in a moment.")
                return
            plan, _rows = self.normalise.plan()
            if plan is None:
                QMessageBox.warning(
                    self, "Normalise to loudest track",
                    "Nothing in this list can be measured, so there is no "
                    "loudest track to normalise to.\n\nEvery file is either "
                    "silent or set to its own Normalise value.")
                return
            master.normalise = replace(master.normalise,
                                       fixed_gain_db=plan.gain_db)

        jobs = list(enumerate(self.model.jobs))
        self.progress.setVisible(True)
        self.progress.setRange(0, len(jobs))
        self.progress.setValue(0)
        self.convert_button.setText("Cancel")

        self.worker = ConvertWorker(jobs, master, self.out_dir,
                                    self.drop_root, "copy", self)
        self.worker.progress.connect(self._convert_progress)
        self.worker.finished_all.connect(self._convert_finished)
        self.worker.start()

    @Slot(int, int, str)
    def _convert_progress(self, row: int, done: int, message: str):
        self.progress.setValue(done)
        self.model.row_changed(row)
        self.status_label.setText(f"{self.model.jobs[row].info.path.name}: {message}")

    @Slot(int, int, int)
    def _convert_finished(self, ok: int, failed: int, skipped: int):
        self.progress.setVisible(False)
        self.convert_button.setText("Convert")
        parts = [f"{ok} converted"]
        if skipped:
            parts.append(f"{skipped} skipped")
        if failed:
            parts.append(f"{failed} failed")
        self.status_label.setText(" · ".join(parts))
        self.worker = None

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        self.preview.shutdown()
        self.pool.waitForDone(3000)
        super().closeEvent(event)

def main():
    app = QApplication([APP_NAME, *sys.argv[1:]])
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setOrganizationDomain(ORG_DOMAIN)
    app.setStyle("Fusion")

    settings = QSettings()
    theme.apply(app, settings.value("theme", theme.DEFAULT_MODE))

    window = MainWindow()
    try:
        app.styleHints().colorSchemeChanged.connect(
            lambda _scheme: window._repaint_theme())
    except AttributeError:
        pass
    window.show()
    sys.exit(app.exec())
