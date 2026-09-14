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

import sys
import threading
import traceback

from PySide6.QtCore import QObject, Signal, qInstallMessageHandler


class ErrorBus(QObject):

    raised = Signal(str, str)         

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def report(self, exc_type, exc, tb, where: str = "") -> None:
        detail = "".join(traceback.format_exception(exc_type, exc, tb))
        frame = _last_frame(tb)
        key = f"{exc_type.__name__}@{frame}"
        with self._lock:
            first = key not in self._seen
            self._seen.add(key)

        if first:
            sys.stderr.write(detail)
            sys.stderr.flush()

        if first:
            summary = f"{exc_type.__name__}: {exc}"
            if frame:
                summary += f"  ({frame})"
            if where:
                summary = f"{where} — {summary}"
            self.raised.emit(summary, detail)


def _last_frame(tb) -> str:
    try:
        frames = traceback.extract_tb(tb)
        if not frames:
            return ""
        last = frames[-1]
        return f"{last.filename.rsplit('/', 1)[-1]}:{last.lineno}"
    except Exception:
        return ""


def install(bus: ErrorBus) -> None:
    previous_hook = sys.excepthook

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            previous_hook(exc_type, exc, tb)
            return
        try:
            bus.report(exc_type, exc, tb)
        except Exception:
            previous_hook(exc_type, exc, tb)

    sys.excepthook = hook

    previous_thread_hook = threading.excepthook

    def thread_hook(args):
        if issubclass(args.exc_type, SystemExit):
            return
        try:
            bus.report(args.exc_type, args.exc_value, args.exc_traceback,
                       where=getattr(args.thread, "name", "") or "worker")
        except Exception:
            previous_thread_hook(args)

    threading.excepthook = thread_hook

    def message_handler(mode, context, message):
        try:
            name = mode.name if hasattr(mode, "name") else str(mode)
        except Exception:
            name = str(mode)
        if "Critical" in name or "Fatal" in name:
            bus.raised.emit(f"Qt: {message}", message)
        else:
            sys.stderr.write(f"{message}\n")

    qInstallMessageHandler(message_handler)
