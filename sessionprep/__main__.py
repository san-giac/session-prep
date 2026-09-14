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

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CONFIG = HERE / "config.yaml"


def _applescript(text: str) -> str:
    for raw, escaped in (("\\", r"\\"), ('"', r'\"'),
                         ("\r\n", r"\n"), ("\n", r"\n"), ("\r", r"\n")):
        text = text.replace(raw, escaped)
    return f'"{text}"'


def _report(body: str) -> None:
    print(body, file=sys.stderr)
    if sys.platform != "darwin" or sys.stderr.isatty():
        return
    try:
        subprocess.run(
            ["osascript", "-e",
             'display dialog {} with title {} buttons {{"OK"}} '
             'default button 1 with icon stop'.format(
                 _applescript(body), _applescript("Session Prep"))],
            capture_output=True, timeout=120)
    except Exception:
        pass


try:
    from .classify_rules import RulesError
except Exception:
    class RulesError(Exception):
        pass


try:
    from .app import main
except RulesError as exc:
    _report("\n".join([
        "The classification rules could not be read, so the app can't start.",
        "",
        str(exc),
        "",
        f"Edit: {CONFIG}",
        "",
        "If the message names a line and column, that is where the parser",
        "gave up — usually an indentation slip or an unclosed quote.",
    ]))
    raise SystemExit(1)
except ModuleNotFoundError as exc:
    missing = exc.name or ""
    if missing.split(".")[0] == "sessionprep":
        _report("\n".join([
            f"The sessionprep package is incomplete — {missing} is missing.",
            "",
            f"Looked in: {HERE}",
        ]))
    else:
        _report("\n".join([
            f"Missing dependency: {missing}",
            "",
            "Install the requirements first:",
            f"    cd {PROJECT}",
            "    python3 -m venv .venv && source .venv/bin/activate",
            "    pip install -r requirements.txt",
        ]))
    raise SystemExit(1)

if __name__ == "__main__":
    main()
