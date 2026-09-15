# Session Prep

<img width="3840" height="2097" alt="sess_prep_02" src="https://github.com/user-attachments/assets/7a3de6f8-8c7b-4036-bbb0-5af94aa5ffb5" />


A tool for analysing and preparing audio files for a range of end uses.

This was mainly made for my own use when preparing audio files for mixing (hence a few functions 
might not be particularly useful to other people, particularly in the re-naming section).

Main features:
- sample rate and bit rate conversion
- loudness normalisation
- mono, clipping, phase checks
- renaming tools

macOS, Python 3.10+.

## Setup

**1. Install the dependencies**

```bash
cd /path/to/session_prep
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Build the app**

```bash
python3 make_app.py
```

Writes `Session Prep.app` next to the script. From here on, launching is a
double-click and Terminal isn't needed again.

## Running it

Double-click `Session Prep.app`, or drag it to the Dock.

Keep the bundle in this folder. It's a launcher, so it needs all files in the folder.
Rerun `python3 make_app.py` after moving the project or switching Python; the
launcher has the paths baked in.

From a terminal, without the bundle:

```bash
source .venv/bin/activate
python3 -m sessionprep
```

Run that from the project folder — `-m` finds the package relative to the
working directory.

### Why you build the bundle rather than download one

Anything unpacked from a downloaded zip carries macOS's quarantine flag and gets
stopped by Gatekeeper. A bundle created on your own machine is not flagged, so
shipping the build script sidesteps the problem entirely.


## Usage

You can find some detailed information here: https://sandrogiacometti.com/things/2026/9/14/session-prep-tool

## Licence
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>. The full text is in
[LICENSE](LICENSE).

### Third-party

| Package | Licence |
| --- | --- |
| PySide6-Essentials (Qt) | LGPL-3.0 |
| python-soxr (libsoxr) | LGPL-2.1-or-later |
| soundfile (libsndfile) | BSD-3-Clause wrapper, LGPL-2.1 library |
| sounddevice (PortAudio) | MIT |
| NumPy, SciPy, numba | BSD |
| PyYAML | MIT |
