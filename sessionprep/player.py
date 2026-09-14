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

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
import soxr

if TYPE_CHECKING:
    from sounddevice import OutputStream

try:
    import sounddevice as sd
    AUDIO_ERROR = ""
except Exception as exc:
    sd = None
    AUDIO_ERROR = str(exc)

OUTPUT_CHANNELS = 2
BLOCKSIZE = 2048
PREFETCH_SECONDS = 4.0
READ_FRAMES = 8192

_RATE_CACHE: dict[int, int] = {}


def playable_samplerate(rate: int) -> int:
    if sd is None or rate <= 0:
        return rate
    if rate in _RATE_CACHE:
        return _RATE_CACHE[rate]
    answer = rate
    try:
        sd.check_output_settings(samplerate=rate, channels=OUTPUT_CHANNELS,
                                 dtype="float32")
    except Exception:
        try:
            default = sd.query_devices(kind="output")["default_samplerate"]
            answer = int(round(default)) or rate
        except Exception:
            answer = rate
    _RATE_CACHE[rate] = answer
    return answer


def _to_stereo(block: np.ndarray) -> np.ndarray:
    if block.ndim == 1:
        block = block[:, None]
    if block.shape[1] == 1:
        return np.repeat(block, OUTPUT_CHANNELS, axis=1)
    if block.shape[1] > OUTPUT_CHANNELS:
        return block[:, :OUTPUT_CHANNELS]
    return block


class Player:
    def __init__(self) -> None:
        self._stream: OutputStream | None = None
        self._stream_rate: int | None = None

        self._path: Path | None = None
        self._src_rate = 0
        self._samplerate = 44100
        self._frames = 0
        self._loop = False

        self._pos = 0
        self._underruns = 0
        self._callbacks = 0

        capacity = max(BLOCKSIZE * 4, int(44100 * PREFETCH_SECONDS))
        self._ring = np.zeros((capacity, OUTPUT_CHANNELS), dtype=np.float32)
        self._capacity = capacity
        self._write = 0
        self._read = 0
        self._eof = False

        self._seek_lock = threading.Lock()
        self._wake = threading.Event()
        self._reader: threading.Thread | None = None
        self._stop_reader = threading.Event()
        self._seek_to: int | None = None

    @property
    def available(self) -> bool:
        return sd is not None

    @property
    def unavailable_reason(self) -> str:
        return AUDIO_ERROR

    @property
    def playing(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def samplerate(self) -> int:
        return self._samplerate

    @property
    def position(self) -> int:
        return self._pos

    @property
    def underruns(self) -> int:
        return self._underruns

    def set_loop(self, loop: bool) -> None:
        self._loop = loop

    def load(self, path: Path | str, play_rate: int | None = None) -> str:
        self.stop()
        self._stop_reading()
        self._path = None
        self._frames = 0
        self._pos = 0

        if path is None:
            return ""
        path = Path(path)
        try:
            info = sf.info(str(path))
        except Exception as exc:
            return f"Could not read {path.name}: {exc}"

        self._path = path
        self._src_rate = info.samplerate
        self._samplerate = int(play_rate or info.samplerate)
        ratio = self._samplerate / self._src_rate
        self._frames = int(round(info.frames * ratio))
        self._reset_ring()
        return ""

    def seek(self, frame: int) -> None:
        target = int(np.clip(frame, 0, max(0, self._frames)))
        self._pos = target
        with self._seek_lock:
            self._seek_to = target
            self._reset_ring()
        self._wake.set()

    def play(self, start: int | None = None) -> str:
        if sd is None:
            return AUDIO_ERROR or "No audio output available"
        if self._path is None:
            return "Nothing loaded"

        if start is not None:
            self.seek(start)
        elif self._pos >= self._frames:
            self.seek(0)

        self._start_reading()
        error = self._ensure_stream()
        if error:
            return error
        try:
            if self._stream.active:
                return ""
            self._stream.stop()
            self._stream.start()
        except Exception as exc:
            return f"Could not start playback: {exc}"
        return ""

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                self._close_stream()

    def toggle(self, start: int | None = None) -> str:
        if self.playing:
            self.stop()
            return ""
        return self.play(start)

    def shutdown(self) -> None:
        self.stop()
        self._stop_reading()
        self._close_stream()

    def _reset_ring(self) -> None:
        self._read = self._write = 0
        self._eof = False

    def _start_reading(self) -> None:
        if self._reader is not None and self._reader.is_alive():
            self._wake.set()
            return
        self._stop_reader.clear()
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name="preview-reader")
        self._reader.start()

    def _stop_reading(self) -> None:
        self._stop_reader.set()
        self._wake.set()
        reader, self._reader = self._reader, None
        if reader is not None and reader.is_alive():
            reader.join(timeout=2.0)

    def _read_loop(self) -> None:
        handle = None
        resampler = None
        try:
            while not self._stop_reader.is_set():
                pending = self._seek_to
                if pending is not None or handle is None:
                    with self._seek_lock:
                        pending, self._seek_to = self._seek_to, None
                    if handle is None:
                        handle = sf.SoundFile(str(self._path))
                    resampler = self._new_resampler()
                    source_frame = int(round((pending or 0)
                                             * self._src_rate / self._samplerate))
                    handle.seek(min(source_frame, len(handle)))
                    self._eof = False

                space = self._capacity - (self._write - self._read) - 1
                if space < READ_FRAMES:
                    self._wake.wait(0.01)
                    self._wake.clear()
                    continue

                block = handle.read(READ_FRAMES, dtype="float32", always_2d=True)
                last = block.shape[0] < READ_FRAMES
                if block.shape[0]:
                    out = _to_stereo(block)
                    if resampler is not None:
                        out = resampler.resample_chunk(
                            np.ascontiguousarray(out), last=last)
                    if out.size:
                        self._push(_to_stereo(np.asarray(out, dtype=np.float32)))

                if last:
                    if self._loop:
                        handle.seek(0)
                        resampler = self._new_resampler()
                    else:
                        self._eof = True
                        self._wake.wait(0.05)
                        self._wake.clear()
        except Exception:
            self._eof = True
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

    def _new_resampler(self):
        if self._samplerate == self._src_rate:
            return None
        return soxr.ResampleStream(self._src_rate, self._samplerate,
                                   OUTPUT_CHANNELS, dtype="float32",
                                   quality="HQ")

    def _push(self, block: np.ndarray) -> None:
        n = block.shape[0]
        start = self._write % self._capacity
        first = min(n, self._capacity - start)
        self._ring[start:start + first] = block[:first]
        if first < n:
            self._ring[:n - first] = block[first:]
        self._write += n

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        self._stream_rate = None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def _ensure_stream(self) -> str:
        if self._stream is not None and self._stream_rate == self._samplerate:
            return ""
        self._close_stream()
        try:
            self._stream = sd.OutputStream(
                samplerate=self._samplerate,
                channels=OUTPUT_CHANNELS,
                dtype="float32",
                blocksize=BLOCKSIZE,
                latency="high",
                callback=self._callback,
            )
            self._stream_rate = self._samplerate
            self._underruns = self._callbacks = 0
        except Exception as exc:
            self._stream = None
            self._stream_rate = None
            return f"Could not open output at {self._samplerate} Hz: {exc}"
        return ""

    def _callback(self, outdata, frames, time_info, status):
        self._callbacks += 1
        if status and status.output_underflow:
            self._underruns += 1

        if not self._seek_lock.acquire(blocking=False):
            outdata.fill(0)
            return
        try:
            available = self._write - self._read
            take = min(frames, available)
            if take > 0:
                start = self._read % self._capacity
                first = min(take, self._capacity - start)
                outdata[:first] = self._ring[start:start + first]
                if first < take:
                    outdata[first:take] = self._ring[:take - first]
                self._read += take
                self._pos += take
            if take < frames:
                outdata[take:] = 0.0
                if self._eof and self._write == self._read:
                    raise sd.CallbackStop
        finally:
            self._seek_lock.release()
