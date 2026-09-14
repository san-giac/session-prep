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

from enum import Enum

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap if not args else args[0]

class DitherMode(Enum):
    NONE = "none"
    TPDF = "tpdf"
    SHAPED = "shaped"

E_WEIGHTED_9 = np.array(
    [2.412, -3.370, 3.937, -4.174, 3.353, -2.205, 1.281, -0.569, 0.0847],
    dtype=np.float64,
)

SHAPING_MAX_RATE = 50000

@njit(cache=True, nogil=True)
def _shape_kernel(x, coeffs, scale, lo, hi, noise, hist):
    n = x.shape[0]
    order = coeffs.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        fb = 0.0
        for k in range(order):
            fb += coeffs[k] * hist[k]
        v = x[i] * scale - fb
        q = np.rint(v + noise[i])
        err = q - v
        if err > 2.0:
            err = 2.0
        elif err < -2.0:
            err = -2.0
        for k in range(order - 1, 0, -1):
            hist[k] = hist[k - 1]
        hist[0] = err
        if q > hi:
            q = hi
        elif q < lo:
            q = lo
        out[i] = q
    return out


def _tpdf(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.random(n) - rng.random(n)

_INT_DTYPE = {8: np.int16, 16: np.int16, 24: np.int32, 32: np.int32}
_INT_SHIFT = {8: 8, 16: 0, 24: 8, 32: 0}


class Quantiser:

    def __init__(self, bits: int, mode: DitherMode = DitherMode.SHAPED,
                 samplerate: int = 44100, channels: int = 2,
                 seed: int | None = None) -> None:
        if bits not in _INT_DTYPE:
            raise ValueError(f"unsupported bit depth: {bits}")
        self.bits = bits
        self.mode = (DitherMode.TPDF
                     if mode is DitherMode.SHAPED and samplerate > SHAPING_MAX_RATE
                     else mode)
        self.scale = float(1 << (bits - 1))
        self.lo, self.hi = -self.scale, self.scale - 1.0
        self._rng = np.random.default_rng(seed)
        self._hist = [np.zeros(E_WEIGHTED_9.shape[0], dtype=np.float64)
                      for _ in range(max(1, channels))]

    def process(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data[:, None]
        frames, channels = data.shape
        out = np.empty((frames, channels), dtype=np.float64)

        for ch in range(channels):
            col = np.ascontiguousarray(data[:, ch], dtype=np.float64)
            if self.mode is DitherMode.NONE:
                out[:, ch] = np.clip(np.rint(col * self.scale), self.lo, self.hi)
            elif self.mode is DitherMode.TPDF:
                out[:, ch] = np.clip(
                    np.rint(col * self.scale + _tpdf(self._rng, frames)),
                    self.lo, self.hi)
            else:
                noise = _tpdf(self._rng, frames)
                out[:, ch] = _shape_kernel(col, E_WEIGHTED_9, self.scale,
                                           self.lo, self.hi, noise,
                                           self._hist[ch])

        ints = out.astype(_INT_DTYPE[self.bits])
        shift = _INT_SHIFT[self.bits]
        return ints << shift if shift else ints


def quantise(
    data: np.ndarray,
    bits: int,
    mode: DitherMode = DitherMode.SHAPED,
    samplerate: int = 44100,
    seed: int | None = None,
) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    return Quantiser(bits, mode, samplerate, data.shape[1], seed).process(data)
