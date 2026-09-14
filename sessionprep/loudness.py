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
from scipy import signal
from scipy.signal import lfilter

BLOCK_SECONDS = 0.4
OVERLAP = 0.75
ABSOLUTE_GATE = -70.0
RELATIVE_GATE = -10.0
OFFSET = -0.691

SHELF_G = 3.999843853973347
SHELF_Q = 0.7071752369554196
SHELF_F0 = 1681.974450955533
HPF_Q = 0.5003270373238773
HPF_F0 = 38.13547087602444

def k_weighting(samplerate: float):
    K = math.tan(math.pi * SHELF_F0 / samplerate)
    Vh = 10.0 ** (SHELF_G / 20.0)
    Vb = Vh ** 0.4996667741545416
    denom = 1.0 + K / SHELF_Q + K * K
    b1 = np.array([
        (Vh + Vb * K / SHELF_Q + K * K) / denom,
        2.0 * (K * K - Vh) / denom,
        (Vh - Vb * K / SHELF_Q + K * K) / denom,
    ])
    a1 = np.array([1.0, 2.0 * (K * K - 1.0) / denom,
                   (1.0 - K / SHELF_Q + K * K) / denom])

    K = math.tan(math.pi * HPF_F0 / samplerate)
    denom = 1.0 + K / HPF_Q + K * K
    a2 = np.array([1.0, 2.0 * (K * K - 1.0) / denom,
                   (1.0 - K / HPF_Q + K * K) / denom])
    b2 = np.array([1.0, -2.0, 1.0])
    return (b1, a1), (b2, a2)

def channel_weights(channels: int) -> np.ndarray:
    if channels == 1:
        return np.array([2.0])
    return np.ones(channels)

class LoudnessMeter:
    def __init__(self, samplerate: int, channels: int):
        self.samplerate = samplerate
        self.channels = channels
        (self.b1, self.a1), (self.b2, self.a2) = k_weighting(samplerate)
        self.zi1 = np.zeros((2, channels))
        self.zi2 = np.zeros((2, channels))
        self.weights = channel_weights(channels)

        self.step = max(1, int(round(samplerate * BLOCK_SECONDS * (1 - OVERLAP))))
        self.per_block = int(round(1 / (1 - OVERLAP)))
        self._segments: list[np.ndarray] = []
        self._acc = np.zeros(channels)
        self._acc_n = 0
        self._total = 0

    def process(self, data: np.ndarray) -> None:
        if data.ndim == 1:
            data = data[:, None]
        if not data.size:
            return
        self._total += data.shape[0]
        y, self.zi1 = lfilter(self.b1, self.a1, data, axis=0, zi=self.zi1)
        y, self.zi2 = lfilter(self.b2, self.a2, y, axis=0, zi=self.zi2)
        squared = np.square(y, out=y)

        if self._acc_n:
            need = self.step - self._acc_n
            head = squared[:need]
            self._acc += head.sum(axis=0)
            self._acc_n += head.shape[0]
            if self._acc_n == self.step:
                self._segments.append(self._acc)
                self._acc = np.zeros(self.channels)
                self._acc_n = 0
            squared = squared[head.shape[0]:]

        whole = (squared.shape[0] // self.step) * self.step
        if whole:
            sums = squared[:whole].reshape(-1, self.step, self.channels).sum(axis=1)
            self._segments.extend(sums)

        remainder = squared[whole:]
        if remainder.shape[0]:
            self._acc += remainder.sum(axis=0)
            self._acc_n += remainder.shape[0]

    def _block_energies(self) -> np.ndarray:
        count = len(self._segments) - self.per_block + 1
        if count <= 0:
            return np.empty(0)
        segs = np.asarray(self._segments)
        cumulative = np.concatenate(([np.zeros(self.channels)], np.cumsum(segs, axis=0)))
        window = cumulative[self.per_block:] - cumulative[:-self.per_block]
        mean_square = window / (self.step * self.per_block)
        return mean_square @ self.weights

    def integrated(self) -> tuple[float, bool]:
        energies = self._block_energies()
        if energies.size == 0:
            return self._ungated(), False

        loudness = OFFSET + 10.0 * np.log10(np.maximum(energies, 1e-30))
        keep = energies[loudness > ABSOLUTE_GATE]
        if keep.size == 0:
            return -math.inf, True

        threshold = OFFSET + 10.0 * math.log10(keep.mean()) + RELATIVE_GATE
        final = keep[OFFSET + 10.0 * np.log10(np.maximum(keep, 1e-30)) > threshold]
        if final.size == 0:
            return -math.inf, True
        return OFFSET + 10.0 * math.log10(final.mean()), True

    def _ungated(self) -> float:
        total = self._acc.copy()
        for seg in self._segments:
            total = total + seg
        frames = self._total
        if frames == 0:
            return -math.inf
        energy = float((total / frames) @ self.weights)
        if energy <= 0:
            return -math.inf
        return OFFSET + 10.0 * math.log10(energy)

def measure(data: np.ndarray, samplerate: int) -> tuple[float, bool]:
    if data.ndim == 1:
        data = data[:, None]
    meter = LoudnessMeter(samplerate, data.shape[1])
    meter.process(data)
    return meter.integrated()

VU_RISE_S = 0.300
VU_RISE_FRACTION = 0.99
VU_HOP_S = 0.005
VU_REFERENCE_DBFS = -18.0

_VU_TAU = -VU_RISE_S / math.log(1.0 - VU_RISE_FRACTION ** 2)

class VUMeter:
    def __init__(self, samplerate: int, channels: int):
        self.samplerate = samplerate
        self.channels = channels
        self.hop = max(1, int(round(VU_HOP_S * samplerate)))
        alpha = 1.0 - math.exp(-(self.hop / samplerate) / _VU_TAU)
        self._b = np.array([alpha])
        self._a = np.array([1.0, -(1.0 - alpha)])
        self._zi = np.zeros((1, channels))
        self._max = 0.0
        self._carry = np.zeros((0, channels))

    def process(self, data: np.ndarray) -> None:
        if data.ndim == 1:
            data = data[:, None]
        if not data.size:
            return
        if self._carry.size:
            data = np.concatenate((self._carry, data))
        usable = (data.shape[0] // self.hop) * self.hop
        self._carry = np.array(data[usable:], dtype=np.float64)
        if not usable:
            return

        squared = data[:usable] ** 2
        per_hop = squared.reshape(-1, self.hop, data.shape[1]).mean(axis=1)
        smoothed, self._zi = signal.lfilter(self._b, self._a, per_hop,
                                            axis=0, zi=self._zi)
        if smoothed.size:
            loudest = float(np.max(smoothed))
            if loudest > self._max:
                self._max = loudest

    @property
    def dbfs(self) -> float:
        return 10.0 * math.log10(self._max) if self._max > 0 else -math.inf

    @property
    def vu(self) -> float:
        return self.dbfs - VU_REFERENCE_DBFS

def measure_vu(data: np.ndarray, samplerate: int) -> float:
    if data.ndim == 1:
        data = data[:, None]
    meter = VUMeter(samplerate, data.shape[1])
    meter.process(np.asarray(data, dtype=np.float64))
    return meter.vu

PPM_INTEGRATION_S = 0.005
PPM_BURST_DEFICIT_DB = 2.0
PPM_FALLBACK_DB = 20.0
PPM_FALLBACK_S = 1.7
PPM_HOP_S = 0.0005
NORDIC_ZERO_DBFS = -18.0

_PPM_TAU = PPM_INTEGRATION_S / -math.log(1.0 - 10 ** (-PPM_BURST_DEFICIT_DB / 20.0))

class PPMMeter:
    def __init__(self, samplerate: int):
        self.samplerate = samplerate
        self.hop = max(1, int(round(PPM_HOP_S * samplerate)))
        hop_s = self.hop / samplerate
        self._attack = 1.0 - math.exp(-hop_s / _PPM_TAU)
        self._release = 10 ** (-(PPM_FALLBACK_DB * hop_s / PPM_FALLBACK_S) / 20.0)
        self._env = 0.0
        self._max = 0.0
        self._carry = np.zeros(0)

    def process(self, data: np.ndarray) -> None:
        if data.ndim == 1:
            data = data[:, None]
        if not data.size:
            return
        rectified = np.abs(data).max(axis=1)
        if self._carry.size:
            rectified = np.concatenate((self._carry, rectified))
        usable = (rectified.size // self.hop) * self.hop
        self._carry = rectified[usable:].copy()
        if not usable:
            return

        peaks = rectified[:usable].reshape(-1, self.hop).max(axis=1)
        env, top = self._env, self._max
        attack, release = self._attack, self._release
        for value in peaks:
            if value > env:
                env += attack * (value - env)
            else:
                env *= release
            if env > top:
                top = env
        self._env, self._max = env, top

    @property
    def dbfs(self) -> float:
        return 20.0 * math.log10(self._max) if self._max > 0 else -math.inf

    @property
    def ppm(self) -> float:
        return self.dbfs - NORDIC_ZERO_DBFS

def measure_ppm(data: np.ndarray, samplerate: int) -> float:
    meter = PPMMeter(samplerate)
    meter.process(data)
    return meter.dbfs
