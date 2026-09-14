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
from enum import Enum
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .loudness import LoudnessMeter, PPMMeter, VUMeter

BLOCK = 1 << 18

NORMALISE_REF_DBFS = -6.0
PHASE_CANCEL_DBFS = -40.0
SILENCE_DBFS = -90.0
CLIP_MIN_RUN = 3

START_WINDOW_S = 0.05
START_MIN_RUN = 4
START_FLOOR_PCT = 20
START_MARGIN_DB = 12.0
START_FLOOR_DBFS = -75.0

TRUE_PEAK_OVERSAMPLE = 4
TRUE_PEAK_SCAN_ABOVE_DBFS = -6.0


class StereoKind(Enum):
    MONO = "mono"
    IDENTICAL = "identical"
    NEAR_IDENTICAL = "near"
    INVERTED = "inverted"
    SCALED = "scaled"
    STEREO = "stereo"
    SILENT = "silent"
    MULTICHANNEL = "multi"

    @property
    def is_fake_stereo(self) -> bool:
        return self in (StereoKind.IDENTICAL, StereoKind.NEAR_IDENTICAL, StereoKind.SCALED)


class SilenceFlag(Enum):
    OK = "ok"
    SILENT = "silent"
    CHANNEL_SILENT = "channel_silent"

    @property
    def is_issue(self) -> bool:
        return self is not SilenceFlag.OK


class ClipFlag(Enum):
    OK = "ok"
    INTERSAMPLE = "intersample"
    CLIPPED = "clipped"

    @property
    def is_issue(self) -> bool:
        return self is not ClipFlag.OK


class PolarityFlag(Enum):
    OK = "ok"
    INVERTED = "inverted"
    CANCELS = "cancels"

    @property
    def is_issue(self) -> bool:
        return self is not PolarityFlag.OK


@dataclass
class FileInfo:
    path: Path
    samplerate: int
    channels: int
    frames: int
    subtype: str
    format: str

    @property
    def duration(self) -> float:
        return self.frames / self.samplerate if self.samplerate else 0.0

    @property
    def bit_depth(self) -> int | None:
        return SUBTYPE_BITS.get(self.subtype)


SUBTYPE_BITS = {
    "PCM_S8": 8, "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
}


@dataclass(kw_only=True)
class StereoAnalysis:
    kind: StereoKind
    side_peak_db: float
    side_rms_db: float
    correlation: float
    gain_ratio_db: float
    peak_dbfs: float
    polarity: PolarityFlag = None
    mono_loss_db: float = 0.0
    mono_sum_db: float = -math.inf
    loudness_lufs: float = -math.inf
    loudness_gated: bool = True
    vu_dbfs: float = -math.inf
    ppm_dbfs: float = -math.inf

    silence: SilenceFlag = None
    clipping: ClipFlag = None
    clipped_samples: int = 0
    clip_runs: int = 0
    clip_longest_run: int = 0
    true_peak_dbfs: float = -math.inf
    start_seconds: float = math.inf
    rms_dbfs: float = -math.inf
    channel_peaks_db: tuple = ()

    def __post_init__(self):
        if self.polarity is None:
            self.polarity = PolarityFlag.OK
        if self.silence is None:
            self.silence = SilenceFlag.OK
        if self.clipping is None:
            self.clipping = ClipFlag.OK

    @property
    def is_fake_stereo(self) -> bool:
        return self.kind.is_fake_stereo

    @property
    def has_polarity_issue(self) -> bool:
        return self.polarity.is_issue

    @property
    def has_issue(self) -> bool:
        return (self.polarity.is_issue or self.silence.is_issue or self.clipping.is_issue)

    def redecide(self, threshold_db: float, cancel_db: float) -> None:
        if self.kind in (StereoKind.NEAR_IDENTICAL, StereoKind.STEREO):
            self.kind = (StereoKind.NEAR_IDENTICAL
                         if self.side_rms_db <= threshold_db
                         and self.side_peak_db <= threshold_db + 20.0
                         else StereoKind.STEREO)
        if self.kind in (StereoKind.MONO, StereoKind.SILENT, StereoKind.MULTICHANNEL):
            self.polarity = PolarityFlag.OK
        elif self.correlation <= -0.999:
            self.polarity = PolarityFlag.INVERTED
        elif self.mono_sum_db < cancel_db:
            self.polarity = PolarityFlag.CANCELS
        else:
            self.polarity = PolarityFlag.OK

    @property
    def needs_single_channel_fold(self) -> bool:
        return (self.kind in (StereoKind.SCALED, StereoKind.INVERTED)
                or self.silence is SilenceFlag.CHANNEL_SILENT)


def probe(path: Path | str) -> FileInfo:
    path = Path(path)
    info = sf.info(str(path))
    return FileInfo(
        path=path,
        samplerate=info.samplerate,
        channels=info.channels,
        frames=info.frames,
        subtype=info.subtype,
        format=info.format,
    )


def _db(x: float, floor: float = -200.0) -> float:
    return 20.0 * math.log10(x) if x > 0 else floor


def _full_scale(subtype: str) -> float:
    bits = SUBTYPE_BITS.get(subtype)
    if not bits:
        return 1.0
    return (2 ** (bits - 1) - 1) / 2 ** (bits - 1)


def _scan_runs(mask: np.ndarray, carry: int, min_run: int) -> tuple[int, int, int]:
    n = mask.size
    if n == 0:
        return 0, 0, carry

    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    if starts.size == 0:
        return (1 if carry >= min_run else 0), carry, 0

    lengths = (ends - starts).astype(np.int64)
    runs = longest = 0

    if carry:
        if starts[0] == 0:
            lengths[0] += carry
        elif carry >= min_run:
            runs += 1
            longest = carry

    new_carry = 0
    if ends[-1] == n:
        new_carry = int(lengths[-1])
        lengths = lengths[:-1]

    if lengths.size:
        runs += int(np.count_nonzero(lengths >= min_run))
        longest = max(longest, int(lengths.max()))
    return runs, longest, new_carry


def _true_peak_of(path: Path, rate: int, channels: int) -> float:
    stream = soxr.ResampleStream(rate, rate * TRUE_PEAK_OVERSAMPLE, channels,
                                 dtype="float32", quality="HQ")
    highest = 0.0
    with sf.SoundFile(str(path)) as handle:
        for block in handle.blocks(blocksize=BLOCK, dtype="float32", always_2d=True):
            if not block.size:
                continue
            upsampled = stream.resample_chunk(np.ascontiguousarray(block))
            if upsampled.size:
                highest = max(highest, float(np.abs(upsampled).max()))

    tail = stream.resample_chunk(np.zeros((0, channels), dtype=np.float32), last=True)
    if tail.size:
        highest = max(highest, float(np.abs(tail).max()))
    return highest


def _first_sound(levels: np.ndarray, window_s: float) -> float:
    if levels.size == 0:
        return math.inf
    loudest = float(levels.max())
    if loudest <= 0:
        return math.inf

    quiet = levels[levels > 0]
    floor = float(np.percentile(quiet, START_FLOOR_PCT)) if quiet.size else 0.0

    threshold = max(floor * 10 ** (START_MARGIN_DB / 20.0), 10 ** (START_FLOOR_DBFS / 20.0))
    if threshold >= loudest:
        threshold = loudest * 0.5

    above = levels >= threshold
    run = 0
    for index, value in enumerate(above):
        run = run + 1 if value else 0
        if run >= START_MIN_RUN:
            return (index - run + 1) * window_s
    return math.inf


def analyse(path: Path | str, threshold_db: float = -60.0, cancel_db: float = PHASE_CANCEL_DBFS) -> StereoAnalysis:
    path = Path(path)
    info = sf.info(str(path))
    channels = info.channels
    rate = info.samplerate

    ch_peak = np.zeros(channels)
    fs = _full_scale(info.subtype)
    clip_count = np.zeros(channels, dtype=np.int64)
    clip_runs = np.zeros(channels, dtype=np.int64)
    clip_longest = np.zeros(channels, dtype=np.int64)
    clip_carry = np.zeros(channels, dtype=np.int64)

    channel_energy = np.zeros(channels)
    start_hop = max(1, int(round(START_WINDOW_S * rate)))
    start_levels: list = []
    start_carry = np.zeros(0)
    meter = LoudnessMeter(rate, channels)
    vu_meter = VUMeter(rate, channels)
    ppm_meter = PPMMeter(rate)

    exact = True
    mid_peak = side_peak = 0.0
    lr = 0.0
    n = 0

    with sf.SoundFile(str(path)) as f:
        for block in f.blocks(blocksize=BLOCK, dtype="float64", always_2d=True):
            if not block.size:
                continue
            n += block.shape[0]

            rows = np.ascontiguousarray(block.T)
            magnitude = np.abs(rows)
            peaks = magnitude.max(axis=1)
            ch_peak = np.maximum(ch_peak, peaks)
            channel_energy += np.einsum("ij,ij->i", rows, rows)

            if float(peaks.max()) >= fs or clip_carry.any():
                for ch in range(channels):
                    pinned = magnitude[ch] >= fs
                    hits = int(np.count_nonzero(pinned))
                    if hits or clip_carry[ch]:
                        clip_count[ch] += hits
                        runs, longest, clip_carry[ch] = _scan_runs(pinned, int(clip_carry[ch]), CLIP_MIN_RUN)
                        clip_runs[ch] += runs
                        clip_longest[ch] = max(clip_longest[ch], longest)

            meter.process(block)
            vu_meter.process(block)
            ppm_meter.process(block)

            if channels == 2:
                left, right = rows[0], rows[1]
                if exact and not np.array_equal(left, right):
                    exact = False
                mid = (left + right) * 0.5
                side = (left - right) * 0.5
                mid_peak = max(mid_peak, float(np.abs(mid).max()))
                side_peak = max(side_peak, float(np.abs(side).max()))
                lr += float(np.dot(left, right))
                mono = mid
            elif channels == 1:
                mono = rows[0]
            else:
                mono = rows.mean(axis=0)

            if start_carry.size:
                mono = np.concatenate((start_carry, mono))
            usable = (mono.size // start_hop) * start_hop
            if usable:
                windows = mono[:usable].reshape(-1, start_hop)
                start_levels.append(np.sqrt(np.einsum("ij,ij->i", windows, windows) / start_hop))
            start_carry = np.array(mono[usable:], dtype=np.float64)

            if channels != 2:
                continue

    for ch in range(channels):
        if clip_carry[ch]:
            if clip_carry[ch] >= CLIP_MIN_RUN:
                clip_runs[ch] += 1
            clip_longest[ch] = max(clip_longest[ch], clip_carry[ch])

    loudness_lufs, loudness_gated = meter.integrated()

    rms_value = -math.inf
    if n:
        loudest_channel = float(np.max(channel_energy)) / n
        rms_value = 10.0 * math.log10(loudest_channel) if loudest_channel > 0 else -math.inf

    start_seconds = _first_sound(np.concatenate(start_levels) if start_levels else np.zeros(0), START_WINDOW_S)
    overall_peak = float(ch_peak.max()) if channels else 0.0
    peak_dbfs = _db(overall_peak)
    channel_peaks_db = tuple(_db(float(p)) for p in ch_peak)

    if peak_dbfs >= TRUE_PEAK_SCAN_ABOVE_DBFS:
        true_peak_dbfs = _db(max(_true_peak_of(path, rate, channels), overall_peak))
    else:
        true_peak_dbfs = peak_dbfs

    if n == 0 or peak_dbfs < SILENCE_DBFS:
        silence = SilenceFlag.SILENT
    elif channels >= 2 and any(p < SILENCE_DBFS for p in channel_peaks_db):
        silence = SilenceFlag.CHANNEL_SILENT
    else:
        silence = SilenceFlag.OK

    total_runs = int(clip_runs.sum())
    if total_runs:
        clipping = ClipFlag.CLIPPED
    elif true_peak_dbfs > 0.0:
        clipping = ClipFlag.INTERSAMPLE
    else:
        clipping = ClipFlag.OK

    meter = dict(
        silence=silence, clipping=clipping,
        clipped_samples=int(clip_count.sum()), clip_runs=total_runs,
        clip_longest_run=int(clip_longest.max()) if channels else 0,
        true_peak_dbfs=true_peak_dbfs, channel_peaks_db=channel_peaks_db,
        start_seconds=start_seconds, rms_dbfs=rms_value,
        loudness_lufs=loudness_lufs, loudness_gated=loudness_gated,
        vu_dbfs=vu_meter.dbfs, ppm_dbfs=ppm_meter.dbfs,
    )

    if silence is SilenceFlag.SILENT:
        return StereoAnalysis(
            kind=StereoKind.SILENT, side_peak_db=-math.inf, side_rms_db=-math.inf,
            correlation=1.0, gain_ratio_db=0.0, peak_dbfs=peak_dbfs,
            polarity=PolarityFlag.OK, mono_loss_db=0.0, **meter)
    if channels == 1:
        return StereoAnalysis(
            kind=StereoKind.MONO, side_peak_db=-math.inf, side_rms_db=-math.inf,
            correlation=1.0, gain_ratio_db=0.0, peak_dbfs=peak_dbfs,
            polarity=PolarityFlag.OK, mono_loss_db=0.0, **meter)
    if channels > 2:
        return StereoAnalysis(
            kind=StereoKind.MULTICHANNEL, side_peak_db=math.nan, side_rms_db=math.nan,
            correlation=math.nan, gain_ratio_db=math.nan, peak_dbfs=peak_dbfs,
            polarity=PolarityFlag.OK, mono_loss_db=math.nan, **meter)

    ll, rr = float(channel_energy[0]), float(channel_energy[1])
    denom = math.sqrt(ll * rr)
    correlation = (lr / denom) if denom > 0 else 0.0
    gain_ratio_db = _db(math.sqrt(rr / ll)) if ll > 0 else math.inf

    ref_sq = max(ll, rr)
    mid_energy = max(0.0, (ll + 2.0 * lr + rr) / 4.0)
    mono_loss_db = _db(math.sqrt(mid_energy / ref_sq)) if ref_sq > 0 else -math.inf

    norm_gain = (10.0 ** (NORMALISE_REF_DBFS / 20.0) / overall_peak if overall_peak > 0 else 0.0)
    mono_sum_db = _db(mid_peak * norm_gain)

    side_sq = max(0.0, (ll - 2.0 * lr + rr) / 4.0)
    mid_sq = max(0.0, mid_energy)

    ref_peak = max(mid_peak, side_peak)
    ref_rms = math.sqrt(max(mid_sq, side_sq) / n)
    side_peak_db = _db(side_peak / ref_peak) if ref_peak > 0 else -math.inf
    side_rms_db = _db(math.sqrt(side_sq / n) / ref_rms) if ref_rms > 0 else -math.inf

    if exact:
        kind = StereoKind.IDENTICAL
    elif side_rms_db <= threshold_db and side_peak_db <= threshold_db + 20.0:
        kind = StereoKind.NEAR_IDENTICAL
    elif correlation <= -0.999:
        kind = StereoKind.INVERTED
    elif correlation >= 0.9999 and abs(gain_ratio_db) > 0.05:
        kind = StereoKind.SCALED
    else:
        kind = StereoKind.STEREO

    if correlation <= -0.999:
        polarity = PolarityFlag.INVERTED
    elif mono_sum_db < cancel_db:
        polarity = PolarityFlag.CANCELS
    else:
        polarity = PolarityFlag.OK

    return StereoAnalysis(
        kind=kind, side_peak_db=side_peak_db, side_rms_db=side_rms_db,
        correlation=correlation, gain_ratio_db=gain_ratio_db,
        peak_dbfs=peak_dbfs, polarity=polarity, mono_loss_db=mono_loss_db,
        mono_sum_db=mono_sum_db, **meter)


ENVELOPE_BUCKETS = 8192


@dataclass
class Envelope:
    mins: np.ndarray
    maxs: np.ndarray
    side_min: np.ndarray
    side_max: np.ndarray
    frames: int
    samplerate: int
    channels: int
    peak_frame: int = 0
    peak_magnitude: float = 0.0
    provisional: bool = False

    @property
    def duration(self) -> float:
        return self.frames / self.samplerate if self.samplerate else 0.0


def rebucket(mins: np.ndarray, maxs: np.ndarray, width: int):
    buckets = mins.shape[1]
    width = max(1, int(width))
    if width >= buckets:
        return (np.ascontiguousarray(mins, dtype=np.float32),
                np.ascontiguousarray(maxs, dtype=np.float32))
    edges = np.linspace(0, buckets, width + 1, dtype=np.int64)[:-1]
    lo = np.minimum.reduceat(mins, edges, axis=1)
    hi = np.maximum.reduceat(maxs, edges, axis=1)
    return (np.ascontiguousarray(lo, dtype=np.float32),
            np.ascontiguousarray(hi, dtype=np.float32))


def build_envelope(path: Path | str, buckets: int = ENVELOPE_BUCKETS,
                   block: int = 1 << 16) -> Envelope:
    path = Path(path)
    info = sf.info(str(path))
    channels = max(1, info.channels)
    frames = int(info.frames)
    buckets = max(1, min(buckets, frames or 1))

    mins = np.full((channels, buckets), np.inf, dtype=np.float64)
    maxs = np.full((channels, buckets), -np.inf, dtype=np.float64)
    side_min = np.full(buckets, np.inf, dtype=np.float64)
    side_max = np.full(buckets, -np.inf, dtype=np.float64)

    boundaries = np.linspace(0, frames, buckets + 1, dtype=np.int64)

    peak_magnitude, peak_frame = 0.0, 0
    position = 0

    with sf.SoundFile(str(path)) as handle:
        for chunk in handle.blocks(blocksize=block, dtype="float32",
                                   always_2d=True):
            taken = chunk.shape[0]
            if not taken:
                continue

            first = max(0, int(np.searchsorted(boundaries, position, "right")) - 1)
            last = max(first, min(buckets - 1,
                                  int(np.searchsorted(boundaries, position + taken - 1,
                                                      "right")) - 1))
            starts = np.maximum(boundaries[first:last + 1], position) - position

            block_min = np.minimum.reduceat(chunk, starts, axis=0).T
            block_max = np.maximum.reduceat(chunk, starts, axis=0).T
            np.minimum(mins[:, first:last + 1], block_min, out=mins[:, first:last + 1])
            np.maximum(maxs[:, first:last + 1], block_max, out=maxs[:, first:last + 1])

            if channels >= 2:
                side = (chunk[:, 0] - chunk[:, 1]) * 0.5
                s_lo = np.minimum.reduceat(side, starts)
                s_hi = np.maximum.reduceat(side, starts)
                np.minimum(side_min[first:last + 1], s_lo, out=side_min[first:last + 1])
                np.maximum(side_max[first:last + 1], s_hi, out=side_max[first:last + 1])

            flat = int(np.argmax(np.abs(chunk)))
            row, column = divmod(flat, chunk.shape[1])
            magnitude = float(abs(chunk[row, column]))
            if magnitude > peak_magnitude:
                peak_magnitude, peak_frame = magnitude, position + row

            position += taken

    for array in (mins, maxs):
        array[~np.isfinite(array)] = 0.0
    side_min[~np.isfinite(side_min)] = 0.0
    side_max[~np.isfinite(side_max)] = 0.0

    return Envelope(
        mins=np.ascontiguousarray(mins, dtype=np.float32),
        maxs=np.ascontiguousarray(maxs, dtype=np.float32),
        side_min=np.ascontiguousarray(side_min, dtype=np.float32),
        side_max=np.ascontiguousarray(side_max, dtype=np.float32),
        frames=position, samplerate=info.samplerate, channels=channels,
        peak_frame=peak_frame, peak_magnitude=peak_magnitude,
    )


COARSE_BUCKETS = 1024
COARSE_SLICE = 2048
COARSE_MIN_FRAMES = 1 << 22


def build_coarse_envelope(path: Path | str, buckets: int = COARSE_BUCKETS,
                          slice_frames: int = COARSE_SLICE) -> Envelope:
    path = Path(path)
    info = sf.info(str(path))
    channels = max(1, info.channels)
    frames = int(info.frames)
    buckets = max(1, min(buckets, frames or 1))
    boundaries = np.linspace(0, frames, buckets + 1, dtype=np.int64)

    mins = np.zeros((channels, buckets), dtype=np.float32)
    maxs = np.zeros((channels, buckets), dtype=np.float32)
    side_min = np.zeros(buckets, dtype=np.float32)
    side_max = np.zeros(buckets, dtype=np.float32)
    peak_magnitude, peak_frame = 0.0, 0

    with sf.SoundFile(str(path)) as handle:
        for index in range(buckets):
            start = int(boundaries[index])
            span = int(boundaries[index + 1]) - start
            if span <= 0:
                continue
            handle.seek(start)
            chunk = handle.read(min(span, slice_frames), dtype="float32",
                                always_2d=True)
            if not chunk.size:
                continue
            mins[:, index] = chunk.min(axis=0)
            maxs[:, index] = chunk.max(axis=0)
            if channels >= 2:
                side = (chunk[:, 0] - chunk[:, 1]) * 0.5
                side_min[index] = side.min()
                side_max[index] = side.max()
            local = int(np.argmax(np.abs(chunk)))
            row, column = divmod(local, chunk.shape[1])
            magnitude = float(abs(chunk[row, column]))
            if magnitude > peak_magnitude:
                peak_magnitude, peak_frame = magnitude, start + row

    return Envelope(mins=mins, maxs=maxs, side_min=side_min, side_max=side_max,
                    frames=frames, samplerate=info.samplerate,
                    channels=channels, peak_frame=peak_frame,
                    peak_magnitude=peak_magnitude, provisional=True)


def wants_coarse_pass(path: Path | str) -> bool:
    try:
        return int(sf.info(str(path)).frames) > COARSE_MIN_FRAMES
    except Exception:
        return False
