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
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from . import wavchunks
from .analysis import FileInfo, StereoAnalysis
from .dither import DitherMode, Quantiser
from .loudness import (NORDIC_ZERO_DBFS, VU_REFERENCE_DBFS, LoudnessMeter,
                       PPMMeter, VUMeter)

RESAMPLE_QUALITY = "VHQ"

SUBTYPES = {
    8: "PCM_U8",
    16: "PCM_16",
    24: "PCM_24",
    32: "PCM_32",
}

_CONTAINER_SUBTYPES = {
    ".aif": {8: "PCM_S8"}, ".aiff": {8: "PCM_S8"}, ".aifc": {8: "PCM_S8"},
    ".flac": {8: "PCM_S8", 32: None},
    ".ogg": {8: None, 16: None, 24: None, 32: None},
}


def _subtype_for(dest: Path, bits: int) -> str | None:
    override = _CONTAINER_SUBTYPES.get(dest.suffix.lower(), {})
    if bits in override:
        return override[bits]
    return SUBTYPES.get(bits)

class MonoMethod(Enum):
    AVERAGE = "average"
    LEFT = "left"
    RIGHT = "right"
    LOUDEST = "loudest"

    @property
    def label(self) -> str:
        return {
            "average": "Average (L+R)/2",
            "left": "Left channel",
            "right": "Right channel",
            "loudest": "Loudest channel",
        }[self.value]

class NormaliseMode(Enum):
    NONE = "none"
    LUFS = "lufs"
    PEAK = "peak"
    RMS = "rms"
    VU = "vu"
    PPM_NORDIC = "ppm"

    @property
    def label(self) -> str:
        return {"none": "Off", "lufs": "LUFS", "peak": "Peak", "rms": "RMS",
                "vu": "VU", "ppm": "PPM Nordic"}[self.value]

    @property
    def unit(self) -> str:
        return {"none": "", "lufs": "LUFS", "peak": "dBFS", "rms": "dBFS",
                "vu": "VU", "ppm": "PPM"}[self.value]

@dataclass(frozen=True)
class NormaliseSpec:
    mode: NormaliseMode = NormaliseMode.NONE
    target: float = -18.0
    ceiling_dbtp: float | None = -1.0
    relative: bool = False
    fixed_gain_db: float | None = None

    @property
    def label(self) -> str:
        if self.mode is NormaliseMode.NONE:
            return "Off"
        text = f"{self.mode.label} {self.target:g} {self.mode.unit}".strip()
        return f"{text} (rel)" if self.relative else text


def target_level_db(spec: NormaliseSpec) -> float:
    if spec.mode is NormaliseMode.PPM_NORDIC:
        return NORDIC_ZERO_DBFS + spec.target
    if spec.mode is NormaliseMode.VU:
        return VU_REFERENCE_DBFS + spec.target
    return spec.target


def measured_level_db(analysis: StereoAnalysis | None,
                      mode: NormaliseMode) -> float:
    if analysis is None:
        return -math.inf
    return {
        NormaliseMode.LUFS: analysis.loudness_lufs,
        NormaliseMode.PEAK: analysis.peak_dbfs,
        NormaliseMode.RMS: analysis.rms_dbfs,
        NormaliseMode.VU: analysis.vu_dbfs,
        NormaliseMode.PPM_NORDIC: analysis.ppm_dbfs,
    }.get(mode, -math.inf)


@dataclass
class RelativePlan:
    gain_db: float
    loudest: int                  
    loudest_level_db: float        
    level_after_db: float          
    held_by: str = ""              


def plan_relative(analyses: list[StereoAnalysis | None], spec: NormaliseSpec,
                  prevent_clipping: bool = True) -> RelativePlan | None:
    levels = [(measured_level_db(a, spec.mode), i)
              for i, a in enumerate(analyses)]
    usable = [(level, i) for level, i in levels if level > -math.inf]
    if not usable:
        return None

    loudest_level, loudest = max(usable, key=lambda pair: pair[0])
    target = target_level_db(spec)
    gain = target - loudest_level
    held = ""

    if spec.ceiling_dbtp is not None:
        peaks = [a.true_peak_dbfs for a in analyses
                 if a is not None and a.true_peak_dbfs > -math.inf]
        if peaks and max(peaks) + gain > spec.ceiling_dbtp:
            gain = spec.ceiling_dbtp - max(peaks)
            held = "Max true peak"
    elif prevent_clipping:
        peaks = [a.peak_dbfs for a in analyses
                 if a is not None and a.peak_dbfs > -200.0]
        if peaks and max(peaks) + gain > 0.0:
            gain = -max(peaks)
            held = "Clip guard"

    return RelativePlan(gain_db=gain, loudest=loudest,
                        loudest_level_db=loudest_level,
                        level_after_db=loudest_level + gain, held_by=held)

@dataclass
class TargetSpec:
    samplerate: int | None = None
    bit_depth: int | None = None
    to_mono: bool = False
    mono_method: MonoMethod = MonoMethod.AVERAGE
    dither: DitherMode = DitherMode.SHAPED
    prevent_clipping: bool = True
    preserve_chunks: bool = True
    normalise: NormaliseSpec = field(default_factory=NormaliseSpec)

@dataclass
class ConvertResult:
    source: Path
    dest: Path | None
    ok: bool
    skipped: bool = False
    message: str = ""
    gain_db: float = 0.0
    normalise_gain_db: float = 0.0
    measured_before: float = -math.inf
    lufs_after: float = -math.inf
    true_peak_after: float = -math.inf
    peak_dbfs: float = -math.inf
    chunks_carried: str = ""
    warnings: list[str] = field(default_factory=list)

def is_noop(info: FileInfo, spec: TargetSpec, dest: Path | None = None) -> bool:
    if spec.to_mono and info.channels > 1:
        return False
    if spec.samplerate and spec.samplerate != info.samplerate:
        return False
    if spec.bit_depth is not None:
        want = ("FLOAT" if spec.bit_depth == 0
                else _subtype_for(dest or info.path, spec.bit_depth))
        if want != info.subtype:
            return False
    if spec.normalise.mode is not NormaliseMode.NONE:
        return False
    if dest is not None and dest.suffix.lower() != info.path.suffix.lower():
        return False
    return True

BLOCK_FRAMES = 1 << 16


def _choose_mono_channel(src: Path, channels: int) -> int:
    energy = np.zeros(channels)
    with sf.SoundFile(str(src)) as handle:
        for block in handle.blocks(blocksize=BLOCK_FRAMES, dtype="float64",
                                   always_2d=True):
            if block.size:
                energy += np.einsum("ij,ij->j", block, block)
    return int(np.argmax(energy))


class _Pipeline:

    def __init__(self, src: Path, spec: TargetSpec, dst_rate: int,
                 mono_channel: int = 0) -> None:
        self.src, self.spec = src, spec
        self.dst_rate, self.mono_channel = dst_rate, mono_channel

    def _fold(self, block: np.ndarray) -> np.ndarray:
        if not (self.spec.to_mono and block.shape[1] > 1):
            return block
        method = self.spec.mono_method
        if method is MonoMethod.LEFT:
            return block[:, :1]
        if method is MonoMethod.RIGHT:
            return block[:, 1:2]
        if method is MonoMethod.LOUDEST:
            index = self.mono_channel
            return block[:, index:index + 1]
        return block.mean(axis=1, keepdims=True)

    def __iter__(self):
        with sf.SoundFile(str(self.src)) as handle:
            total = len(handle)
            channels = handle.channels
            out_channels = 1 if (self.spec.to_mono and channels > 1) else channels
            resampler = None
            if self.dst_rate != handle.samplerate:
                resampler = soxr.ResampleStream(
                    handle.samplerate, self.dst_rate, out_channels,
                    dtype="float64", quality=RESAMPLE_QUALITY)

            read = 0
            while True:
                block = handle.read(BLOCK_FRAMES, dtype="float64",
                                    always_2d=True)
                taken = block.shape[0]
                read += taken
                last = taken == 0 or read >= total
                if taken:
                    raw_peak = float(np.abs(block).max())
                    staged = self._fold(block)
                    folded_peak = float(np.abs(staged).max()) if staged.size else 0.0
                else:
                    raw_peak = folded_peak = 0.0
                    staged = np.zeros((0, out_channels))

                if resampler is not None:
                    staged = resampler.resample_chunk(
                        np.ascontiguousarray(staged), last=last)
                if staged.size:
                    yield staged, raw_peak, folded_peak
                elif taken:
                    yield np.zeros((0, out_channels)), raw_peak, folded_peak
                if last:
                    return


@dataclass
class _Measured:
    frames: int = 0
    peak: float = 0.0
    true_peak: float = 0.0
    raw_peak: float = 0.0          
    folded_peak: float = 0.0      
    level: float = -math.inf       
    has_signal: bool = False


def _measure(pipeline: _Pipeline, rate: int, mode: NormaliseMode,
             want_true_peak: bool) -> _Measured:
    found = _Measured()
    channels_seen = 0
    energy = np.zeros(0)
    lufs = vu = ppm = None
    upsampler = None

    for staged, raw_peak, folded_peak in pipeline:
        found.raw_peak = max(found.raw_peak, raw_peak)
        found.folded_peak = max(found.folded_peak, folded_peak)
        if not staged.size:
            continue
        if not channels_seen:
            channels_seen = staged.shape[1]
            energy = np.zeros(channels_seen)
            if mode is NormaliseMode.LUFS:
                lufs = LoudnessMeter(rate, channels_seen)
            elif mode is NormaliseMode.VU:
                vu = VUMeter(rate, channels_seen)
            elif mode is NormaliseMode.PPM_NORDIC:
                ppm = PPMMeter(rate)
            if want_true_peak:
                upsampler = soxr.ResampleStream(
                    rate, rate * 4, channels_seen, dtype="float32",
                    quality="HQ")

        found.frames += staged.shape[0]
        found.peak = max(found.peak, float(np.abs(staged).max()))
        if staged.any():
            found.has_signal = True
        if mode is NormaliseMode.RMS:
            energy += np.einsum("ij,ij->j", staged, staged)
        if lufs is not None:
            lufs.process(staged)
        if vu is not None:
            vu.process(staged)
        if ppm is not None:
            ppm.process(staged)
        if upsampler is not None:
            up = upsampler.resample_chunk(
                np.ascontiguousarray(staged, dtype=np.float32))
            if up.size:
                found.true_peak = max(found.true_peak, float(np.abs(up).max()))

    if upsampler is not None:
        tail = upsampler.resample_chunk(
            np.zeros((0, max(1, channels_seen)), dtype=np.float32), last=True)
        if tail.size:
            found.true_peak = max(found.true_peak, float(np.abs(tail).max()))
        found.true_peak = max(found.true_peak, found.peak)

    if mode is NormaliseMode.PEAK:
        found.level = 20.0 * math.log10(found.peak) if found.peak > 0 else -math.inf
    elif mode is NormaliseMode.RMS and found.frames:
        loudest = float(np.max(energy)) / found.frames if energy.size else 0.0
        found.level = 10.0 * math.log10(loudest) if loudest > 0 else -math.inf
    elif lufs is not None:
        found.level = lufs.integrated()[0]
    elif vu is not None:
        found.level = vu.dbfs
    elif ppm is not None:
        found.level = ppm.dbfs
    return found

def convert(
    info: FileInfo,
    spec: TargetSpec,
    dest: Path,
    on_noop: str = "copy",
) -> ConvertResult:
    src = info.path
    warnings: list[str] = []

    if on_noop != "process" and is_noop(info, spec, dest):
        if on_noop == "skip":
            return ConvertResult(src, None, True, skipped=True,
                                 message="Already in target format")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return ConvertResult(src, dest, True, skipped=True,
                             message="Copied unchanged (already in target format)")

    try:
        with sf.SoundFile(str(src)) as handle:
            src_rate, channels = handle.samplerate, handle.channels
    except Exception as exc:
        return ConvertResult(src, None, False, message=f"Read failed: {exc}")

    dst_rate = spec.samplerate or src_rate
    mono_channel = 0
    if (spec.to_mono and channels > 1
            and spec.mono_method is MonoMethod.LOUDEST):
        try:
            mono_channel = _choose_mono_channel(src, channels)
        except Exception as exc:
            return ConvertResult(src, None, False, message=f"Read failed: {exc}")

    def pipeline() -> _Pipeline:
        return _Pipeline(src, spec, dst_rate, mono_channel)

    normalising = (spec.normalise.mode is not NormaliseMode.NONE
                   and spec.normalise.fixed_gain_db is None)
    wants_ceiling = (normalising and spec.normalise.ceiling_dbtp is not None)
    measured = None
    if normalising or spec.prevent_clipping:
        try:
            measured = _measure(pipeline(), dst_rate, spec.normalise.mode,
                                want_true_peak=wants_ceiling)
        except Exception as exc:
            return ConvertResult(src, None, False, message=f"Read failed: {exc}")

    if measured is not None and spec.to_mono and channels > 1:
        before, after = measured.raw_peak, measured.folded_peak
        if before > 0 and after < before * 0.708:
            lost = 20 * math.log10(after / before) if after > 0 else -math.inf
            warnings.append(
                f"Mono fold lost {abs(lost):.1f} dB of peak — channels are "
                f"partly out of phase; consider a single-channel fold instead"
            )

    norm_gain_db = 0.0
    measured_before = -math.inf
    norm_true_peak = None
    if spec.normalise.mode is not NormaliseMode.NONE:
        if spec.normalise.fixed_gain_db is not None:
            norm_gain_db = spec.normalise.fixed_gain_db
        elif measured is not None and measured.has_signal:
            measured_before = measured.level
            if measured_before == -math.inf:
                warnings.append("Too quiet to measure; left at source level")
            else:
                norm_gain_db = target_level_db(spec.normalise) - measured_before
                if wants_ceiling and measured.true_peak > 0:
                    norm_true_peak = measured.true_peak
                    after = (20.0 * math.log10(norm_true_peak) + norm_gain_db)
                    ceiling = spec.normalise.ceiling_dbtp
                    if after > ceiling:
                        held = norm_gain_db - (after - ceiling)
                        warnings.append(
                            f"{spec.normalise.mode.label} target needed "
                            f"{norm_gain_db:+.2f} dB; held to {held:+.2f} dB "
                            f"by the {ceiling:g} dBTP ceiling")
                        norm_gain_db = held

    gain_db = 0.0
    hard_clipped = False
    if measured is not None:
        peak_after = measured.peak * (10.0 ** (norm_gain_db / 20.0))
        if peak_after > 1.0:
            if spec.prevent_clipping:
                trim = (1.0 - 1e-9) / peak_after
                gain_db = 20 * math.log10(trim)
                warnings.append(f"Peak was {20 * math.log10(peak_after):+.2f} dBFS; "
                                f"applied {gain_db:.2f} dB to avoid clipping")
            else:
                hard_clipped = True

    total_gain = 10.0 ** ((norm_gain_db + gain_db) / 20.0)

    bit_depth = spec.bit_depth
    if bit_depth is None:
        bit_depth = info.bit_depth or 24
    float_out = bit_depth == 0

    folding = spec.to_mono and info.channels > 1
    untouched = (dst_rate == src_rate
                 and not folding
                 and abs(total_gain - 1.0) < 1e-12
                 and info.bit_depth and bit_depth >= info.bit_depth)

    mode = spec.dither
    if untouched:
        mode = DitherMode.NONE
    elif measured is not None and not measured.has_signal:
        mode = DitherMode.NONE

    subtype = "FLOAT" if float_out else _subtype_for(dest, bit_depth)
    if subtype is None:
        return ConvertResult(
            src, None, False,
            message=f"{bit_depth}-bit is not supported in a "
                    f"{dest.suffix.lstrip('.').upper()} file")

    dest.parent.mkdir(parents=True, exist_ok=True)
    final_peak = 0.0
    clipped_samples = 0
    frames_written = 0
    after_meter = (LoudnessMeter(dst_rate, 1 if spec.to_mono else channels)
                   if spec.normalise.mode is NormaliseMode.LUFS else None)
    quantiser = None
    raw_peak = folded_peak = 0.0

    try:
        with sf.SoundFile(str(dest), "w", samplerate=dst_rate,
                          channels=1 if (spec.to_mono and channels > 1) else channels,
                          subtype=subtype) as out:
            for staged, block_raw, block_folded in pipeline():
                raw_peak = max(raw_peak, block_raw)
                folded_peak = max(folded_peak, block_folded)
                if not staged.size:
                    continue
                if total_gain != 1.0:
                    staged = staged * total_gain
                if hard_clipped:
                    clipped_samples += int(np.count_nonzero(np.abs(staged) > 1.0))
                    staged = np.clip(staged, -1.0, 1.0)
                final_peak = max(final_peak, float(np.abs(staged).max()))
                frames_written += staged.shape[0]
                if after_meter is not None:
                    after_meter.process(staged)
                if float_out:
                    out.write(staged.astype(np.float32))
                else:
                    if quantiser is None:
                        quantiser = Quantiser(bit_depth, mode, dst_rate,
                                              staged.shape[1])
                    out.write(quantiser.process(staged))
    except Exception as exc:
        return ConvertResult(src, None, False, message=f"Write failed: {exc}")

    if hard_clipped and clipped_samples:
        warnings.append(f"Clipped {clipped_samples} samples")

    if measured is None and spec.to_mono and channels > 1:
        if raw_peak > 0 and folded_peak < raw_peak * 0.708:
            lost = (20 * math.log10(folded_peak / raw_peak)
                    if folded_peak > 0 else -math.inf)
            warnings.append(
                f"Mono fold lost {abs(lost):.1f} dB of peak — channels are "
                f"partly out of phase; consider a single-channel fold instead"
            )

    carried = ""
    if spec.preserve_chunks and src.suffix.lower() == ".wav" and dest.suffix.lower() == ".wav":
        chunks = wavchunks.read_chunks(src)
        if chunks:
            chunks = wavchunks.rescale(chunks, src_rate, dst_rate)
            if wavchunks.append_chunks(dest, chunks):
                carried = wavchunks.describe(chunks)

    final_peak_db = 20 * math.log10(final_peak) if final_peak > 0 else -math.inf

    true_peak_after = -math.inf
    if final_peak > 0:
        if norm_true_peak and not hard_clipped:
            scaled = norm_true_peak * total_gain
            true_peak_after = 20 * math.log10(scaled)
        elif measured is not None and measured.true_peak > 0 and not hard_clipped:
            true_peak_after = 20 * math.log10(measured.true_peak * total_gain)
        else:
            true_peak_after = final_peak_db

    return ConvertResult(
        src, dest, True,
        message="Converted",
        gain_db=gain_db,
        normalise_gain_db=norm_gain_db,
        measured_before=measured_before,
        lufs_after=(after_meter.integrated()[0] if after_meter is not None
                    else -math.inf),
        true_peak_after=true_peak_after,
        peak_dbfs=final_peak_db,
        chunks_carried=carried,
        warnings=warnings,
    )
