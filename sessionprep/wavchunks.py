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

import struct
from pathlib import Path

CARRY = {b"smpl", b"cue ", b"inst", b"LIST", b"acid", b"labl", b"ltxt", b"note"}
UINT32_MAX = 0xFFFFFFFF

def _walk(handle, want: set | None = None):
    handle.seek(0)
    header = handle.read(12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        return
    end = struct.unpack("<I", header[4:8])[0] + 8
    pos = 12
    while pos + 8 <= end:
        handle.seek(pos)
        head = handle.read(8)
        if len(head) < 8:
            return
        cid = head[:4]
        size = struct.unpack("<I", head[4:])[0]
        body = b""
        if want is None or cid in want:
            body = handle.read(size)
            if len(body) < size:
                return
        yield cid, pos, size, body
        pos += 8 + size + (size & 1)


def read_chunks(path: Path | str) -> dict[bytes, bytes]:
    path = Path(path)
    try:
        with Path(path).open("rb") as handle:
            return {cid: body
                    for cid, _pos, _size, body in _walk(handle, CARRY)
                    if cid in CARRY}
    except OSError:
        return {}


def _rescale_smpl(body: bytes, ratio: float, new_rate: int) -> bytes:
    if len(body) < 36:
        return body
    out = bytearray(body)

    period = min(UINT32_MAX, max(1, int(round(1_000_000_000 / new_rate))))
    struct.pack_into("<I", out, 8, period)

    num_loops = struct.unpack_from("<I", out, 28)[0]
    base = 36
    for i in range(num_loops):
        off = base + i * 24
        if off + 24 > len(out):
            break
        start, end_sample = struct.unpack_from("<II", out, off + 8)

        new_start = min(UINT32_MAX, max(0, int(round(start * ratio))))
        new_end = min(UINT32_MAX, max(0, int(round(end_sample * ratio))))

        struct.pack_into("<II", out, off + 8, new_start, new_end)
    return bytes(out)

def _rescale_cue(body: bytes, ratio: float) -> bytes:
    if len(body) < 4:
        return body
    out = bytearray(body)
    count = struct.unpack_from("<I", out, 0)[0]
    for i in range(count):
        off = 4 + i * 24
        if off + 24 > len(out):
            break
        position = struct.unpack_from("<I", out, off + 4)[0]
        sample_offset = struct.unpack_from("<I", out, off + 20)[0]

        new_pos = min(UINT32_MAX, max(0, int(round(position * ratio))))
        new_offset = min(UINT32_MAX, max(0, int(round(sample_offset * ratio))))

        struct.pack_into("<I", out, off + 4, new_pos)
        struct.pack_into("<I", out, off + 20, new_offset)
    return bytes(out)


def rescale(chunks: dict[bytes, bytes], src_rate: int, dst_rate: int) -> dict[bytes, bytes]:
    if not chunks:
        return {}
    ratio = dst_rate / src_rate
    out = dict(chunks)
    if b"smpl" in out:
        out[b"smpl"] = _rescale_smpl(out[b"smpl"], ratio, dst_rate)
    if b"cue " in out and ratio != 1.0:
        out[b"cue "] = _rescale_cue(out[b"cue "], ratio)
    return out

def append_chunks(path: Path | str, chunks: dict[bytes, bytes]) -> bool:
    if not chunks:
        return False
    path = Path(path)
    try:
        with path.open("r+b") as handle:
            existing = {cid for cid, _pos, _size, _body in _walk(handle, set())}
            if not existing:
                return False   

            payload = bytearray()
            for cid, body in chunks.items():
                if cid in existing:
                    continue
                payload += cid + struct.pack("<I", len(body)) + body
                if len(body) & 1:
                    payload += b"\x00"
            if not payload:
                return False

            handle.seek(0, 2)
            total_size = handle.tell() + len(payload)
            if total_size - 8 > UINT32_MAX:
                return False

            handle.write(bytes(payload))
            handle.seek(4)
            handle.write(struct.pack("<I", total_size - 8))
    except OSError:
        return False
    return True

def describe(chunks: dict[bytes, bytes]) -> str:
    bits = []
    smpl = chunks.get(b"smpl")
    if smpl and len(smpl) >= 32:
        loops = struct.unpack_from("<I", smpl, 28)[0]
        if loops:
            bits.append(f"{loops} loop{'s' if loops != 1 else ''}")
    cue = chunks.get(b"cue ")
    if cue and len(cue) >= 4:
        count = struct.unpack_from("<I", cue, 0)[0]
        if count:
            bits.append(f"{count} marker{'s' if count != 1 else ''}")
    if b"inst" in chunks:
        bits.append("inst")
    return ", ".join(bits)