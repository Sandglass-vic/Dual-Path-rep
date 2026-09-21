#!/usr/bin/env python3
"""Read marker-prefixed JSON records out of a benchmark log.

`srun --label` processes task output line by line and splits any line longer
than 1000 characters, re-emitting the task label on the continuation. A result
record is well over that, so a single record can arrive as several log lines.
The reader below stitches those fragments back together instead of failing with
`JSONDecodeError: Unterminated string`. In a two-node phase both tasks emit at
once and their fragments interleave, so the fragments are buffered per task
label rather than in one shared buffer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# srun --label prefixes every emitted chunk with the task id.
LABEL = re.compile(r"^\d+:\s?")
# A record never needs more fragments than this; the guard keeps an unrelated
# log line from being glued onto a buffer forever.
MAX_FRAGMENTS = 16


def records_with_prefix(path: Path, prefix: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    buffers: dict[str, tuple[str, int]] = {}
    for raw in Path(path).read_text().splitlines():
        match = LABEL.match(raw)
        label = match.group(0) if match else ""
        line = raw[len(label) :]
        index = line.find(prefix)
        if index >= 0:
            buffer, fragments = line[index + len(prefix) :], 1
        elif label in buffers:
            buffer, fragments = buffers[label]
            buffer += line
            fragments += 1
        else:
            continue
        try:
            records.append(json.loads(buffer))
        except json.JSONDecodeError:
            if fragments >= MAX_FRAGMENTS:
                buffers.pop(label, None)
            else:
                buffers[label] = (buffer, fragments)
            continue
        buffers.pop(label, None)
    if buffers:
        raise RuntimeError(
            f"truncated {prefix.strip()} record at the end of {path}"
        )
    return records
