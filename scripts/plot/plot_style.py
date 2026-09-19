"""Shared labels, title suffixes, and compact plotting colors."""

from __future__ import annotations

from typing import Any


COLORS = {
    # Okabe-Ito-inspired colors: distinct in print and on dark projectors.
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#7B61A8",
    "cyan": "#56B4E9",
    "brown": "#A6761D",
    "pink": "#CC79A7",
    "gray": "#F1F3F5",
    "yellow": "#F0E442",
    # Light neutral for queue wait. Stays a shade darker than the near-white
    # "gray" used for Misc so the two remain distinguishable.
    "lightgray": "#C9CDD2",
}


def metadata_title_suffix(metadata: dict[str, Any]) -> str:
    dataset = str(metadata.get("dataset", f"{metadata.get('max_len_k', '?')}K"))
    agents = int(metadata.get("agents", 0))
    path = str(metadata.get("l3_read_path", "prefill")).lower()
    path_label = "Decode-Path" if path == "decode" else "Prefill-Path"
    return f"-{dataset}-{agents}Agent-{path_label}"


def add_title_suffix(title: str, suffix: str) -> str:
    return f"{title}{suffix}" if suffix else title
