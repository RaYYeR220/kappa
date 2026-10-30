"""Candidate behaviour bouts from body-shape kinematics.

Seen from above, a mouse that rears up shows less of its back: its mask area drops
well below its own recent typical size while its centre stays put. Grooming shows
as a still body whose outline keeps changing. These rules are deliberately generous
— they only propose stretches worth a look; the adjudicator decides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from kappa.model import Span, SubjectTrack


@dataclass(frozen=True)
class CandidateRule:
    area_drop: float = 0.82
    baseline_s: float = 20.0
    min_s: float = 0.24
    merge_gap_s: float = 0.4
    pad_s: float = 0.6
    max_s: float = 12.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def _merge(runs: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def rearing_candidates(track: SubjectTrack, rule: CandidateRule | None = None) -> list[Span]:
    """Stretches where the visible body area falls below ``area_drop`` of its rolling median."""
    rule = rule or CandidateRule()
    fps, n = track.fps, track.n_frames
    area = np.where(track.present, track.area_px, np.nan)
    if not np.isfinite(area).any():
        return []
    filled = np.where(np.isfinite(area), area, np.nanmedian(area))
    window = max(3, int(rule.baseline_s * fps) | 1)
    baseline = median_filter(filled, size=window, mode="nearest")
    low = np.isfinite(area) & (area < rule.area_drop * baseline)
    runs = _merge(_runs(low), round(rule.merge_gap_s * fps))
    pad, shortest, longest = round(rule.pad_s * fps), round(rule.min_s * fps), round(rule.max_s * fps)
    spans = []
    for start, end in runs:
        if end - start < shortest:
            continue
        start, end = max(0, start - pad), min(n, end + pad)
        while end - start > longest:
            spans.append(Span(start, start + longest))
            start += longest
        spans.append(Span(start, end))
    return spans
