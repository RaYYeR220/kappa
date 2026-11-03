"""From stored model outputs to reported numbers, deterministically.

``analyse`` makes no model calls: it takes the raw segmenter outputs (the main
tracking run and any identity repairs), the confirmed protocol and the behaviour
verdicts, and rebuilds every track, event and number from them. Re-running it on
the same inputs gives the same manifest, which is what makes a run auditable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from kappa.geometry import Calibration, project, zone_raster
from kappa.integrity import check
from kappa.metrics import compute_metrics
from kappa.model import BehaviourEvent, IntegrityEvent, MetricResult, Span, SubjectTrack
from kappa.protocol import AssayProtocol
from kappa.repair import apply_resolutions, splice
from kappa.sam.parse import parse_output
from kappa.subjects import Assembly, assemble
from kappa.video import VideoInfo


class NeedsCalibration(ValueError):
    """The arena corners are unknown, so no position can be converted to centimetres."""


@dataclass(frozen=True)
class Repair:
    frame: int
    raw: str
    by: str


@dataclass
class Analysis:
    protocol: AssayProtocol
    calibration: Calibration
    assembly: Assembly
    tracks: dict[int, SubjectTrack]
    events: list[IntegrityEvent]
    results: list[MetricResult]
    analysed: Span
    n_detections: int
    parser_diagnostics: int
    resolutions: dict[str, str] = field(default_factory=dict)
    outside_arena: int = 0


ARENA_MARGIN = 0.04


def in_arena(detections, calib: Calibration, protocol: AssayProtocol):
    """Detections whose centre lies on the arena floor, and how many were dropped.

    A segmenter asked for "fish" also finds the fish's reflections in the tank walls
    and floor edge; those sit outside the floor polygon, where no subject can be. A
    margin of a few percent of the arena keeps animals pressed against a wall.
    """
    if not detections:
        return list(detections), 0
    xs = np.array([d.cx for d in detections])
    ys = np.array([d.cy for d in detections])
    x_cm, y_cm = calib.to_cm(xs, ys)
    w, h = protocol.arena.width_cm, protocol.arena.height_cm
    mx, my = ARENA_MARGIN * w, ARENA_MARGIN * h
    if protocol.arena.shape == "circle":
        r = w / 2
        keep = np.hypot(x_cm - r, y_cm - r) <= r + mx
    else:
        keep = (x_cm >= -mx) & (x_cm <= w + mx) & (y_cm >= -my) & (y_cm <= h + my)
    kept = [d for d, k in zip(detections, keep, strict=True) if k]
    return kept, len(detections) - len(kept)


def analyse(
    protocol: AssayProtocol,
    video: VideoInfo,
    raw: str,
    *,
    repairs: Sequence[Repair] = (),
    resolutions: Mapping[str, str] | None = None,
    behaviour_events: Sequence[BehaviourEvent] = (),
    scored_behaviours: Sequence[str] = (),
    extra_events: Sequence[IntegrityEvent] = (),
    analysed: Span | None = None,
    workers: int | None = None,
) -> Analysis:
    if protocol.arena.corners_px is None:
        raise NeedsCalibration("mark the arena corners before analysing")
    calib = Calibration(protocol.arena)
    rasters = {
        z.id: zone_raster(z, protocol.arena, calib, video.width, video.height) for z in protocol.zones
    }
    parsed = parse_output(raw, zone_rasters=rasters, workers=workers)
    policy = protocol.integrity
    px_per_cm = calib.px_per_cm
    detections, outside = in_arena(parsed.detections, calib, protocol)
    asm = assemble(
        detections, video.n_frames, video.fps, protocol.subject.expected_count, policy, px_per_cm
    )
    asm = check(asm, policy, px_per_cm)
    settled: dict[str, str] = dict(resolutions or {})
    diagnostics = len(parsed.diagnostics)
    for repair in sorted(repairs, key=lambda r: r.frame):
        retrack = parse_output(repair.raw, zone_rasters=rasters, workers=workers)
        diagnostics += len(retrack.diagnostics)
        kept, _ = in_arena(retrack.detections, calib, protocol)
        asm, settled_here = splice(asm, kept, repair.frame, policy, repair.by)
        settled.update(settled_here)
        asm = check(asm, policy, px_per_cm)
    events = apply_resolutions([*asm.events, *extra_events], settled)
    tracks = {sid: project(t, calib) for sid, t in asm.tracks.items()}
    span = analysed or Span(0, video.n_frames)
    results = compute_metrics(protocol, tracks, events, behaviour_events, span, scored_behaviours)
    return Analysis(
        protocol=protocol,
        calibration=calib,
        assembly=asm,
        tracks=tracks,
        events=events,
        results=results,
        analysed=span,
        n_detections=len(parsed.detections),
        outside_arena=outside,
        parser_diagnostics=diagnostics,
        resolutions=settled,
    )
