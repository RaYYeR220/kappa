"""From a confirmed protocol and integrity-checked tracks to reported numbers.

Every metric is computed once over the analysed span, then read off per window (the
whole span and each bin), so a bin never sees a different path, visit or bout from
the whole. Each number then goes through the integrity gate on its own window, which
is how one bad stretch locks only the bins it touches.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from kappa.metrics import behaviour, kinematics, zones
from kappa.metrics.frames import SubjectView, bins
from kappa.metrics.gating import Outcome, Unmeasurable, gate
from kappa.model import BehaviourEvent, IntegrityEvent, MetricResult, Span, SubjectTrack
from kappa.protocol import AssayProtocol, MetricSpec

UNITS = {
    "distance": "cm",
    "mean_speed": "cm/s",
    "time_in_zone": "s",
    "fraction_time_in_zone": "%",
    "zone_entries": "count",
    "latency_to_zone": "s",
    "immobility_time": "s",
    "immobility_bouts": "count",
    "mean_inter_subject_distance": "cm",
    "behaviour_count": "count",
    "behaviour_duration": "s",
}
PATHED = frozenset({"distance", "mean_speed", "immobility_time", "immobility_bouts"})
SMOOTHED = PATHED | {"mean_inter_subject_distance"}
ZONED = frozenset({"time_in_zone", "fraction_time_in_zone", "zone_entries", "latency_to_zone"})
BEHAVIOURAL = frozenset({"behaviour_count", "behaviour_duration"})
GROUP = "mean_inter_subject_distance"


def compute_metrics(
    protocol: AssayProtocol,
    tracks: Mapping[int, SubjectTrack],
    events: Sequence[IntegrityEvent],
    behaviour_events: Sequence[BehaviourEvent],
    analysed: Span,
) -> list[MetricResult]:
    """Every metric in ``protocol``, per subject and per bin, each verified, locked or refused.

    Results come in protocol order; within a metric, by subject, the whole span first
    and then its bins. Spacing between subjects is one result per window with
    ``subject=None``. Nothing here raises for awkward data: what cannot be measured
    comes back REFUSED with the reason.
    """
    session = _Session(protocol, tracks, behaviour_events, analysed)
    events = tuple(events)
    return [r for spec in protocol.metrics for r in session.report(spec, events)]


def describe(spec: MetricSpec, protocol: AssayProtocol, fps: float | None) -> dict[str, Any]:
    """The parameters that define a metric's number, enough to write its methods section."""
    p: dict[str, Any] = {"metric": spec.kind, "fps": fps}
    smoothing = protocol.smoothing
    if spec.kind in SMOOTHED:
        p["smoothing"] = smoothing.method
        if smoothing.method != "none":
            p["smoothing_window_s"] = smoothing.window_s
            if fps:
                p["smoothing_window_frames"] = kinematics.window_frames(smoothing, fps)
            if smoothing.method == "savgol":
                p["smoothing_polyorder"] = kinematics.SAVGOL_POLYORDER
    if spec.kind in {"distance", "mean_speed"}:
        p["min_step_cm"] = smoothing.min_step_cm
    if spec.kind in {"immobility_time", "immobility_bouts"}:
        p["speed"] = kinematics.SPEED_ESTIMATOR
        p["immobile_below_cm_s"] = spec.immobile_below_cm_s
        p["immobile_min_s"] = spec.immobile_min_s
    if spec.kind in ZONED:
        p.update(_zone_params(spec, protocol))
    if spec.kind in BEHAVIOURAL:
        p["behaviour"] = spec.behaviour
        found = next((b for b in protocol.behaviours if b.id == spec.behaviour), None)
        if found is not None:
            p["behaviour_definition"] = found.definition
            p["behaviour_detector"] = found.detector
    return p


def _zone_params(spec: MetricSpec, protocol: AssayProtocol) -> dict[str, Any]:
    arena, rule = protocol.arena, protocol.entry_rule
    p: dict[str, Any] = {
        "zone": spec.zone,
        "arena_shape": arena.shape,
        "arena_cm": (arena.width_cm, arena.height_cm),
        "entry_mode": rule.mode,
    }
    zone = next((z for z in protocol.zones if z.id == spec.zone), None)
    if zone is not None:
        p["zone_kind"] = zone.kind
        p.update(
            {
                "center_fraction": {"zone_area_fraction": zone.fraction},
                "periphery_band": {"zone_band_cm": zone.band_cm},
                "rect_cm": {"zone_rect_cm": zone.rect_cm},
            }[zone.kind]
        )
    if rule.mode == "body_fraction":
        p["entry_body_fraction"] = rule.body_fraction
    if spec.kind in {"zone_entries", "latency_to_zone"}:
        p["min_dwell_s"] = rule.min_dwell_s
    return p


class _Session:
    """One measurement run: the views, and the paths and visits derived from them once."""

    def __init__(
        self,
        protocol: AssayProtocol,
        tracks: Mapping[int, SubjectTrack],
        behaviour_events: Sequence[BehaviourEvent],
        analysed: Span,
    ) -> None:
        self.protocol = protocol
        self.analysed = analysed
        self.behaviour_events = tuple(behaviour_events)
        self.views = {sid: SubjectView.of(tracks[sid], analysed) for sid in sorted(tracks)}
        rates = {v.fps for v in self.views.values()}
        self.fps = next(iter(rates)) if len(rates) == 1 else None
        self.problem = self._session_problem(rates)
        self._paths: dict[int, kinematics.Path | Unmeasurable] = {}
        self._visits: dict[tuple[int, str], zones.ZoneVisits | Unmeasurable] = {}

    def _session_problem(self, rates: set[float]) -> str | None:
        """A reason that makes every number of this run unmeasurable, if there is one."""
        if not self.views:
            return "no subjects were tracked"
        if len(rates) > 1:
            return "the subject tracks disagree on the frame rate"
        if not (math.isfinite(self.fps) and self.fps > 0):
            return "the tracks have no valid frame rate"
        if self.analysed.length == 0:
            return "the analysed span is empty"
        return None

    def report(self, spec: MetricSpec, events: Sequence[IntegrityEvent]) -> list[MetricResult]:
        unit = UNITS.get(spec.kind, "")
        method = describe(spec, self.protocol, self.fps)
        if self.problem is not None:
            outcome = Unmeasurable(self.problem)
            return [gate(spec.id, unit, None, None, self.analysed, outcome, events, None, method)]
        windows: list[tuple[Span | None, Span]] = [(None, self.analysed)]
        size = 0
        if spec.bin_s is not None:
            size = max(1, round(spec.bin_s * self.fps))
            windows += [(b, b) for b in bins(self.analysed, spec.bin_s, self.fps)]
        subjects: list[int | None] = [None] if spec.kind == GROUP else list(self.views)
        results = []
        for sid in subjects:
            for bin_, window in windows:
                params = {**method, "duration_s": window.length / self.fps}
                if bin_ is not None:
                    params |= {"bin_s": spec.bin_s, "partial_bin": bin_.length < size}
                outcome = self.measure(spec, sid, window)
                depends = None if sid is None else (sid,)
                results.append(
                    gate(spec.id, unit, sid, bin_, window, outcome, events, depends, params)
                )
        return results

    def path(self, sid: int) -> kinematics.Path | Unmeasurable:
        if sid not in self._paths:
            view = self.views[sid]
            self._paths[sid] = (
                Unmeasurable(zones.NOT_PROJECTED)
                if view.x is None or view.y is None
                else kinematics.trace(view, self.protocol.smoothing)
            )
        return self._paths[sid]

    def visits(self, sid: int, zone_id: str) -> zones.ZoneVisits | Unmeasurable:
        key = (sid, zone_id)
        if key not in self._visits:
            self._visits[key] = zones.zone_visits(self.views[sid], self.protocol, zone_id)
        return self._visits[key]

    def measure(self, spec: MetricSpec, sid: int | None, window: Span) -> Outcome:
        if sid is None:
            return self._spacing(window)
        if not self.views[sid].observed_in(window):
            return Unmeasurable(
                f"subject {sid} was never observed in frames [{window.start}, {window.end})"
            )
        kind = spec.kind
        if kind in PATHED:
            path = self.path(sid)
            if isinstance(path, Unmeasurable):
                return path
            match kind:
                case "distance":
                    return kinematics.distance(path, window)
                case "mean_speed":
                    return kinematics.mean_speed(path, window)
                case "immobility_time":
                    return kinematics.immobility_time(path, spec, window)
                case "immobility_bouts":
                    return kinematics.immobility_bout_count(path, spec, window)
        if kind in ZONED:
            if spec.zone is None:
                return Unmeasurable(f"{kind} needs a zone")
            zv = self.visits(sid, spec.zone)
            if isinstance(zv, Unmeasurable):
                return zv
            match kind:
                case "time_in_zone":
                    return zones.time_in_zone(zv, window)
                case "fraction_time_in_zone":
                    return zones.fraction_time_in_zone(zv, window)
                case "zone_entries":
                    return zones.zone_entries(zv, window)
                case "latency_to_zone":
                    return zones.latency_to_zone(zv, window)
        if kind in BEHAVIOURAL:
            if spec.behaviour is None:
                return Unmeasurable(f"{kind} needs a behaviour")
            found = behaviour.bouts(self.behaviour_events, spec.behaviour, sid, self.analysed)
            count = kind == "behaviour_count"
            measure = behaviour.behaviour_count if count else behaviour.behaviour_duration
            return measure(found, window, spec.behaviour, sid, self.fps)
        return Unmeasurable(f"metric kind {kind} is not supported for a single subject")

    def _spacing(self, window: Span) -> Outcome:
        paths = [self.path(sid) for sid in self.views]
        refused = next((p for p in paths if isinstance(p, Unmeasurable)), None)
        if refused is not None:
            return refused
        return kinematics.inter_subject_distance(paths, window)
