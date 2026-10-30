"""Identity repair: a person settles who is who on one frame, the segmenter re-tracks from there.

When the instrument can't tell two animals apart after they touch, it locks their
numbers and asks for a click. The person marks each subject on the frame right after
the contact; SAM 3.1 re-tracks every subject forward from that frame under the ids
the person gave. Frames before the anchor keep their original track; frames from the
anchor on come from the re-track. The identity doubt the person settled is marked
resolved by them, and every other check runs again on the spliced track, so a wrong
click shows up as a new event instead of passing silently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from meta_sam_parser import FramePrompt, PromptBox, PromptObject, PromptPoint

from kappa.model import IDENTITY_KINDS, Detection, IntegrityEvent, SubjectTrack
from kappa.subjects import Assembly, _fill_gaps, _put


@dataclass(frozen=True)
class Anchor:
    """One subject marked by a person on one frame (a box, a point, or both)."""

    subject: int
    frame: int
    box: tuple[int, int, int, int] | None = None
    point: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.box is None and self.point is None:
            raise ValueError("an anchor needs a box or a point")


def prompt_objects(anchors: Sequence[Anchor]) -> list[PromptObject]:
    """Segmenter prompts that label each marked object with its subject number."""
    frames = {a.frame for a in anchors}
    if len(frames) != 1:
        raise ValueError("all anchors of one repair must be on the same frame")
    if len({a.subject for a in anchors}) != len(anchors):
        raise ValueError("each subject can be marked once per repair")
    objects = []
    for a in anchors:
        box = PromptBox(*a.box) if a.box is not None else None
        points = (PromptPoint(*a.point),) if a.point is not None else ()
        objects.append(PromptObject(id=a.subject, prompts=[FramePrompt(frame=a.frame, box=box, points=points)]))
    return objects


def _cut(track: SubjectTrack, frame: int) -> SubjectTrack:
    t = replace(
        track,
        present=track.present.copy(),
        cx=track.cx.copy(),
        cy=track.cy.copy(),
        area_px=track.area_px.copy(),
        major_px=track.major_px.copy(),
        minor_px=track.minor_px.copy(),
        angle_rad=track.angle_rad.copy(),
        confidence=track.confidence.copy(),
        source=track.source.copy(),
        interpolated=track.interpolated.copy(),
        x_cm=None,
        y_cm=None,
        zone_fraction={k: v.copy() for k, v in track.zone_fraction.items()},
    )
    t.present[frame:] = False
    t.interpolated[frame:] = False
    t.source[frame:] = -1
    for arr in (t.cx, t.cy, t.area_px, t.major_px, t.minor_px, t.angle_rad, t.confidence):
        arr[frame:] = np.nan
    for arr in t.zone_fraction.values():
        arr[frame:] = np.nan
    return t


def splice(
    assembly: Assembly,
    retrack: Sequence[Detection],
    frame: int,
    policy,
    by: str,
    tolerance_frames: int = 0,
) -> tuple[Assembly, dict[str, str]]:
    """Replace every subject's track from ``frame`` on with the person-anchored re-track.

    ``retrack`` detections carry the subject number as their track id. Returns the
    spliced assembly (to be re-checked) and the resolutions to apply after checking:
    identity events whose repair frame is the anchor frame (within ``tolerance_frames``)
    are resolved by ``by``. Earlier identity doubts stay open — the anchor says who is
    who from here on, not who was who before.
    """
    tracks = {}
    for sid, track in assembly.tracks.items():
        cut = _cut(track, frame)
        for d in retrack:
            if d.track_id == sid and d.frame >= frame:
                _put(cut, d.frame, d)
        tracks[sid] = cut
    gap_events: list[IntegrityEvent] = []
    for track in tracks.values():
        gap_events.extend(e for e in _fill_gaps(track, policy) if e.span.end > frame)
    kept = [e for e in assembly.events if e.span.start < frame]
    resolutions = {
        e.id: by
        for e in assembly.events
        if e.kind in IDENTITY_KINDS
        and e.fix_frame is not None
        and frame - tolerance_frames <= e.fix_frame <= frame
    }
    seen = {e.id for e in kept}
    events = kept + [e for e in gap_events if e.id not in seen]
    return replace(assembly, tracks=tracks, events=events), resolutions


def apply_resolutions(events: Sequence[IntegrityEvent], resolutions: Mapping[str, str]) -> list[IntegrityEvent]:
    return [
        replace(e, resolved_by=resolutions[e.id]) if e.id in resolutions and e.resolved_by is None else e
        for e in events
    ]
