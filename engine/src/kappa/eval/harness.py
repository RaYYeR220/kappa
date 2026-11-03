"""The pre-registered rearing evaluation on the Sturman et al. (2020) open-field videos.

Every video goes through the same lifecycle a researcher would use (create, confirm
the fixed evaluation protocol, track, score rearing, write the manifest); the
accepted rearing bouts are then graded against the three raters with the statistics
in :mod:`kappa.eval.report`. Nothing here is tuned per video, and runs are cached, so
re-running the harness re-grades the stored outputs instead of calling models again.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kappa.agents.protocol_agent import last_model, locate_arena
from kappa.eval.datasets import fetch
from kappa.eval.report import INSTRUMENT, aggregate, bouts_from_events
from kappa.eval.sturman import SturmanLabels, compare_video, load_sturman
from kappa.model import BehaviourEvent, Span, Status
from kappa.runner import RunError, Runner

TUNING = ("OFT_12", "OFT_52", "OFT_54", "OFT_15", "OFT_41")
TEST = (
    "OFT_16", "OFT_14", "OFT_6", "OFT_50", "OFT_43", "OFT_5", "OFT_51", "OFT_23",
    "OFT_49", "OFT_38", "OFT_58", "OFT_11", "OFT_44", "OFT_24", "OFT_39",
)
# Kappa behaviour id → the raters' label it is graded against.
GRADED = {"rearing": "rear", "supported_rearing": "supported_rear", "unsupported_rearing": "unsupported_rear"}
SHIFT_S = 30.0


def with_union(labels: SturmanLabels) -> SturmanLabels:
    """The labels plus ``rear``: each rater's supported and unsupported rears together."""
    bouts = {
        video: {
            rater: {**by, "rear": sorted(by.get("supported_rear", []) + by.get("unsupported_rear", []))}
            for rater, by in raters.items()
        }
        for video, raters in labels.bouts.items()
    }
    return replace(labels, bouts=bouts, behaviours=(*labels.behaviours, "rear"))


def _events(state) -> list[BehaviourEvent]:
    return [
        BehaviourEvent(e["behaviour"], e["subject"], Span(*e["span"]), e["confidence"], e["source"], e.get("rationale", ""))
        for record in state.behaviours.values()
        for e in record.events
    ]


def prepare(runner: Runner, video_id: str, protocol: dict[str, Any], by: str, log: Callable[[str], None]) -> str:
    """Bring one video's run to the scored state, doing only the steps not yet done."""
    run_id = f"sturman-{video_id}"
    try:
        state = runner.load(run_id)
    except RunError:
        source = fetch(f"sturman_video_{video_id}")
        log(f"{video_id}: normalising")
        sampling = protocol.get("sampling") or {}
        runner.create(source, start_s=sampling.get("start_s", 0.0), end_s=sampling.get("end_s", 600.0), fps=sampling.get("fps"), run_id=run_id)
        state = runner.load(run_id)
    if state.confirmed_by is None:
        info = state.info
        frames = runner.keyframes(state, count=1)
        corners = locate_arena(runner.llm, frames[0], "rect", info.width, info.height)
        if corners is None:
            raise RunError(f"{video_id}: the arena corners could not be located")
        body = json.loads(json.dumps(protocol))
        body["arena"]["corners_px"] = [list(c) for c in corners]
        runner.confirm(run_id, body, by=by, via="cli")
        state = runner.load(run_id)
        state.arena_model = last_model()
        runner.save(state)
    if state.sam_main_sha256 is None:
        log(f"{video_id}: tracking")
        runner.track(run_id)
        state = runner.load(run_id)
    if "rearing" not in state.behaviours:
        log(f"{video_id}: scoring rearing")
        runner.score(run_id, "rearing")
    runner.manifest(run_id)
    return run_id


def grade(runner: Runner, labels: SturmanLabels, video_ids: Sequence[str]) -> dict[str, Any]:
    """Agreement with the raters over ``video_ids``, plus the controls and integrity counts."""
    reports: dict[str, list] = {b: [] for b in GRADED}
    shifted: list = []
    per_video = []
    integrity = {s.value: 0 for s in Status}
    for video_id in video_ids:
        state = runner.load(f"sturman-{video_id}")
        fps = state.info.fps
        analysis = runner.analysis(state)
        for r in analysis.results:
            integrity[r.status.value] += 1
        events = _events(state)
        row: dict[str, Any] = {"video": video_id, "run_id": state.run_id}
        for behaviour, label in GRADED.items():
            bouts = bouts_from_events(events, behaviour, fps)
            report = compare_video(labels, video_id, label, bouts, fps=fps)
            reports[behaviour].append(report)
            row[behaviour] = {
                "count": report.instrument_count,
                "rater_counts": dict(report.rater_counts),
                "mean_kappa": report.instrument_mean_kappa,
            }
        duration = state.info.n_frames / fps
        rear = bouts_from_events(events, "rearing", fps)
        moved = [((a + SHIFT_S) % duration, (b + SHIFT_S) % duration) for a, b in rear]
        moved = [(a, b) for a, b in moved if b > a]
        shifted.append(compare_video(labels, video_id, "rear", moved, fps=fps))
        per_video.append(row)
    datasets = {b: aggregate(rs) for b, rs in reports.items() if rs}
    control = aggregate(shifted) if shifted else None
    return {"datasets": datasets, "shifted": control, "per_video": per_video, "integrity": integrity}


def summary(graded: dict[str, Any], test_videos: Sequence[str], preregistration_commit: str, config_commit: str | None) -> dict[str, Any]:
    """The ``kappa.validation/1`` document the web app's validation page reads."""
    rear = graded["datasets"]["rearing"]
    vs = [{"rater": p.b if p.a == INSTRUMENT else p.a, "kappa": round(p.kappa, 3)} for p in rear.instrument_pairs]
    humans = [{"raters": [p.a, p.b], "kappa": round(p.kappa, 3)} for p in rear.human_pairs]
    threshold = min(h["kappa"] for h in humans)
    mean = sum(v["kappa"] for v in vs) / len(vs)
    secondary = []
    for behaviour, label in (("supported_rearing", "Supported rearing"), ("unsupported_rearing", "Unsupported rearing")):
        ds = graded["datasets"].get(behaviour)
        if ds is not None and ds.human_kappa_range:
            lo, hi = ds.human_kappa_range
            secondary.append({"label": f"{label}, frame κ", "kappa": f"{ds.instrument_mean_kappa:.2f}", "humans": f"{lo:.2f}–{hi:.2f}"})
    for q in rear.quantities:
        if q.who == INSTRUMENT and q.quantity == "count":
            humans_r = [x.icc_2_1 for x in rear.quantities if x.who != INSTRUMENT and x.quantity == "count"]
            secondary.append(
                {
                    "label": "Rear count per video, ICC(2,1) vs mean rater",
                    "kappa": f"{q.icc_2_1:.2f}",
                    "humans": f"{min(humans_r):.2f}–{max(humans_r):.2f}" if humans_r else None,
                    "note": f"Pearson r {q.r:.2f}",
                }
            )
    shifted = graded["shifted"]
    controls = []
    if shifted is not None:
        k = shifted.instrument_mean_kappa
        controls.append(
            {
                "id": "shifted",
                "description": f"Kappa's rearing bouts shifted circularly by {SHIFT_S:.0f} s",
                "expected": "κ near 0",
                "observed": f"κ {k:.3f}",
                "passed": abs(k) < 0.1,
            }
        )
    return {
        "schema": "kappa.validation/1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "preregistration_commit": preregistration_commit,
        "config_commit": config_commit,
        "behaviour": "rearing",
        "fps": 25,
        "test_videos": list(test_videos),
        "primary": {
            "kappa_vs_raters": vs,
            "human_vs_human": humans,
            "mean_kappa": round(mean, 3),
            "threshold": round(threshold, 3),
            "passed": bool(mean >= threshold),
        },
        "secondary": secondary,
        "negative_controls": controls,
        "integrity": graded["integrity"],
    }


def run(
    runner: Runner,
    video_ids: Sequence[str],
    protocol_path: Path,
    by: str,
    out: Path,
    preregistration_commit: str,
    config_commit: str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    labels = with_union(load_sturman(fetch("sturman_oft_labels")))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for video_id in video_ids:
        prepare(runner, video_id, protocol, by, log)
    graded = grade(runner, labels, video_ids)
    doc = summary(graded, video_ids, preregistration_commit, config_commit)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    detail = {
        "per_video": graded["per_video"],
        "datasets": {b: d.to_dict() for b, d in graded["datasets"].items()},
        "shifted": graded["shifted"].to_dict() if graded["shifted"] else None,
    }
    (out / "detail.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    return doc
