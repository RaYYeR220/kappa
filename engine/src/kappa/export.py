"""Export a finished run as a static bundle the web app can replay without a server.

A bundle is the run's normalised video, the segmenter's raw output (the browser
renders the masks itself with Meta's SAM client libraries), per-frame subject
tracks, the gated results and the manifest. Locked values are not in it: the
bundle carries exactly what a public reader is allowed to see.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from kappa.public import redact_manifest
from kappa.runner import Runner
from kappa.views import event_view as _event
from kappa.views import result_view as _result
from kappa.views import rounded as _round


def export_run(runner: Runner, run_id: str, dest: Path, title: str | None = None) -> Path:
    state = runner.load(run_id)
    analysis = runner.analysis(state)
    manifest = (
        runner.store.get_json(state.manifest_sha256)
        if state.manifest_sha256 and runner.store.has(state.manifest_sha256)
        else runner.manifest(run_id)
    )
    manifest = redact_manifest(manifest)
    out = dest / run_id
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(runner.video_path(state), out / "video.mp4")
    (out / "sam.txt").write_text(runner.store.get_text(state.sam_main_sha256), encoding="utf-8")
    for repair in state.repairs:
        (out / f"sam-repair-{repair.frame}.txt").write_text(runner.store.get_text(repair.raw_sha256), encoding="utf-8")
    tracks: dict[str, Any] = {
        str(sid): {
            "cx": _round(t.cx, 1),
            "cy": _round(t.cy, 1),
            "x_cm": _round(t.x_cm, 2),
            "y_cm": _round(t.y_cm, 2),
            "area_px": _round(t.area_px, 0),
            "interpolated": [int(v) for v in t.interpolated],
        }
        for sid, t in analysis.tracks.items()
    }
    verdicts = {
        behaviour: runner.store.get_json(record.verdicts_sha256) for behaviour, record in state.behaviours.items()
    }
    files = {
        "run.json": {
            "run_id": run_id,
            "title": title or analysis.protocol.title,
            "question": state.question,
            "source_name": state.source_name,
            "video": state.video,
            "video_sha256": state.video_sha256,
            "protocol": state.protocol,
            "protocol_sha256": analysis.protocol.content_hash(),
            "confirmed_by": state.confirmed_by,
            "assumptions": state.assumptions,
            "repairs": [r.model_dump(mode="json", exclude={"raw_sha256"}) | {"file": f"sam-repair-{r.frame}.txt"} for r in state.repairs],
            "corners_px": analysis.protocol.arena.corners_px,
        },
        "results.json": {
            "analysed": [analysis.analysed.start, analysis.analysed.end],
            "results": [_result(r, {e.id: e for e in analysis.events}, state.info.fps) for r in analysis.results],
            "events": [_event(e) for e in analysis.events],
        },
        "tracks.json": {"fps": state.info.fps, "n_frames": state.info.n_frames, "subjects": tracks},
        "verdicts.json": verdicts,
        "manifest.json": manifest,
    }
    for name, body in files.items():
        (out / name).write_text(json.dumps(body, separators=(",", ":")), encoding="utf-8")
    return out
