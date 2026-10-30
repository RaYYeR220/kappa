"""A locked behaviour count can't be rebuilt from its bouts or verdicts on any public surface.

A count is its accepted bouts counted, and each bout is a verdict that said yes, so
a surface that withholds the number withholds those too. The audit copy on disk
keeps everything.
"""

import asyncio
import json
import shutil
from pathlib import Path

import pytest
from run_fakes import FakeApi, make_video, protocol, two_mice

from kappa.export import export_run
from kappa.model import Status
from kappa.public import disclosed_behaviours, disclosed_verdicts, redact_manifest
from kappa.repair import Anchor
from kappa.runner import Runner


def _row(subject, status, behaviour="rearing", metric="rearing_count"):
    return {"metric_id": metric, "subject": subject, "status": status, "params": {"behaviour": behaviour} if behaviour else {}}


def _bout(subject, behaviour="rearing"):
    return {"behaviour": behaviour, "subject": subject, "span": {"start": 10, "end": 20}, "confidence": 0.9}


def test_bouts_leave_only_when_every_result_for_them_is_verified():
    results = [
        _row(1, "verified"), _row(1, "verified", metric="rearing_time"),
        _row(2, "verified"), _row(2, "locked"),  # one locked bin is enough to hold the bouts back
        _row(3, "refused"),
        _row(4, "verified", behaviour=None, metric="distance"),  # not a behaviour result
    ]
    disclosed = disclosed_behaviours(results)
    assert disclosed == {("rearing", 1)}
    events = [_bout(1), _bout(2), _bout(3), _bout(4), _bout(1, "grooming")]  # nothing measures grooming
    assert redact_manifest({"results": results, "behaviour_events": events})["behaviour_events"] == [_bout(1)]
    verdicts = [{"subject": s, "present": True} for s in (1, 2, 3, 4)]
    assert disclosed_verdicts(verdicts, "rearing", disclosed) == [{"subject": 1, "present": True}]
    assert disclosed_verdicts(verdicts, "grooming", disclosed) == []
    assert disclosed_behaviours([_row(None, "refused"), _row(1, "verified")]) == frozenset(), "a run-wide refusal"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_a_locked_count_shows_no_bouts_or_verdicts_anywhere(tmp_path, monkeypatch):
    ms = pytest.importorskip("kappa.mcp_server")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kappa.api import create_app

    raw, fix, retrack, (a, b) = two_mice(rearing=True)
    runner = Runner(home=tmp_path / "home")
    runner._api = FakeApi(runner.store, main=raw, retrack=retrack)
    monkeypatch.setattr(ms, "runner_factory", lambda: runner)
    run_id = ms.create_run(str(make_video(tmp_path / "trial.mp4")))["run_id"]
    runner.confirm(run_id, protocol(count=2, rearing=True), by="Dr. Example", via="cli")
    asyncio.run(ms.track(run_id))
    scored = ms.score_behaviour(run_id, "rearing")

    state = runner.load(run_id)
    bouts = state.behaviours["rearing"].events
    counts = [r for r in runner.analysis(state).results if r.metric_id == "rearing_count"]
    assert sorted(e["subject"] for e in bouts) == [1, 2], "each mouse reared once"
    assert [(r.status, r.provisional) for r in counts] == [(Status.LOCKED, 1.0)] * 2, "and the contact locks both counts"

    def public_outputs():
        manifest = ms.get_manifest(run_id)  # rebuilds the stored manifest the others serve
        client = TestClient(create_app(runner, public=True))
        bundle = export_run(runner, run_id, tmp_path / "bundle")
        shown = {
            "mcp get_manifest": manifest["behaviour_events"],
            "mcp manifest resource": json.loads(ms.run_manifest(run_id))["behaviour_events"],
            "api manifest": client.get(f"/runs/{run_id}/manifest").json()["behaviour_events"],
            "api verdicts": client.get(f"/runs/{run_id}/verdicts/rearing").json(),
            "export manifest": json.loads((bundle / "manifest.json").read_text())["behaviour_events"],
            "export verdicts": json.loads((bundle / "verdicts.json").read_text())["rearing"],
        }
        states = [client.get(f"/runs/{run_id}").json(), *client.get("/runs").json()]
        return manifest, shown, states

    manifest, shown, states = public_outputs()
    assert all(v == [] for v in shown.values()), shown
    assert all(s["behaviours"]["rearing"].keys() == {"verdicts_sha256", "failed_spans"} for s in states)
    assert (scored["candidates_reviewed"], scored["withheld_subjects"]) == (0, [1, 2])
    assert [r["value"] for r in scored["results"]] == [None, None]
    audit = json.loads(Path(manifest["path"]).read_text())
    assert len(audit["behaviour_events"]) == 2, "the audit copy on disk keeps every bout"

    # Once a person settles who is who, the counts are verified and their bouts may be shown.
    marks = [Anchor(1, fix, point=(round(a[fix, 0]), round(a[fix, 1]))), Anchor(2, fix, point=(round(b[fix, 0]), round(b[fix, 1])))]
    runner.repair(run_id, marks, by="A. Person", via="cli")
    _, shown, _ = public_outputs()
    assert all(sorted(e["subject"] for e in v) == [1, 2] for v in shown.values()), shown
