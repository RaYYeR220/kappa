"""The MCP tools keep the engine's refusals: no number without a person, no provisional values."""

import asyncio
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from pydantic import ValidationError
from run_fakes import FakeApi, make_video, one_mouse, protocol, two_mice

from kappa import library
from kappa import mcp_server as ms
from kappa.mcp_server import AnchorIn, ResourceError, ToolError
from kappa.model import Status
from kappa.runner import Runner

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return str(make_video(tmp_path_factory.mktemp("video") / "trial.mp4"))


@pytest.fixture
def runner(tmp_path, monkeypatch):
    r = Runner(home=tmp_path / "home")
    r._api = FakeApi(r.store, main=one_mouse())
    monkeypatch.setattr(ms, "runner_factory", lambda: r)
    return r


def test_nothing_is_measured_before_a_person_confirms(runner, video):
    created = ms.create_run(video)
    run_id = created["run_id"]
    assert created["video"]["n_frames"] == 150

    draft = ms.draft_protocol(run_id, "How far does the mouse walk and how long is it in the centre?")
    assert draft["confirmed"] is False and draft["missing"] == []
    assert draft["assumptions"] == ["Arena is 80 x 80 cm."] and draft["refused"] == []
    assert Path(draft["draft_file"]).exists()

    for call in (
        lambda: asyncio.run(ms.track(run_id)),
        lambda: ms.get_results(run_id),
        lambda: ms.get_manifest(run_id),
        lambda: ms.score_behaviour(run_id, "rearing"),
    ):
        with pytest.raises(ToolError, match="not been (confirmed|tracked)"):
            call()
    for name in ("", "   ", "assistant", "user"):
        with pytest.raises(ToolError, match="person"):
            ms.confirm_protocol(run_id, draft["draft"], name)
    assert runner.load(run_id).confirmed_by is None

    confirmed = ms.confirm_protocol(run_id, json.dumps(draft["draft"]), " Dr. Example ")
    assert confirmed["confirmed_by"] == "Dr. Example"
    assert confirmed["protocol_sha256"] == draft["draft_sha256"]

    tracked = asyncio.run(ms.track(run_id))
    assert tracked["summary"] == {"verified": 2, "locked": 0, "refused": 0}
    distance = next(r for r in ms.get_results(run_id)["results"] if r["metric_id"] == "distance")
    assert distance["status"] == "verified" and distance["value"] == pytest.approx(60, rel=0.02)

    manifest = ms.get_manifest(run_id)
    assert manifest["protocol"]["confirmed_by"] == "Dr. Example" and manifest["withheld"]
    assert Path(manifest["path"]).exists()
    assert json.loads(ms.run_manifest(run_id))["manifest_sha256"] == manifest["manifest_sha256"]
    rows = ms.list_runs()["runs"]
    assert [(r["run_id"], r["stage"], r["confirmed_by"]) for r in rows] == [(run_id, "tracked", "Dr. Example")]


def test_get_results_hides_provisional_values(runner, video):
    raw, fix, retrack, (a, b) = two_mice()
    runner._api = FakeApi(runner.store, main=raw, retrack=retrack)
    run_id = ms.create_run(video)["run_id"]
    ms.confirm_protocol(run_id, protocol(count=2), "Dr. Example")
    asyncio.run(ms.track(run_id))

    internal = runner.analysis(runner.load(run_id)).results
    assert internal and all(r.status is Status.LOCKED and r.provisional is not None for r in internal)

    view = ms.get_results(run_id)
    assert "provisional" not in json.dumps(view)
    for row in view["results"]:
        assert row["status"] == "locked" and row["value"] is None and row["uncertainty"] == {}
        assert row["fix_frame"] == fix and row["blocked_by"] and row["reasons"][0].startswith("contact")
    assert [b["fix_frame"] for b in view["open_blockers"]] == [fix]

    manifest = ms.get_manifest(run_id)
    for doc in (manifest, json.loads(ms.run_manifest(run_id))):
        assert all("provisional" not in r and r["evidence"] == [] and r["uncertainty"] == {} for r in doc["results"])
    audit = json.loads(Path(manifest["path"]).read_text())
    assert all(r["provisional"] is not None for r in audit["results"]), "the audit copy keeps them"

    with pytest.raises(ToolError, match="person"):
        ms.repair_identity(run_id, fix, [AnchorIn(subject=1, point=(1, 2))], "")
    with pytest.raises(ValidationError):
        AnchorIn(subject=1)
    marks = [
        AnchorIn(subject=1, point=(round(a[fix, 0]), round(a[fix, 1]))),
        AnchorIn(subject=2, box=(round(b[fix, 0]) - 20, round(b[fix, 1]) - 10, round(b[fix, 0]) + 20, round(b[fix, 1]) + 10)),
    ]
    repaired = ms.repair_identity(run_id, fix, marks, "A. Person")
    assert repaired["summary"]["locked"] == 0 and repaired["open_blockers"] == []
    assert all(r["status"] == "verified" and r["value"] is not None for r in ms.get_results(run_id)["results"])


def test_score_behaviour(runner, video):
    run_id = ms.create_run(video)["run_id"]
    ms.confirm_protocol(run_id, protocol(rearing=True), "Dr. Example")
    asyncio.run(ms.track(run_id))
    scored = ms.score_behaviour(run_id, "rearing")
    assert (scored["candidates_reviewed"], scored["reviews_failed"]) == (0, 0)
    assert [(r["metric_id"], r["status"], r["value"]) for r in scored["results"]] == [("rearing_count", "verified", 0)]
    with pytest.raises(ToolError, match="no behaviour grooming"):
        ms.score_behaviour(run_id, "grooming")


def test_plain_errors(runner, video, tmp_path):
    with pytest.raises(ToolError, match="no video at"):
        ms.create_run(str(tmp_path / "missing.mp4"))
    with pytest.raises(ToolError, match="not a run id"):
        ms.get_results("../../etc")
    with pytest.raises(ToolError, match="no run nope"):
        ms.get_results("nope")
    run_id = ms.create_run(video)["run_id"]
    with pytest.raises(ToolError, match="not valid JSON"):
        ms.confirm_protocol(run_id, "{nope", "Dr. Example")
    with pytest.raises(ToolError, match="the protocol is not valid: metrics"):
        ms.confirm_protocol(run_id, {**protocol(), "metrics": []}, "Dr. Example")
    with pytest.raises(ToolError, match="mark the arena corners"):
        ms.confirm_protocol(run_id, library.load("open_field"), "Dr. Example")
    with pytest.raises(ResourceError, match="no manifest yet"):
        ms.run_manifest(run_id)
    with pytest.raises(ResourceError, match="no run"):
        ms.run_manifest("nope")


def test_library_resource():
    entries = json.loads(ms.protocol_library())["protocols"]
    assert [e["name"] for e in entries] == library.names()
    assert all(e["needs"] == ["arena.corners_px"] for e in entries)


def test_over_the_wire(runner, video):
    client_module = pytest.importorskip("mcp.client.client")
    run_id = ms.create_run(video)["run_id"]
    ms.confirm_protocol(run_id, protocol(), "Dr. Example")
    progress: list[float] = []

    async def on_progress(value, total, message):
        progress.append(value)

    async def session():
        async with client_module.Client(ms.mcp) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            templates = [t.uri_template for t in (await client.list_resource_templates()).resource_templates]
            library_text = (await client.read_resource("kappa://protocols/library")).contents[0].text
            missing = await client.call_tool("get_results", {"run_id": "nope"})
            tracked = await client.call_tool("track", {"run_id": run_id}, progress_callback=on_progress)
            return tools, templates, library_text, missing, tracked

    tools, templates, library_text, missing, tracked = asyncio.run(session())
    assert tools == {
        "create_run", "draft_protocol", "confirm_protocol", "track", "get_results",
        "get_manifest", "list_runs", "repair_identity", "score_behaviour",
    }
    assert templates == ["kappa://runs/{run_id}/manifest"]
    assert len(json.loads(library_text)["protocols"]) == 3
    assert missing.is_error and "no run nope" in missing.content[0].text
    assert not tracked.is_error
    assert progress and progress[-1] == 150
