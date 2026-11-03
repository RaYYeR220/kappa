"""The MCP tools keep the engine's refusals: no number without a person, no provisional values.

No tool can confirm a protocol or apply a repair, so the person is simulated the
ways a person acts: the kappa command, or a Runner call as the web workspace makes.
"""

import asyncio
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from pydantic import ValidationError
from run_fakes import FakeApi, make_video, one_mouse, protocol, two_mice, with_pin

from kappa import cli, library
from kappa import mcp_server as ms
from kappa.mcp_server import AnchorIn, ResourceError, ToolError
from kappa.model import Status
from kappa.runner import Runner

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

TOOLS = {
    "create_run", "draft_protocol", "submit_for_confirmation", "track", "get_results",
    "get_manifest", "list_runs", "propose_repair", "score_behaviour",
}


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return str(make_video(tmp_path_factory.mktemp("video") / "trial.mp4"))


@pytest.fixture
def runner(tmp_path, monkeypatch):
    r = Runner(home=tmp_path / "home")
    r._api = FakeApi(r.store, main=one_mouse())
    monkeypatch.setattr(ms, "runner_factory", lambda: r)
    monkeypatch.setattr(cli, "runner_factory", lambda: r)
    return r


def person(*argv) -> int:
    """A person at the terminal, running the kappa command on the same runner."""
    return cli.main(with_pin(list(argv), cli.runner_factory()))


def test_nothing_is_measured_before_a_person_confirms(runner, video, capsys):
    created = ms.create_run(video)
    run_id = created["run_id"]
    assert created["video"]["n_frames"] == 150

    draft = ms.draft_protocol(run_id, "How far does the mouse walk and how long is it in the centre?")
    assert draft["confirmed"] is False and draft["missing"] == []
    assert draft["assumptions"] == ["Arena is 80 x 80 cm."] and draft["refused"] == []
    assert Path(draft["draft_file"]).exists()

    edited = {**draft["draft"], "title": "Open field, edited"}
    submitted = ms.submit_for_confirmation(run_id, json.dumps(edited))
    assert submitted["status"] == "awaiting_person" and submitted["confirmed_protocol_sha256"] is None
    assert submitted["command"] == f'kappa confirm {run_id} --sha256 {submitted["draft_sha256"][:16]} --by "<your name>"'
    assert "web workspace" in submitted["next"]
    assert json.loads(Path(submitted["draft_file"]).read_text())["title"] == "Open field, edited"

    # Whatever the agent calls, an unconfirmed run gives no number and takes no repair.
    assert not hasattr(ms, "confirm_protocol") and not hasattr(ms, "repair_identity")
    for call in (
        lambda: asyncio.run(ms.track(run_id)),
        lambda: ms.get_results(run_id),
        lambda: ms.get_manifest(run_id),
        lambda: ms.score_behaviour(run_id, "rearing"),
        lambda: ms.propose_repair(run_id, 10, [AnchorIn(subject=1, point=(1, 2))]),
    ):
        with pytest.raises(ToolError, match="not been (confirmed|tracked)"):
            call()
    state = runner.load(run_id)
    assert (state.protocol, state.confirmed_by, state.draft_source) == (None, None, "mcp")
    assert ms.list_runs()["runs"][0]["stage"] == "drafted"

    assert person("confirm", run_id, "--by", "Dr. Example") == 0
    assert "Confirmed by Dr. Example." in capsys.readouterr().out
    assert runner.load(run_id).protocol["title"] == "Open field, edited"

    tracked = asyncio.run(ms.track(run_id))
    assert tracked["summary"] == {"verified": 2, "locked": 0, "refused": 0}
    results = ms.get_results(run_id)
    assert results["protocol_sha256"] == submitted["draft_sha256"] and results["confirmed_via"] == "cli"
    distance = next(r for r in results["results"] if r["metric_id"] == "distance")
    assert distance["status"] == "verified" and distance["value"] == pytest.approx(60, rel=0.02)

    manifest = ms.get_manifest(run_id)
    assert manifest["protocol"]["confirmed_by"] == "Dr. Example (via cli)" and manifest["withheld"]
    assert "confirmed by Dr. Example via the kappa command line" in manifest["methods"]
    assert Path(manifest["path"]).exists()
    assert json.loads(ms.run_manifest(run_id))["manifest_sha256"] == manifest["manifest_sha256"]
    rows = ms.list_runs()["runs"]
    assert [(r["run_id"], r["stage"], r["confirmed_by"], r["confirmed_via"]) for r in rows] == [
        (run_id, "tracked", "Dr. Example", "cli")
    ]

    # A later submission waits for a person too: the run keeps the protocol they confirmed.
    again = ms.submit_for_confirmation(run_id, {**edited, "title": "Changed again"})
    assert again["status"] == "awaiting_person"
    assert again["confirmed_protocol_sha256"] == submitted["draft_sha256"] != again["draft_sha256"]
    state = runner.load(run_id)
    assert (state.protocol["title"], state.confirmed_by, state.confirmed_via) == ("Open field, edited", "Dr. Example", "cli")
    assert ms.get_results(run_id)["protocol_sha256"] == submitted["draft_sha256"]


def test_get_results_hides_provisional_values(runner, video, capsys):
    raw, fix, retrack, (a, b) = two_mice()
    runner._api = FakeApi(runner.store, main=raw, retrack=retrack)
    run_id = ms.create_run(video)["run_id"]
    runner.confirm(run_id, protocol(count=2), by="Dr. Example", via="web")  # a person in the web workspace
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

    # The agent proposes marks; they are checked, but nothing changes until a person applies them.
    with pytest.raises(ValidationError):
        AnchorIn(subject=1)
    for frame, anchors, message in (
        (fix, [], "mark at least one subject"),
        (150, [AnchorIn(subject=1, point=(10, 10))], "outside the video"),
        (fix, [AnchorIn(subject=1, point=(900, 10))], "outside the 800x800 frame"),
        (fix, [AnchorIn(subject=1, point=(1, 2)), AnchorIn(subject=1, point=(3, 4))], "each subject can be marked once"),
    ):
        with pytest.raises(ToolError, match=message):
            ms.propose_repair(run_id, frame, anchors)
    marks = [
        AnchorIn(subject=1, point=(round(a[fix, 0]), round(a[fix, 1]))),
        AnchorIn(subject=2, box=(round(b[fix, 0]) - 20, round(b[fix, 1]) - 10, round(b[fix, 0]) + 20, round(b[fix, 1]) + 10)),
    ]
    proposed = ms.propose_repair(run_id, fix, marks)
    assert proposed["status"] == "awaiting_person"
    assert proposed["command"].startswith(f"kappa repair {run_id} --from-proposal {fix} --sha256 ")
    assert Path(proposed["proposal_file"]) == runner.path(run_id) / f"repair.proposal-{fix}.json"
    assert runner.load(run_id).repairs == [] and ms.get_results(run_id)["summary"]["verified"] == 0

    assert person("repair", run_id, "--from-proposal", fix, "--by", "A. Person") == 0
    out = capsys.readouterr().out
    assert f"Applying the marks proposed for frame {fix}: subject 1 at point" in out and "subject 2 at box" in out
    view = ms.get_results(run_id)
    assert view["open_blockers"] == []
    assert all(r["status"] == "verified" and r["value"] is not None for r in view["results"])

    manifest = ms.get_manifest(run_id)
    assert manifest["protocol"]["confirmed_by"] == "Dr. Example (via web)"
    assert [(r["frame"], r["by"], r["via"]) for r in manifest["repairs"]] == [(fix, "A. Person", "cli")]
    assert f"(frame {fix}, by A. Person via the kappa command line)" in manifest["methods"]


def test_score_behaviour(runner, video):
    run_id = ms.create_run(video)["run_id"]
    runner.confirm(run_id, protocol(rearing=True), by="Dr. Example", via="cli")
    asyncio.run(ms.track(run_id))
    scored = ms.score_behaviour(run_id, "rearing")
    assert (scored["bouts_found"], scored["withheld_subjects"], scored["reviews_failed"]) == (0, [], 0)
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
        ms.submit_for_confirmation(run_id, "{nope")
    with pytest.raises(ToolError, match="the protocol is not valid: metrics"):
        ms.submit_for_confirmation(run_id, {**protocol(), "metrics": []})
    assert runner.load(run_id).draft is None
    no_corners = ms.submit_for_confirmation(run_id, library.load("open_field"))
    assert no_corners["status"] == "awaiting_person" and no_corners["missing"] == ["arena.corners_px"]
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
    runner.confirm(run_id, protocol(), by="Dr. Example", via="cli")
    progress: list[float] = []

    async def on_progress(value, total, message):
        progress.append(value)

    async def session():
        async with client_module.Client(ms.mcp) as client:
            tools = (await client.list_tools()).tools
            templates = [t.uri_template for t in (await client.list_resource_templates()).resource_templates]
            library_text = (await client.read_resource("kappa://protocols/library")).contents[0].text
            missing = await client.call_tool("get_results", {"run_id": "nope"})
            tracked = await client.call_tool("track", {"run_id": run_id}, progress_callback=on_progress)
            return tools, templates, library_text, missing, tracked

    tools, templates, library_text, missing, tracked = asyncio.run(session())
    assert {t.name for t in tools} == TOOLS
    schemas = {t.name: json.dumps(t.input_schema) for t in tools}
    assert "protocol_json" in schemas["submit_for_confirmation"] and "anchors" in schemas["propose_repair"]
    assert not any("confirmed_by" in s for s in schemas.values()), "no tool takes an approver's name"
    assert templates == ["kappa://runs/{run_id}/manifest"]
    assert len(json.loads(library_text)["protocols"]) == 3
    assert missing.is_error and "no run nope" in missing.content[0].text
    assert not tracked.is_error
    assert progress and progress[-1] == 150
