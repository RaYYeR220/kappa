"""HTTP surface: public mode is read-only, locked numbers never leave the server."""

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from test_runner import FakeApi, runner  # noqa: E402,F401

from kappa.api import create_app  # noqa: E402


@pytest.fixture
def tracked(runner):  # noqa: F811
    r, src = runner
    state = r.create(src, run_id="demo")
    state = r.draft("demo", "distance and centre time")
    r.confirm("demo", state.draft, by="Dr. Example")
    r.track("demo")
    return r


def test_public_mode_refuses_every_write(tracked):
    client = TestClient(create_app(tracked, public=True))
    assert client.get("/health").json() == {"ok": True, "public": True}
    assert client.post("/runs/demo/track").status_code == 403
    assert client.post("/runs/demo/draft", json={"question": "q"}).status_code == 403
    run = client.get("/runs/demo").json()
    assert run["tracked"] is True and (run["confirmed_by"], run["confirmed_via"]) == ("Dr. Example", "api")


def test_results_and_tracks(tracked):
    client = TestClient(create_app(tracked, public=True))
    body = client.get("/runs/demo/results").json()
    distance = next(r for r in body["results"] if r["metric_id"] == "distance")
    assert distance["status"] == "verified" and distance["value"] == pytest.approx(60, rel=0.02)
    assert "provisional" not in json.dumps(body)
    t = client.get("/runs/demo/tracks").json()
    assert t["n_frames"] == 150 and len(t["subjects"]["1"]["cx"]) == 150
    assert client.get("/runs/demo/sam").text.startswith("<0f>")
    assert client.get("/runs/nope").status_code == 404


def test_writes_fail_closed_without_a_configured_token(tracked, monkeypatch):
    monkeypatch.delenv("KAPPA_OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("KAPPA_INSECURE_DEV", raising=False)
    client = TestClient(create_app(tracked, public=False))
    assert client.post("/runs/demo/track").status_code == 503


def test_operator_token_guards_writes(tracked, monkeypatch):
    monkeypatch.setenv("KAPPA_OPERATOR_TOKEN", "secret")
    client = TestClient(create_app(tracked, public=False))
    assert client.post("/runs/demo/track").status_code == 401
    assert client.post("/runs/demo/track", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_the_web_workspace_confirms_and_repairs_as_web(tracked, monkeypatch):
    monkeypatch.setenv("KAPPA_OPERATOR_TOKEN", "secret")
    auth = {"Authorization": "Bearer secret"}
    client = TestClient(create_app(tracked, public=False))
    protocol = tracked.load("demo").protocol
    body = client.post("/runs/demo/confirm", json={"protocol": protocol, "by": "Dr. Web"}, headers=auth).json()
    assert (body["confirmed_by"], body["confirmed_via"]) == ("Dr. Web", "web")

    repair = {"frame": 10, "anchors": [{"subject": 1, "x": 140, "y": 400}], "by": "Dr. Web"}
    started = client.post("/runs/demo/repair", json=repair, headers=auth).json()
    with client.stream("GET", started["stream"], headers=auth) as stream:
        events = [json.loads(line[6:]) for line in stream.iter_lines() if line.startswith("data: ")]
    assert events[-1]["type"] == "done", events
    assert [(r.frame, r.by, r.via) for r in tracked.load("demo").repairs] == [(10, "Dr. Web", "web")]
    manifest = tracked.manifest("demo")
    assert manifest["protocol"]["confirmed_by"] == "Dr. Web (via web)"
    assert manifest["repairs"][0]["via"] == "web" and "via the Kappa web workspace" in manifest["methods"]


def test_track_job_streams_segmenter_output(tracked, monkeypatch):
    monkeypatch.setenv("KAPPA_OPERATOR_TOKEN", "secret")
    client = TestClient(create_app(tracked, public=False))
    started = client.post("/runs/demo/track", headers={"Authorization": "Bearer secret"}).json()
    assert client.get(started["stream"]).status_code == 401
    with client.stream("GET", started["stream"], headers={"Authorization": "Bearer secret"}) as stream:
        events = [json.loads(line[6:]) for line in stream.iter_lines() if line.startswith("data: ")]
    assert events[0]["type"] == "sam" and events[-1]["type"] == "done"
