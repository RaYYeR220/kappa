"""The kappa command against a fake Model API: plain messages, right exit codes, no leaked numbers."""

import json
import shutil

import pytest
from run_fakes import FakeApi, make_video, one_mouse, protocol, stub_rearing_model, two_mice, with_pin, write_json

from kappa import cli, public
from kappa.model import Status
from kappa.repair import Anchor
from kappa.runner import RunError, Runner

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("video") / "trial.mp4")


@pytest.fixture
def kappa(tmp_path, monkeypatch, capsys):
    """Run the CLI on a runner whose Model API is fake; returns (code, stdout, stderr)."""
    runner = Runner(home=tmp_path / "home")
    runner._api = FakeApi(runner.store, main=one_mouse())
    monkeypatch.setattr(cli, "runner_factory", lambda: runner)

    def run(*argv):
        code = cli.main(with_pin(list(argv), runner))
        out, err = capsys.readouterr()
        return code, out, err

    run.runner = runner
    return run


def test_lifecycle(kappa, video, tmp_path):
    code, out, err = kappa("new", video, "--id", "demo")
    assert (code, out) == (0, "demo\n")
    assert "800x800" in err

    code, out, _ = kappa("draft", "demo", "How far does the mouse walk and how long is it in the centre?")
    assert code == 0
    assert "title: Open field" in out and "  - id: center" in out
    assert "Arena is 80 x 80 cm." in out and "Refused" in out
    draft_file = kappa.runner.path("demo") / "protocol.draft.json"
    assert json.loads(draft_file.read_text())["arena"]["corners_px"][2] == [799, 799]

    code, _, err = kappa("track", "demo")
    assert code == 2 and "not been confirmed" in err and "Traceback" not in err
    code, _, err = kappa("confirm", "demo", "--by", "  ")
    assert code == 2 and "name of the person" in err

    code, out, _ = kappa("confirm", "demo", "--by", "Dr. Example")
    assert code == 0 and "Confirmed by Dr. Example." in out

    code, out, err = kappa("track", "demo")
    assert code == 0 and "tracking: " in err and "lines, done" in err
    assert "verified" in out

    code, out, _ = kappa("results", "demo")
    assert code == 0
    header = out.splitlines()[2]
    assert header.split() == ["metric", "subject", "bin", "status", "value", "unit", "reason"]
    distance = next(ln for ln in out.splitlines() if ln.startswith("distance"))
    assert "verified" in distance and " cm" in distance

    code, out, _ = kappa("results", "demo", "--json")
    view = json.loads(out)
    row = next(r for r in view["results"] if r["metric_id"] == "distance")
    assert row["status"] == "verified" and row["value"] == pytest.approx(60, rel=0.02)

    out_file = tmp_path / "m.json"
    code, out, _ = kappa("manifest", "demo", "--out", out_file)
    assert code == 0 and "Methods:" in out and "confirmed by Dr. Example (via cli)" in out
    written = json.loads(out_file.read_text())
    assert written["run_id"] == "demo" and written["protocol"]["confirmed_via"] == "cli"

    code, out, _ = kappa("runs")
    assert code == 0 and "demo" in out and "tracked" in out and "Dr. Example" in out


def test_confirm_takes_the_edited_draft(kappa, video):
    kappa("new", video, "--id", "demo")
    kappa("draft", "demo", "distance")
    draft_file = kappa.runner.path("demo") / "protocol.draft.json"
    body = json.loads(draft_file.read_text())
    body["arena"]["width_cm"] = body["arena"]["height_cm"] = 90
    draft_file.write_text(json.dumps(body))
    code, out, _ = kappa("confirm", "demo", "--by", "Dr. Example")
    assert code == 0 and "edited from the draft" in out
    assert kappa.runner.load("demo").protocol["arena"]["width_cm"] == 90


def test_locked_values_never_print(kappa, video, tmp_path):
    raw, fix, retrack, (a, b) = two_mice()
    kappa.runner._api = FakeApi(kappa.runner.store, main=raw, retrack=retrack)
    kappa("new", video, "--id", "pair")
    code, _, _ = kappa("confirm", "pair", "--by", "Dr. Example", "--protocol", write_json(tmp_path / "p.json", protocol(count=2)))
    assert code == 0
    kappa("track", "pair")

    provisional = [
        r.provisional for r in kappa.runner.analysis(kappa.runner.load("pair")).results if r.status is Status.LOCKED
    ]
    assert provisional and all(p is not None for p in provisional)

    code, out, _ = kappa("results", "pair")
    assert code == 0
    locked = [ln for ln in out.splitlines() if " locked " in ln]
    assert len(locked) == len(provisional)
    for ln in locked:
        assert ln.split()[4] == "-" and f"fix at frame {fix}" in ln
    for p in provisional:
        assert f"{p:.2f}" not in out
    assert f"--frame {fix}" in out and "settle at frame" in out

    code, out, _ = kappa("results", "pair", "--json")
    assert "provisional" not in out
    rows = json.loads(out)["results"]
    assert all(r["value"] is None and r["fix_frame"] == fix and r["blocked_by"] for r in rows)

    code, out, _ = kappa(
        "repair", "pair", "--frame", fix, "--by", "A. Person",
        "--subject", f"1:{a[fix, 0]:.0f},{a[fix, 1]:.0f}", "--box", f"2:{b[fix, 0] - 20:.0f},{b[fix, 1] - 10:.0f},{b[fix, 0] + 20:.0f},{b[fix, 1] + 10:.0f}",
    )
    assert code == 0 and "Repair at frame" in out
    view = json.loads(kappa("results", "pair", "--json")[1])
    assert all(r["status"] == "verified" for r in view["results"] if r["metric_id"] == "distance")


def test_repair_from_a_proposal(kappa, video, tmp_path):
    raw, fix, retrack, (a, b) = two_mice()
    kappa.runner._api = FakeApi(kappa.runner.store, main=raw, retrack=retrack)
    kappa("new", video, "--id", "pair")
    kappa("confirm", "pair", "--by", "Dr. Example", "--protocol", write_json(tmp_path / "p.json", protocol(count=2)))
    kappa("track", "pair")
    code, _, err = kappa("repair", "pair", "--from-proposal", fix, "--by", "A. Person")
    assert code == 2 and f"no repair proposal for frame {fix}" in err

    marks = [Anchor(1, fix, point=(round(a[fix, 0]), round(a[fix, 1]))), Anchor(2, fix, point=(round(b[fix, 0]), round(b[fix, 1])))]
    kappa.runner.propose_repair("pair", marks, source="mcp")
    code, _, err = kappa("repair", "pair", "--from-proposal", fix, "--by", "A. Person", "--subject", "1:5,5")
    assert code == 2 and "leave out --subject and --box" in err
    code, _, err = kappa("repair", "pair", "--frame", fix, "--from-proposal", fix, "--by", "A. Person")
    assert code == 1 and "not allowed with" in err
    assert kappa.runner.load("pair").repairs == []

    code, out, _ = kappa("repair", "pair", "--from-proposal", fix, "--by", "A. Person")
    assert code == 0 and f"Repair at frame {fix} by A. Person recorded for subjects 1, 2." in out
    assert [(r.frame, r.by, r.via) for r in kappa.runner.load("pair").repairs] == [(fix, "A. Person", "cli")]
    view = json.loads(kappa("results", "pair", "--json")[1])
    assert view["open_blockers"] == [] and all(r["status"] == "verified" for r in view["results"])


def test_repair_arguments():
    anchors = cli.anchors_from(5, [(1, (10, 20))], [(1, (0, 0, 30, 40)), (2, (50, 50, 60, 60))])
    assert [(x.subject, x.point, x.box) for x in anchors] == [(1, (10, 20), (0, 0, 30, 40)), (2, None, (50, 50, 60, 60))]
    with pytest.raises(RunError):
        cli.anchors_from(5, [(1, (10, 20)), (1, (11, 21))], [])
    with pytest.raises(RunError):
        cli.anchors_from(5, [], [])


def test_errors_are_plain(kappa, video, monkeypatch):
    code, _, err = kappa("results", "nope")
    assert (code, err) == (2, "kappa: no run nope\n")
    code, _, err = kappa("new", "missing.mp4")
    assert code == 2 and "no video at missing.mp4" in err
    code, _, err = kappa("results", "../escape")
    assert code == 2 and "not a run id" in err
    code, _, err = kappa("repair", "demo", "--frame", "3", "--by", "x", "--subject", "1-2")
    assert code == 1 and "SUBJECT:X,Y" in err
    code, _, _ = kappa("new", video, "--id", "demo")
    code, _, err = kappa("new", video, "--id", "demo")
    assert code == 2 and "already exists" in err
    bad = kappa.runner.home / "bad.json"
    bad.write_text("{nope")
    code, _, err = kappa("confirm", "demo", "--by", "Dr. Example", "--protocol", bad)
    assert code == 1 and "bad.json is not valid JSON" in err
    code, _, err = kappa("confirm", "demo", "--by", "Dr. Example")
    assert code == 2 and "has no draft yet" in err

    kappa.runner._api = None
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    code, _, err = kappa("draft", "demo", "distance")
    assert code == 1 and err == "kappa: MODEL_API_KEY is not set\n"

    def boom(args, runner):
        raise ZeroDivisionError("bad maths")

    monkeypatch.setattr(cli, "cmd_runs", boom)
    code, _, err = kappa("runs")
    assert code == 1 and "unexpected error (ZeroDivisionError)" in err and "Traceback" not in err


def test_batch_shares_one_protocol(kappa, video, tmp_path):
    second = tmp_path / "second.mp4"
    shutil.copy(video, second)
    body = protocol()
    path = write_json(tmp_path / "cohort.json", body)
    code, out, err = kappa("batch", path, video, second, tmp_path / "missing.mp4", "--by", "Dr. Example")
    assert code == 2
    rows = out.splitlines()
    assert rows[0].split()[:5] == ["video", "run", "verified", "locked", "refused"]
    assert sum("trial.mp4" in r or "second.mp4" in r for r in rows[1:3]) == 2
    assert "no video at" in rows[3]
    from kappa.protocol import AssayProtocol

    digest = AssayProtocol.model_validate(body).content_hash()
    assert f"2 of 3 videos measured under protocol sha256 {digest}" in out
    runs = [s for s in kappa.runner.list() if s.manifest_sha256]
    assert len(runs) == 2
    assert {kappa.runner.store.get_json(s.manifest_sha256)["protocol"]["sha256"] for s in runs} == {digest}

    no_corners = {**body, "arena": {**body["arena"], "corners_px": None}}
    code, _, err = kappa("batch", write_json(tmp_path / "bad.json", no_corners), video, "--by", "Dr. Example")
    assert code == 2 and "no arena corners" in err


def _rearing_run(kappa, video, tmp_path, rearing_in_video):
    kappa.runner._api = FakeApi(kappa.runner.store, main=one_mouse(rearing=rearing_in_video))
    kappa("new", video, "--id", "rear")
    kappa("confirm", "rear", "--by", "Dr. Example", "--protocol", write_json(tmp_path / "p.json", protocol(rearing=True)))


def test_score_reports_through_the_gate(kappa, video, tmp_path, monkeypatch):
    stub_rearing_model(monkeypatch)
    _rearing_run(kappa, video, tmp_path, rearing_in_video=False)
    code, _, err = kappa("score", "rear", "rearing")
    assert code == 2 and "not been tracked" in err
    kappa("track", "rear")
    code, out, err = kappa("results", "rear")
    rearing = next(ln for ln in out.splitlines() if ln.startswith("rearing_count"))
    assert "refused" in rearing and "has not been scored" in rearing

    code, out, err = kappa("score", "rear", "rearing")
    assert code == 0, err
    assert "Scored rearing" in out
    rearing = next(ln for ln in out.splitlines() if ln.startswith("rearing_count"))
    assert rearing.split()[3:5] == ["verified", "0"]

    code, _, err = kappa("score", "rear", "grooming")
    assert code == 2 and "no behaviour grooming" in err


def test_score_a_real_candidate(kappa, video, tmp_path, monkeypatch):
    stub_rearing_model(monkeypatch)
    _rearing_run(kappa, video, tmp_path, rearing_in_video=True)
    kappa("track", "rear")
    code, out, err = kappa("score", "rear", "rearing")
    assert code == 0, err
    assert "Scored rearing" in out
    rearing = next(ln for ln in out.splitlines() if ln.startswith("rearing_count"))
    assert rearing.split()[3:5] == ["verified", "1"]


def test_confirm_is_pinned_to_the_reviewed_protocol(kappa, video):
    kappa("new", video, "--id", "pin")
    kappa("draft", "pin", "How far does it walk?")
    code, _, err = kappa("confirm", "pin", "--by", "Dr. Example", "--unpinned")
    assert code == 2 and "--sha256" in err
    reviewed = public.protocol_hash(public.read_draft(kappa.runner, kappa.runner.load("pin")))
    draft = kappa.runner.draft_path("pin")
    body = json.loads(draft.read_text())
    body["zones"][0]["fraction"] = 0.5  # the file changes after the person reviewed it
    draft.write_text(json.dumps(body))
    code, _, err = kappa("confirm", "pin", "--by", "Dr. Example", "--sha256", reviewed[:16])
    assert code == 2 and "not the one you named" in err
    assert kappa.runner.load("pin").confirmed_by is None
