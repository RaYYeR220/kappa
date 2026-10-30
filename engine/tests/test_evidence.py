import json

from kappa.evidence import Manifest, methods_paragraph, summary
from kappa.model import EventKind, Evidence, IntegrityEvent, MetricResult, Severity, Span, Status
from kappa.protocol import AssayProtocol

PROTOCOL = AssayProtocol.model_validate(
    {
        "title": "Open field",
        "question": "distance and centre time",
        "subject": {"concept": "mouse", "expected_count": 1, "label": "mouse"},
        "arena": {"shape": "rect", "width_cm": 45, "height_cm": 45},
        "zones": [{"id": "center", "kind": "center_fraction", "fraction": 0.25}],
        "entry_rule": {"mode": "body_fraction"},
        "metrics": [{"id": "distance", "kind": "distance"}, {"id": "ct", "kind": "time_in_zone", "zone": "center"}],
    }
)
EVENT = IntegrityEvent("teleport:1:100", EventKind.TELEPORT, Severity.BLOCKING, (1,), Span(100, 101), "jump", 101)
RESULTS = [
    MetricResult("distance", 1, None, "cm", Status.VERIFIED, value=3120.5, evidence=(Evidence(Span(0, 15000), 1, "whole test"),)),
    MetricResult("ct", 1, None, "s", Status.LOCKED, provisional=41.0, reasons=("teleport:1:100",), params={"zone": "center"}),
]


def _manifest():
    return Manifest(
        run_id="run-1",
        video_sha256="f39e12bba185ab2935e7443f7fd4006dce221ea074ace57856189986eebd9852",
        video={"width": 928, "height": 576, "fps": 25.0, "duration_s": 600.0, "n_frames": 15000},
        protocol=PROTOCOL,
        confirmed_by="researcher",
        models={"segmenter": "sam-3.1"},
        artifacts={"sam_raw": "ab" * 32},
        events=[EVENT],
        results=RESULTS,
    )


def test_manifest_is_json_and_hashes_the_protocol():
    data = json.loads(_manifest().to_json())
    assert data["protocol"]["sha256"] == PROTOCOL.content_hash()
    assert data["integrity_events"][0]["kind"] == "teleport"
    assert data["results"][1]["status"] == "locked"
    assert data["results"][1]["value"] is None and data["results"][1]["provisional"] == 41.0
    assert data["summary"] == {"verified": 1, "locked": 1, "refused": 0}


def test_methods_paragraph_states_only_run_facts():
    text = methods_paragraph(_manifest())
    assert "45 x 45 cm arena" in text
    assert "centre zone covering 25% of the floor area" in text
    assert "at least 70% of its body area" in text
    assert "1 were withheld" in text
    assert PROTOCOL.content_hash()[:12] in text
    assert "sam-3.1" in text


def test_the_manifest_says_who_approved_and_through_which_channel():
    assert _manifest().to_dict()["protocol"]["confirmed_by"] == "researcher"
    m = _manifest()
    m.confirmed_via = "web"
    m.repairs = [{"frame": 120, "by": "A. Person", "via": "cli", "anchors": []}]
    data = m.to_dict()
    assert (data["protocol"]["confirmed_by"], data["protocol"]["confirmed_via"]) == ("researcher (via web)", "web")
    assert data["repairs"] == m.repairs
    assert "confirmed by researcher via the Kappa web workspace" in data["methods"]
    assert "(frame 120, by A. Person via the kappa command line)" in data["methods"]


def test_summary_counts_every_status():
    assert summary([]) == {"verified": 0, "locked": 0, "refused": 0}
