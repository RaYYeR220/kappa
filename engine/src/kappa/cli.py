"""The ``kappa`` command: a measurement run, one step per subcommand.

    kappa new trial.mp4                    prints the run id
    kappa draft RUN "How far does it walk?"
    kappa confirm RUN --sha256 H --by "Dr. Name"   only a confirmed protocol can run
    kappa track RUN
    kappa results RUN
    kappa manifest RUN
    kappa repair RUN --from-proposal N --sha256 H --by "Dr. Name"   apply marks an agent proposed

Confirming and repairing are a person's acts, so they happen here (or in the web
workspace), never over MCP: an agent can only propose a protocol or a repair, and
the manifest records that the person approved it "via cli".

Runs live under ``$KAPPA_HOME`` (default ``./kappa-data``). Model calls need
``MODEL_API_KEY``. Exit status: 0 ok, 2 when Kappa refuses a step, 1 for anything else.
Set ``KAPPA_DEBUG=1`` to see the traceback of an unexpected error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Self

from kappa import public
from kappa.evidence import confirmed_label
from kappa.protocol import AssayProtocol
from kappa.repair import Anchor
from kappa.runner import RunError, Runner

# Tests swap this for a factory whose runner has a fake Model API.
runner_factory: Callable[[], Runner] = Runner


def _err(message: str) -> None:
    print(f"kappa: {message}", file=sys.stderr)


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    cells = [[str(c) for c in header]] + [["" if c is None else str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    return ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip() for row in cells]


class _Progress:
    """A counter rewritten in place on one stderr line; the line is ended even on failure.

    On a terminal the count updates live; a log file gets only the final count.
    """

    def __init__(self, label: str, unit: str) -> None:
        self.label, self.unit, self.count, self.shown = label, unit, 0, -1
        self.live = sys.stderr.isatty()
        self._lock = threading.Lock()  # verdicts arrive from worker threads

    def add(self, n: int = 1) -> None:
        with self._lock:
            self.count += n
            if self.live and self.count != self.shown:
                self.shown = self.count
                sys.stderr.write(f"\r{self.label}: {self.count} {self.unit}")
                sys.stderr.flush()

    def lines(self, delta: str) -> None:
        """``on_delta`` for streamed segmenter output: one line per frame."""
        self.add(delta.count("\n"))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, failed: object, *_: object) -> None:
        if failed is None:
            sys.stderr.write(f"\r{self.label}: {self.count} {self.unit}, done\n")
        elif self.shown >= 0:
            sys.stderr.write("\n")
        sys.stderr.flush()


# -- protocol as readable text ----------------------------------------------------


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_scalar(v) for v in value) + "]"
    return str(value)


def _inline(value: Any) -> bool:
    if isinstance(value, dict):
        return False
    if isinstance(value, (list, tuple)):
        return all(_inline(v) for v in value)
    return True


def yamlish(value: dict[str, Any], indent: int = 0) -> list[str]:
    """A dict as YAML-like lines, for reading (not parsing). Empty values are left out."""
    pad = "  " * indent
    lines: list[str] = []
    for key, v in value.items():
        if v is None or v == "" or v == [] or v == {}:
            continue
        if _inline(v):
            lines.append(f"{pad}{key}: {_scalar(v)}")
        elif isinstance(v, dict):
            lines += [f"{pad}{key}:", *yamlish(v, indent + 1)]
        else:
            lines.append(f"{pad}{key}:")
            for item in v:
                if isinstance(item, dict):
                    sub = yamlish(item, indent + 2)
                    lines += [f"{pad}  - {sub[0].lstrip()}", *sub[1:]]
                else:
                    lines.append(f"{pad}  - {_scalar(item)}")
    return lines


# -- results as a table -----------------------------------------------------------


def _number(value: float, unit: str) -> str:
    return f"{value:.0f}" if unit == "count" else f"{value:.2f}"


def _value_cell(row: dict[str, Any]) -> str:
    if row["value"] is None:  # locked or refused: there is no number to show
        return "-"
    text = _number(row["value"], row["unit"])
    spread = sum(abs(v) for v in row["uncertainty"].values() if math.isfinite(v))
    return f"{text} ± {_number(spread, row['unit'])}" if spread > 0 else text


def _reason_cell(row: dict[str, Any], blockers: dict[str, dict[str, Any]]) -> str:
    if row["status"] == "locked":
        parts = []
        for event_id in row["blocked_by"]:
            b = blockers.get(event_id)
            parts.append(f"{b['kind']} frames {b['frames'][0]}-{b['frames'][1]}" if b else event_id)
        text = "; ".join(parts) or "; ".join(row["reasons"])
        return text + (f"; fix at frame {row['fix_frame']}" if row["fix_frame"] is not None else "")
    return "; ".join(row["reasons"])


def results_lines(view: dict[str, Any], metric_ids: set[str] | None = None) -> list[str]:
    blockers = {b["id"]: b for b in view["open_blockers"]}
    rows = []
    for r in view["results"]:
        if metric_ids is not None and r["metric_id"] not in metric_ids:
            continue
        span = r["bin_s"]
        rows.append([
            r["metric_id"],
            "group" if r["subject"] is None else r["subject"],
            "all" if span is None else f"{span[0]:g}-{span[1]:g}s",
            r["status"],
            _value_cell(r),
            r["unit"],
            _reason_cell(r, blockers),
        ])
    return _table(["metric", "subject", "bin", "status", "value", "unit", "reason"], rows)


def _summary_line(view: dict[str, Any]) -> str:
    s = view["summary"]
    return f"{s['verified']} verified, {s['locked']} locked, {s['refused']} refused"


def _blocker_lines(view: dict[str, Any]) -> list[str]:
    if not view["open_blockers"]:
        return []
    lines = ["", "Open integrity events (each locks the numbers it touches):"]
    for b in view["open_blockers"]:
        who = ",".join(str(s) for s in b["subjects"]) or "scene"
        fix = f", settle at frame {b['fix_frame']}" if b["fix_frame"] is not None else ""
        lines.append(f"  {b['kind']} frames {b['frames'][0]}-{b['frames'][1]}, subjects {who}{fix}")
        lines.append(f"    {b['detail']}")
    fixes = sorted(b["fix_frame"] for b in view["open_blockers"] if b["fix_frame"] is not None)
    if fixes:
        lines.append(
            f"Look at frame {fixes[0]}, mark each animal under its number, then:\n"
            f'  kappa repair {view["run_id"]} --frame {fixes[0]} --by "<name>" --subject 1:x,y --subject 2:x,y'
        )
    return lines


# -- commands ---------------------------------------------------------------------


def _read_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from None


def cmd_new(args: argparse.Namespace, runner: Runner) -> int:
    if args.id is not None:
        public.check_run_id(args.id)
        if (runner.path(args.id) / "run.json").exists():
            raise RunError(f"run {args.id} already exists")
    source = Path(args.video)
    if not source.is_file():
        raise RunError(f"no video at {source}")
    state = runner.create(source, start_s=args.start, end_s=args.end, fps=args.fps, run_id=args.id)
    v = state.video
    print(state.run_id)
    _err(f"{state.source_name}: {v['width']}x{v['height']}, {v['fps']:g} fps, {v['n_frames']} frames")
    return 0


def cmd_draft(args: argparse.Namespace, runner: Runner) -> int:
    state = runner.draft(args.run, args.question)
    path = runner.draft_path(state.run_id)
    draft = state.draft or {}
    body = {k: v for k, v in draft.items() if k not in {"refused", "schema_version"}}
    lines = [f"Draft protocol for run {state.run_id} (sha256 {public.protocol_hash(draft)[:12]}, not confirmed)", ""]
    lines += yamlish(body)
    lines += ["", "Assumptions (confirm or correct each):"]
    lines += [f"  - {a}" for a in state.assumptions] or ["  (none)"]
    lines += ["", "Refused (asked for, but Kappa won't measure it):"]
    lines += [f"  - {r['request']}: {r['reason']}" for r in draft.get("refused", [])] or ["  (none)"]
    lines += ["", f"Wrote {path}"]
    if draft.get("arena", {}).get("corners_px") is None:
        lines.append("The arena corners are missing: set arena.corners_px in that file before confirming.")
    lines.append(
        f'Edit it if anything is wrong, then: kappa confirm {state.run_id} --sha256 {public.protocol_hash(draft)[:16]} --by "<your name>"'
        " (after editing, kappa confirm prints the new sha256)"
    )
    print("\n".join(lines))
    return 0


def cmd_confirm(args: argparse.Namespace, runner: Runner) -> int:
    state = runner.load(args.run)
    if args.protocol is not None:
        protocol = _read_json(args.protocol)
    else:
        protocol = public.read_draft(runner, state)
    public.check_pinned(args.sha256, public.protocol_hash(protocol) or "", "protocol")
    state = runner.confirm(args.run, protocol, by=args.by, via="cli")
    edited = state.draft is not None and state.protocol != AssayProtocol.model_validate(state.draft).model_dump(mode="json")
    note = " (edited from the draft)" if edited else ""
    print(f"Confirmed by {state.confirmed_by}{note}. Protocol sha256 {public.protocol_hash(state.protocol)}")
    print(f"Next: kappa track {state.run_id}")
    return 0


def cmd_track(args: argparse.Namespace, runner: Runner) -> int:
    with _Progress("tracking", "lines") as progress:
        state = runner.track(args.run, on_delta=progress.lines)
    view = public.results_view(runner, state)
    print(f"Tracked run {state.run_id}: {_summary_line(view)}. See: kappa results {state.run_id}")
    return 0


def cmd_score(args: argparse.Namespace, runner: Runner) -> int:
    # Progress only. Verdicts are not shown: what they add up to is reported through the gate.
    with _Progress("reviewing", "candidates") as progress:
        state = runner.score(args.run, args.behaviour, on_verdict=lambda _verdict: progress.add())
    record = state.behaviours[args.behaviour]
    total = len(runner.store.get_json(record.verdicts_sha256))
    failed = len(record.failed_spans)
    print(f"Scored {args.behaviour}: {total} candidates reviewed, {failed} reviews failed.")
    if failed:
        print("A failed review locks the windows it touches; score again to retry.")
    protocol = state.confirmed
    ids = {m.id for m in protocol.metrics if m.behaviour == args.behaviour}
    if ids:
        print()
        print("\n".join(results_lines(public.results_view(runner, state), ids)))
    return 0


def _subject_point(text: str) -> tuple[int, tuple[int, int]]:
    try:
        subject, _, rest = text.partition(":")
        x, y = (round(float(v)) for v in rest.split(","))
        return int(subject), (x, y)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected SUBJECT:X,Y such as 1:320,240, got {text!r}") from None


def _subject_box(text: str) -> tuple[int, tuple[int, int, int, int]]:
    try:
        subject, _, rest = text.partition(":")
        x1, y1, x2, y2 = (round(float(v)) for v in rest.split(","))
        return int(subject), (x1, y1, x2, y2)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected SUBJECT:X1,Y1,X2,Y2 such as 1:300,220,340,260, got {text!r}") from None


def anchors_from(frame: int, points: Sequence[tuple[int, Any]], boxes: Sequence[tuple[int, Any]]) -> list[Anchor]:
    for name, marks in (("--subject", points), ("--box", boxes)):
        subjects = [s for s, _ in marks]
        if len(subjects) != len(set(subjects)):
            raise RunError(f"{name} names the same subject twice")
    by_point, by_box = dict(points), dict(boxes)
    subjects = sorted(set(by_point) | set(by_box))
    if not subjects:
        raise RunError("mark at least one subject with --subject or --box")
    return [Anchor(s, frame, box=by_box.get(s), point=by_point.get(s)) for s in subjects]


def _mark(anchor: Anchor) -> str:
    parts = [f"point {anchor.point[0]},{anchor.point[1]}"] if anchor.point else []
    parts += [f"box {','.join(str(v) for v in anchor.box)}"] if anchor.box else []
    return f"subject {anchor.subject} at {' and '.join(parts)}"


def cmd_repair(args: argparse.Namespace, runner: Runner) -> int:
    if args.from_proposal is not None:
        if args.subject or args.box:
            raise RunError("--from-proposal takes its marks from the proposal; leave out --subject and --box")
        anchors = runner.repair_proposal(args.run, args.from_proposal)
        public.check_pinned(args.sha256, public.marks_hash(anchors), "proposal")
        print(f"Applying the marks proposed for frame {args.from_proposal}: {'; '.join(_mark(a) for a in anchors)}")
    else:
        anchors = anchors_from(args.frame, args.subject or [], args.box or [])
    frame = args.frame if args.from_proposal is None else args.from_proposal
    state = runner.repair(args.run, anchors, by=args.by, via="cli")
    view = public.results_view(runner, state)
    print(f"Repair at frame {frame} by {args.by.strip()} recorded for subjects "
          f"{', '.join(str(a.subject) for a in anchors)}.")
    print(f"Now {_summary_line(view)}; {len(view['open_blockers'])} integrity events still open.")
    print(f"See: kappa results {state.run_id}")
    return 0


def cmd_results(args: argparse.Namespace, runner: Runner) -> int:
    state = runner.load(args.run)
    view = public.results_view(runner, state)
    if args.json:
        print(json.dumps(view, indent=2))
        return 0
    who = confirmed_label(view["confirmed_by"], view["confirmed_via"])
    print(f"Run {state.run_id}, protocol sha256 {view['protocol_sha256'][:12]}, confirmed by {who}")
    print()
    print("\n".join(results_lines(view)))
    print()
    print(_summary_line(view) + (". Locked values are withheld until their events are settled." if view["summary"]["locked"] else "."))
    if blockers := _blocker_lines(view):
        print("\n".join(blockers))
    return 0


def cmd_manifest(args: argparse.Namespace, runner: Runner) -> int:
    manifest = runner.manifest(args.run, review=args.review)
    state = runner.load(args.run)
    text = json.dumps(manifest, indent=2)
    lines = [f"Manifest {runner.path(args.run) / 'manifest.json'}"]
    if args.out is not None:
        Path(args.out).write_text(text, encoding="utf-8")
        lines.append(f"Also written to {args.out}")
    p = manifest["protocol"]
    s = manifest["summary"]
    lines += [
        f"sha256 {state.manifest_sha256}",
        f"protocol sha256 {p['sha256']}, confirmed by {p['confirmed_by']}",
        f"{s['verified']} verified, {s['locked']} locked, {s['refused']} refused",
    ]
    for review in manifest["review"]:
        lines += ["", f"Review ({review.get('model', 'reviewer')}): {review['verdict']}"]
        lines += [f"  - [{f['severity']}] {f['claim']} {f['suggestion']}".rstrip() for f in review["findings"]]
    lines += ["", "Methods:", textwrap.fill(manifest["methods"], width=100)]
    print("\n".join(lines))
    return 0


def cmd_runs(args: argparse.Namespace, runner: Runner) -> int:
    rows = [public.run_row(s) for s in runner.list()]
    if not rows:
        print(f"No runs in {runner.home}")
        return 0
    table = [
        [
            r["run_id"], r["created_at"], r["video"], r["stage"], (r["protocol_sha256"] or "-")[:12],
            confirmed_label(r["confirmed_by"], r["confirmed_via"]) if r["confirmed_by"] else "-",
        ]
        for r in rows
    ]
    print("\n".join(_table(["run", "created", "video", "stage", "protocol", "confirmed by"], table)))
    return 0


def cmd_batch(args: argparse.Namespace, runner: Runner) -> int:
    if not args.by.strip():
        raise RunError("a confirmation needs the name of the person confirming")
    body = _read_json(args.protocol)
    protocol = AssayProtocol.model_validate(body)
    if protocol.arena.corners_px is None:
        raise RunError("the protocol has no arena corners; a cohort shares one protocol, so mark them once in the file")
    digest = protocol.content_hash()
    sampling = protocol.sampling
    _err(f"protocol {protocol.title!r}, sha256 {digest[:12]}, confirmed by {args.by.strip()}")
    rows, codes = [], []
    for i, video in enumerate(args.videos, start=1):
        label = f"[{i}/{len(args.videos)}] {Path(video).name}"
        run_id = "-"
        try:
            source = Path(video)
            if not source.is_file():
                raise RunError(f"no video at {source}")
            state = runner.create(source, start_s=sampling.start_s, end_s=sampling.end_s, fps=sampling.fps)
            run_id = state.run_id
            runner.confirm(run_id, body, by=args.by, via="cli")
            with _Progress(f"{label} tracking", "lines") as progress:
                runner.track(run_id, on_delta=progress.lines)
            manifest = runner.manifest(run_id)
            if manifest["protocol"]["sha256"] != digest:
                raise RunError("the run's protocol hash differs from the cohort's")
            s = manifest["summary"]
            rows.append([Path(video).name, run_id, s["verified"], s["locked"], s["refused"], ""])
            codes.append(0)
        except public.EXPECTED_ERRORS as error:
            _err(f"{label}: {public.plain_error(error)}")
            rows.append([Path(video).name, run_id, "-", "-", "-", public.plain_error(error)])
            codes.append(2 if isinstance(error, RunError) else 1)
    print("\n".join(_table(["video", "run", "verified", "locked", "refused", "error"], rows)))
    ok = codes.count(0)
    print(f"\n{ok} of {len(codes)} videos measured under protocol sha256 {digest}")
    if any(c == 1 for c in codes):
        return 1
    return 2 if any(codes) else 0


# -- entry point ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kappa",
        description="A measuring instrument made from any camera.",
        epilog="Exit status: 0 ok, 2 when Kappa refuses a step, 1 for anything else.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("new", help="start a run from a video; prints the run id")
    p.add_argument("video")
    p.add_argument("--start", type=float, default=0.0, metavar="S", help="start at S seconds")
    p.add_argument("--end", type=float, default=None, metavar="S", help="stop at S seconds")
    p.add_argument("--fps", type=float, default=None, metavar="F", help="resample to F frames per second")
    p.add_argument("--id", default=None, help="run id (default: random)")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("draft", help="draft a protocol from an English question")
    p.add_argument("run")
    p.add_argument("question")
    p.set_defaults(func=cmd_draft)

    p = sub.add_parser("confirm", help="a person accepts the protocol; only a confirmed protocol runs")
    p.add_argument("run")
    p.add_argument("--by", required=True, metavar="NAME", help="the person confirming")
    p.add_argument("--protocol", default=None, metavar="FILE", help="protocol JSON (default: the run's draft)")
    p.add_argument("--sha256", default=None, metavar="HASH", help="sha256 (or its first 12+ characters) of the protocol you reviewed")
    p.set_defaults(func=cmd_confirm)

    p = sub.add_parser("track", help="track every subject with SAM 3.1")
    p.add_argument("run")
    p.set_defaults(func=cmd_track)

    p = sub.add_parser("score", help="score one behaviour of the protocol")
    p.add_argument("run")
    p.add_argument("behaviour", metavar="behaviour_id")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("repair", help="settle who is who on one frame, then re-track from there")
    p.add_argument("run")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--frame", type=int, metavar="N", help="the frame you marked with --subject or --box")
    where.add_argument(
        "--from-proposal", type=int, metavar="N", help="apply the marks proposed for frame N (repair.proposal-N.json)"
    )
    p.add_argument("--by", required=True, metavar="NAME", help="the person marking or approving the marks")
    p.add_argument("--sha256", default=None, metavar="HASH", help="with --from-proposal: sha256 (12+ characters) of the marks you reviewed")
    p.add_argument("--subject", type=_subject_point, action="append", metavar="ID:X,Y", help="a point on subject ID")
    p.add_argument("--box", type=_subject_box, action="append", metavar="ID:X1,Y1,X2,Y2", help="a box around subject ID")
    p.set_defaults(func=cmd_repair)

    p = sub.add_parser("results", help="every result with its status; locked values are withheld")
    p.add_argument("run")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_results)

    p = sub.add_parser("manifest", help="write the run's manifest")
    p.add_argument("run")
    p.add_argument("--review", action="store_true", help="also ask Muse Wonder for a reviewer's read")
    p.add_argument("--out", default=None, metavar="FILE", help="also write the manifest JSON here")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("runs", help="list runs")
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("batch", help="run one confirmed protocol on many videos")
    p.add_argument("protocol", metavar="protocol.json")
    p.add_argument("videos", nargs="+", metavar="video")
    p.add_argument("--by", required=True, metavar="NAME", help="the person confirming the protocol")
    p.set_defaults(func=cmd_batch)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # a console that can't print '±' gets '?'
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit:  # usage errors and --help
        return 0 if exit.code in (0, None) else 1
    try:
        if getattr(args, "run", None) is not None:
            public.check_run_id(args.run)
        return args.func(args, runner_factory())
    except RunError as error:
        _err(str(error))
        return 2
    except public.EXPECTED_ERRORS as error:
        _err(public.plain_error(error))
        return 1
    except KeyboardInterrupt:
        _err("interrupted")
        return 1
    except Exception as error:
        if os.environ.get("KAPPA_DEBUG"):
            raise
        detail = str(error).rstrip(".")
        _err(f"unexpected error ({type(error).__name__}): {detail}. Set KAPPA_DEBUG=1 for the traceback.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
