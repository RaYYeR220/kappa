"""Track a video on the deployed kappa-sam service through the engine's ModalSam backend.

The raw text lands in the engine store exactly as a Model API run would put it, then
is parsed locally (CPU) and summarised. With ``--gt`` (a 3D-ZeF ``gt.txt``) it also
scores identity against ground truth (IDF1, ID switches, MOTA)::

    cd engine
    uv run python ../tracker/spike.py --video ../data/cache/zf02_top_half.mp4 --concept zebrafish \\
        --gt ../data/raw/zef/data/ZebraFish-02/gt/gt.txt --gt-scale 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from kappa.eval.mot import clear_mot, idf1
from kappa.sam.client import track_concept
from kappa.sam.modal_backend import ModalSam
from kappa.sam.parse import parse_output
from kappa.store import Store

ROOT = Path(__file__).resolve().parents[1]


def _q(values, qs=(0, 5, 25, 50, 75, 95, 100)) -> dict[str, float]:
    if len(values) == 0:
        return {}
    v = np.percentile(np.asarray(values, dtype=float), qs)
    return {f"p{q}": round(float(x), 4) for q, x in zip(qs, v)}


def summarise(detections, n_frames: int) -> dict:
    per_frame = Counter(d.frame for d in detections)
    by_id: dict[int, list] = {}
    for d in detections:
        by_id.setdefault(d.track_id, []).append(d)
    jump, where = 0.0, None
    for tid, ds in by_id.items():
        ds.sort(key=lambda d: d.frame)
        for a, b in zip(ds, ds[1:]):
            if b.frame == a.frame + 1:
                j = float(np.hypot(b.cx - a.cx, b.cy - a.cy))
                if j > jump:
                    jump, where = j, {"id": tid, "frame": b.frame}
    return {
        "frames": n_frames,
        "frames_with_detections": len(per_frame),
        "objects_per_frame": dict(sorted(Counter(per_frame.values()).items())),
        "frames_without_detections": n_frames - len(per_frame),
        "distinct_ids": sorted(by_id),
        "frames_per_id": {tid: len(ds) for tid, ds in sorted(by_id.items())},
        "confidence": _q([d.confidence for d in detections if d.confidence is not None]),
        "mask_area_px": _q([d.area_px for d in detections]),
        "max_centroid_jump_px": round(jump, 2),
        "max_jump_at": where,
    }


def load_zef_gt(path: Path, n_frames: int, scale: float) -> list[dict[int, tuple]]:
    """3D-ZeF gt.txt: frame (1-based), id, ..., camT left/top/width/height in columns 7-10 (0-based)."""
    frames: list[dict[int, tuple]] = [{} for _ in range(n_frames)]
    for row in path.read_text().splitlines():
        c = row.split(",")
        f, oid = int(c[0]) - 1, int(c[1])
        left, top, w, h = (float(x) * scale for x in c[7:11])
        if 0 <= f < n_frames and w > 0 and h > 0:
            frames[f][oid] = (left, top, left + w, top + h)
    return frames


def identity_scores(detections, gt_path: Path, n_frames: int, scale: float) -> dict:
    gt = load_zef_gt(gt_path, n_frames, scale)
    pred: list[dict[int, tuple]] = [{} for _ in range(n_frames)]
    for d in detections:
        if d.frame < n_frames:
            pred[d.frame][d.track_id] = d.box
    out = {}
    for thr in (0.5, 0.3):
        ids = idf1(gt, pred, iou_threshold=thr)
        mot = clear_mot(gt, pred, iou_threshold=thr)
        out[f"iou{thr}"] = {
            "idf1": round(ids.idf1, 4), "idp": round(ids.idp, 4), "idr": round(ids.idr, 4),
            "assignment": {int(k): int(v) for k, v in ids.assignment.items()},
            "mota": round(mot.mota, 4), "motp": round(mot.motp, 4), "id_switches": mot.id_switches,
            "fp": mot.fp, "fn": mot.fn, "n_gt": mot.n_gt,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--concept", required=True)
    p.add_argument("--frames", type=int, default=None, help="frame count (default: from ffprobe)")
    p.add_argument("--gt", type=Path, default=None)
    p.add_argument("--gt-scale", type=float, default=1.0)
    p.add_argument("--chunk-frames", type=int, default=None)
    p.add_argument("--overlap", type=int, default=None)
    p.add_argument("--carry", choices=("match", "prompt"), default=None)
    p.add_argument("--gpu", default=None)
    p.add_argument("--out", type=Path, default=None, help="write the summary JSON here")
    args = p.parse_args()

    home = Path(os.environ.get("KAPPA_HOME", ROOT / "data" / "home"))
    store = Store(home / "store")
    sam = ModalSam(store=store, gpu=args.gpu, chunk_frames=args.chunk_frames, overlap=args.overlap,
                   carry=args.carry)
    t0 = time.time()
    file_id = sam.upload(args.video)
    t_up = time.time() - t0
    count = [0]

    def progress(delta: str) -> None:
        count[0] += delta.count("\n")
        if count[0] % 500 == 0 and delta.endswith("\n"):
            print(f"  {count[0]} lines  ({time.time() - t0:.0f} s)", flush=True)

    raw, digest = track_concept(sam, file_id, args.concept, on_delta=progress)
    wall = time.time() - t0
    t1 = time.time()
    parsed = parse_output(raw)
    t_parse = time.time() - t1
    n_frames = args.frames or (sam.last_stats or {}).get("frames") or (max(d.frame for d in parsed.detections) + 1)
    summary = {
        "video": str(args.video.name),
        "concept": args.concept,
        "raw_sha256": digest,
        "raw_bytes": len(raw.encode()),
        "upload_s": round(t_up, 1),
        "wall_s": round(wall, 1),
        "parse_s": round(t_parse, 1),
        "parse_diagnostics": list(parsed.diagnostics)[:5],
        **summarise(parsed.detections, n_frames),
        "service_stats": sam.last_stats,
    }
    if args.gt is not None:
        summary["vs_ground_truth"] = identity_scores(parsed.detections, args.gt, n_frames, args.gt_scale)
    text = json.dumps(summary, indent=2, default=str)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
