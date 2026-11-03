"""kappa-sam: Meta's open-weight SAM 3.1 on a Modal GPU, speaking the Model API's wire format.

Kappa was built against Meta Model API's hosted ``sam-3.1``. This service runs the same
model family ourselves (the official ``sam3.1_multiplex.pt`` checkpoint, sha256-pinned)
and streams back exactly the text the hosted API streams: one line per frame,
``<Nf>id<|box;...;c=..|><|mask;x=0;y=0;c=..;data=h,w,!...|>,...``, boxes inclusive,
masks cropped to the box and ``one_bit``-encoded, frames with nothing visible omitted.
The engine's parser and the web app's ``@meta-sam/parser`` read it unchanged.

Deploy::

    modal deploy tracker/app.py

Call (the engine's ``ModalSam`` backend does this)::

    Sam31 = modal.Cls.from_name("kappa-sam", "Sam31")
    for line in Sam31().track.remote_gen("videos/<sha256>.mp4", "mouse"):
        ...

``text`` is either a noun phrase (concept prompt) or object prompts in the Model API's
grammar (``<30f>1<|box;...|>,2<|point;...|>-<|point;...|>``) with
``propagation_direction`` ``both`` / ``forward`` / ``backward``. ``video`` is a path in
the ``kappa-sam-videos`` volume or the mp4's bytes. Run statistics (fps, GPU seconds,
estimated cost, peak memory per chunk) go to the logs and to the ``kappa-sam-stats``
Dict under ``job``; the stream itself carries nothing but grammar lines.

Chunking (long videos)
----------------------
A SAM 3.1 session holds every frame on the GPU as a 1008x1008 fp16 tensor (6.1 MB)
plus per-frame tracker memory, and its per-frame cost grows with the frames it has seen;
a 15,000-frame video would need ~150 GB in one session. So a video runs in chunks of
``chunk_frames`` (default 400 = 16 s at 25 fps: ~2.4 GB of frames, 11-20 GB peak on an
H100), each a fresh session closed before the next. Consecutive chunks share
``overlap`` frames (default 48): a chunk starts 48 frames before the previous one ended
and emits only frames after it, so every frame is emitted exactly once.

Identity across a boundary (``carry``):

* ``prompt`` (default, and always used for box/point prompts): each object seen in the
  previous chunk's last 48 frames is prompted in the next chunk, on the last frame it
  was seen, with its box plus a positive point at its mask's most interior pixel (the
  point says which body is meant when two boxes overlap), under the same id. The
  tracker continues across the boundary with no warm-up, and propagation starts at the
  earliest carried prompt, so the overlap costs nothing when objects are visible at
  the end of a chunk; the 48 frames let an object missed on the last frames still be
  carried. Concept detection runs on the first chunk only; a chunk with nothing to
  carry runs the concept detector again (new ids above any used so far).
* ``match`` (optional, concepts only): every chunk runs the concept detector + tracker
  on its own, and its objects take the ids of the previous chunk's objects they
  overlap (mean mask IoU >= 0.3 over >= 3 shared frames where both are present,
  one-to-one); the rest get new ids. Recovers lost objects, but a fresh concept session
  needs ~20-30 frames before SAM 3.1 emits an object, so frames are lost and ids can
  break at boundaries.

Measured (H100, chunk 400, overlap 48): mouse, 1,500 frames of OFT_12: ``prompt`` 1 id,
1,488/1,500 frames, 11.2 fps; ``match`` 4 ids, 1,377/1,500 frames, 12.2 fps. Zebrafish
(3D-ZeF ZebraFish-02, 900 frames, 5 fish): ``prompt`` IDF1 0.490 / 23 ID switches /
5.3 fps; ``match`` 0.492 / 7 / 9.0 fps (both ~50 % precision: SAM 3.1 also segments
the fish's reflections in the tank floor). One unchunked 900-frame session: 0.372 / 9.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import modal

APP_NAME = "kappa-sam"
CLASS_NAME = "Sam31"
# Bump with every change to what the service returns; the engine names it in its cache
# keys and sends it with each call, and a container of another version refuses the call.
SERVICE_VERSION = 3
SAM3_COMMIT = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"  # facebookresearch/sam3 main, 2026-10-23
CKPT_NAME = "sam3.1_multiplex.pt"
CKPT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"  # facebook/sam3.1 LFS
MIRRORS = (
    "jetjodh/sam3.1",
    "research21/sam3.1",
    "PIA-SPACE-LAB/sam3.1",
    "jtreminio/sam3.1",
    "AEmotionStudio/sam3.1",
)
WEIGHTS_DIR = "/weights"
VOLUME_NAME = "kappa-sam-videos"
VOLUME_MOUNT = "/data"
STATS_DICT = "kappa-sam-stats"
GPU = "H100"  # cheapest per frame as well as fastest: see the run statistics
CHUNK_FRAMES = 400
SMALL_GPU_CHUNK = 400  # cap on GPUs under 30 GB
OVERLAP_FRAMES = 48
CARRY = "prompt"
GROUNDING_BATCH = 4
IMAGE_SIZE = 1008
CPU_CORES = 4.0
MEMORY_MIB = 16384
# Modal list prices (USD per second), for the cost estimate in the run statistics.
GPU_USD_PER_S = {"H100": 0.001097, "A100-SXM4-80GB": 0.000694, "A100": 0.000583, "L40S": 0.000542, "A10": 0.000306,
                 "L4": 0.000222}
CPU_USD_PER_CORE_S = 0.0000131
MEM_USD_PER_GIB_S = 0.00000222

_HERE = Path(__file__).resolve().parent
_ENCODER = _HERE.parent / "engine" / "src" / "kappa" / "sam" / "fastencode.py"


def _fetch_weights() -> None:
    """Download the checkpoint from the first mirror whose bytes hash to the official sha256."""
    import hashlib
    import shutil

    from huggingface_hub import hf_hub_download

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    errors = []
    for repo in MIRRORS:
        try:
            path = hf_hub_download(repo_id=repo, filename=CKPT_NAME, cache_dir="/tmp/hf")
        except Exception as error:  # noqa: BLE001 - try the next mirror
            errors.append(f"{repo}: {error}")
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while block := handle.read(1 << 24):
                digest.update(block)
        if digest.hexdigest() != CKPT_SHA256:
            errors.append(f"{repo}: sha256 {digest.hexdigest()} != {CKPT_SHA256}")
            continue
        shutil.copyfile(path, f"{WEIGHTS_DIR}/{CKPT_NAME}")
        for extra in ("config.json",):
            try:
                shutil.copyfile(hf_hub_download(repo_id=repo, filename=extra, cache_dir="/tmp/hf"), f"{WEIGHTS_DIR}/{extra}")
            except Exception:  # noqa: BLE001 - the builder does not read it
                pass
        with open(f"{WEIGHTS_DIR}/SOURCE", "w") as handle:
            json.dump({"repo": repo, "sha256": CKPT_SHA256}, handle)
        shutil.rmtree("/tmp/hf", ignore_errors=True)
        print(f"checkpoint from {repo}, sha256 verified")
        return
    raise RuntimeError("no mirror served the official SAM 3.1 checkpoint:\n" + "\n".join(errors))


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision==0.25.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        f"sam3 @ git+https://github.com/facebookresearch/sam3.git@{SAM3_COMMIT}",
        "numpy==1.26.4",
        "numba==0.61.2",
        "opencv-python-headless==4.10.0.84",
        "setuptools<81",  # sam3 imports pkg_resources
        "einops",
        "pycocotools",
        "psutil",
        "scipy",
    )
    .run_function(_fetch_weights)
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TQDM_DISABLE": "1"})
    .add_local_file(_ENCODER, "/root/fastencode.py")
    .add_local_file(_HERE / "sam_wire.py", "/root/sam_wire.py")
)

app = modal.App(APP_NAME)
videos = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
stats_store = modal.Dict.from_name(STATS_DICT, create_if_missing=True)


class _FrameReader:
    """Sequential RGB frames from an mp4, with cheap re-reads of a recent range."""

    def __init__(self, path: str) -> None:
        import cv2

        self._cv2 = cv2
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video {path}")
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        self._cap, self._pos = cap, 0
        self._recent: dict[int, object] = {}

    def read(self, lo: int, hi: int) -> list:
        """Frames ``lo`` .. ``hi - 1`` (fewer if the video ends early)."""
        cv2 = self._cv2
        out, start = [], lo
        while start < hi and start in self._recent:  # the overlap with the previous read
            out.append(self._recent[start])
            start += 1
        if start < self._pos:  # needed frames were dropped from the cache: start over
            self._cap.release()
            self._cap, self._pos, out, start = cv2.VideoCapture(self.path), 0, [], lo
        while self._pos < start:
            if not self._cap.grab():
                return out
            self._pos += 1
        while self._pos < hi:
            ok, frame = self._cap.read()
            if not ok:
                self.count = self._pos
                break
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._pos += 1
        self._recent = {lo + i: f for i, f in enumerate(out)}
        return out


@app.cls(
    image=image,
    gpu=GPU,
    volumes={VOLUME_MOUNT: videos},
    timeout=4 * 3600,
    scaledown_window=120,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
)
class Sam31:
    @modal.enter()
    def load(self) -> None:
        import numpy as np
        import sam3.model.sam3_multiplex_tracking as tracking
        import torch
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        from sam_wire import encode_mask

        t0 = time.time()
        # FlashAttention 3 is Hopper-only; L4/A10G use PyTorch SDPA.
        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=f"{WEIGHTS_DIR}/{CKPT_NAME}", use_fa3=False, compile=False, async_loading_frames=False
        )
        self.model = self.predictor.model
        # batches of 4 frames through the detector: ~97 % of batch-16 speed at half the memory
        self.model.batched_grounding_batch_size = GROUNDING_BATCH
        self._preloaded: dict[str, tuple] = {}
        original = tracking.load_resource_as_video_frames

        def load_resource(resource_path, *args, **kwargs):
            if isinstance(resource_path, str) and resource_path in self._preloaded:
                return self._preloaded.pop(resource_path)
            return original(resource_path, *args, **kwargs)

        tracking.load_resource_as_video_frames = load_resource
        encode_mask(np.ones((3, 3), dtype=np.uint8))  # compile the encoder now
        self.gpu_name = torch.cuda.get_device_name(0)
        self.gpu_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        self.gpu_rate = next((r for k, r in GPU_USD_PER_S.items() if k in self.gpu_name), 0.0)
        self.load_s = time.time() - t0
        print(f"[kappa-sam] model loaded on {self.gpu_name} in {self.load_s:.1f} s")

    # -- sessions --------------------------------------------------------------

    def _images(self, frames: list):
        """Frames as the model's input: 1008x1008, /255, (x - 0.5) / 0.5, fp16, on the GPU.

        The same transform as sam3's torchcodec loader (bicubic, no antialias). sam3's
        cv2 mp4 loader at this commit skips the /255, so frames are prepared here.
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        out = torch.empty((len(frames), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float16, device="cuda")
        for i in range(0, len(frames), 32):
            batch = torch.from_numpy(np.stack(frames[i : i + 32])).cuda(non_blocking=True)
            x = batch.permute(0, 3, 1, 2).float()
            x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=False).half()
            x /= 255
            x -= 0.5
            x /= 0.5
            out[i : i + len(batch)] = x
        return out

    def _open(self, frames: list, width: int, height: int, prompted: bool) -> tuple[str, dict]:
        key = f"<kappa-chunk-{uuid.uuid4().hex}>"
        self._preloaded[key] = (self._images(frames), height, width)
        state = self.model.init_state(resource_path=key)
        if prompted:
            # Box/point prompts without a text prompt: SAM 3.1 merges tracker masks into
            # per-frame outputs that only exist after a detector pass; start them empty.
            state["cached_frame_outputs"] = {t: {} for t in range(state["num_frames"])}
        sid = key
        now = time.time()
        self.predictor._all_inference_states[sid] = {
            "state": state, "session_id": sid, "start_time": now, "last_use_time": now
        }
        return sid, state

    def _close(self, sid: str) -> None:
        import torch

        self.predictor.close_session(sid)
        torch.cuda.empty_cache()

    def _propagate(self, sid: str, state: dict, start: int, logits: bool) -> Iterator[tuple]:
        """(local frame, ids, masks, confidences) per frame, forward from ``start``."""
        import numpy as np

        for r in self.predictor.handle_stream_request(
            dict(type="propagate_in_video", session_id=sid, propagation_direction="forward",
                 start_frame_index=start, evict_cached_frame_outputs=True)
        ):
            f, out = r["frame_index"], r["outputs"]
            ids = np.asarray(out["out_obj_ids"], dtype=np.int64)
            if ids.size == 0:
                yield f, ids, None, None
                continue
            frame_scores = state["tracker_metadata"].get("obj_id_to_sam2_score_frame_wise", {}).get(f, {})
            conf = []
            for i, oid in enumerate(ids.tolist()):
                s = frame_scores.get(oid)
                if s is None:
                    c = float(out["out_probs"][i])
                else:
                    c = float(s)
                    if logits:
                        c = 1.0 / (1.0 + np.exp(-c))
                conf.append(c)
            yield f, ids, np.asarray(out["out_binary_masks"], dtype=bool), np.asarray(conf)

    def _chunk(self, frames, width, height, text=None, prompts=()) -> Iterator[tuple]:
        """Run one session over ``frames`` with a text prompt or object prompts (local frames)."""
        import torch

        prompted = text is None
        sid, state = self._open(frames, width, height, prompted)
        try:
            if text is not None:
                self.predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=text))
            for p in sorted(prompts, key=lambda p: (p.frame, p.obj_id)):
                coords, labels = p.sam_inputs(width, height)
                self.predictor.handle_request(dict(
                    type="add_prompt", session_id=sid, frame_index=p.frame, obj_id=p.obj_id,
                    points=torch.tensor(coords, dtype=torch.float32),
                    point_labels=torch.tensor(labels, dtype=torch.int32),
                ))
            start = min(p.frame for p in prompts) if prompted else 0
            yield from self._propagate(sid, state, start, logits=prompted)
        finally:
            self._close(sid)

    # -- passes ------------------------------------------------------------------

    def _pass(self, reader, order, chunk, overlap, stats, text=None, prompts=(), emit_first=True, carry=CARRY):
        """Track over the global frames ``order`` (processing order) in chunks.

        Yields (global frame, ids, masks, confidences). ``prompts`` are user object prompts
        (global frames); ``text`` a concept on the first frame of ``order``.

        Identity across a chunk boundary: with ``carry="prompt"`` (always, for object
        prompts) the next chunk is prompted with each object's box + interior point under
        its id. With ``carry="match"`` (concept prompts) every chunk runs the concept
        detector and its objects take the ids of the previous chunk's objects they overlap
        (mask IoU over the ``overlap`` shared frames, one-to-one); the rest get new ids.
        """
        import numpy as np
        import torch
        from sam_wire import carry_prompt, match_ids

        n = len(order)
        position = {g: i for i, g in enumerate(order)}
        user = [p for p in prompts if p.frame in position]
        a, b = 0, min(n, chunk)
        carried: list = []
        next_id = 0
        concept = text is not None
        emit_from = 0 if emit_first else 1
        previous: dict[int, tuple] = {}  # pos -> (ids, small masks) of the last chunk's tail
        while a < n:
            lo, hi = min(order[a], order[b - 1]), max(order[a], order[b - 1])
            frames = reader.read(lo, hi + 1)
            if order[0] > order[-1]:
                frames = frames[::-1]
            if len(frames) < b - a:  # the video ended early
                b = a + len(frames)
                if b <= a:
                    break
            local = [
                type(p)(frame=position[p.frame] - a, obj_id=p.obj_id, box=p.box, points=list(p.points))
                for p in user if a <= position[p.frame] < b
            ] + [p for p in carried]
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            if concept:
                source = self._chunk(frames, reader.width, reader.height, text=text)
            elif local:
                source = self._chunk(frames, reader.width, reader.height, prompts=local)
            else:
                break  # nothing left to track
            # local id -> global id; the first chunk keeps the model's ids, a re-detected
            # chunk is matched against the previous chunk's tail once that has gone by
            mapping: dict[int, int] | None = {} if (a == 0 or not concept) else None
            identity = a == 0 or not concept
            pending: list[tuple] = []
            recent: dict[int, tuple[int, np.ndarray]] = {}
            tail: dict[int, tuple] = {}
            for f, ids, masks, conf in source:
                pos = a + f
                if mapping is None:
                    if pos < emit_from:
                        if ids.size:
                            pending.append((pos, ids, masks[:, ::2, ::2]))
                        continue
                    mapping = match_ids(previous, pending) if carry == "match" else {}
                    stats["chunks_matched"].append({"at": order[a], "pairs": {int(k): int(v) for k, v in mapping.items()}})
                if ids.size == 0:
                    continue
                if not identity:
                    for oid in ids.tolist():
                        if oid not in mapping:
                            mapping[oid] = next_id
                            next_id += 1
                    ids = np.array([mapping[i] for i in ids.tolist()], dtype=np.int64)
                for i, oid in enumerate(ids.tolist()):
                    next_id = max(next_id, oid + 1)
                    if masks[i].any():
                        recent[oid] = (pos, masks[i])
                if pos >= b - overlap:
                    tail[pos] = (ids, masks[:, ::2, ::2])
                if pos >= emit_from:
                    yield order[pos], ids, masks, conf
            stats["chunks"].append({
                "frames": [order[a], order[b - 1]], "mode": "concept" if concept else "prompts",
                "objects_in": len(local) if not concept else None, "wall_s": round(time.time() - t0, 2),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            })
            if b >= n:
                break
            next_a = max(b - overlap, a + 1)
            carried = []
            if text is None or carry == "prompt":
                # hand every object seen in the last `overlap` frames to the next chunk
                for oid, (pos, mask) in sorted(recent.items()):
                    if pos >= b - overlap and (p := carry_prompt(pos - next_a, oid, mask)) is not None:
                        carried.append(p)
            user = [p for p in user if position[p.frame] >= b]
            previous = tail
            emit_from = b
            concept = text is not None and (carry == "match" or not carried)
            a, b = next_a, min(n, next_a + chunk)

    def _run(self, path: str, text: str, direction: str, chunk: int, overlap: int, carry: str,
             stats: dict) -> Iterator[str]:
        from sam_wire import (
            DIRECTIONS,
            PromptError,
            format_line,
            is_object_prompt,
            parse_object_prompts,
        )

        reader = _FrameReader(path)
        w, h, n = reader.width, reader.height, reader.count
        stats.update(width=w, height=h, frames=n, fps=reader.fps)
        if overlap < 1 or chunk <= 2 * overlap:
            raise ValueError(f"need chunk_frames > 2 * overlap >= 2, got {chunk}/{overlap}")
        if carry not in ("match", "prompt"):
            raise ValueError(f"carry must be match or prompt, got {carry!r}")

        def lines(source):
            for g, ids, masks, conf in source:
                line = format_line(g, ids, masks, conf, w, h)
                if line is not None:
                    stats["lines"] += 1
                    yield line

        if not is_object_prompt(text):
            if not text.strip():
                raise PromptError("the prompt is empty")
            stats["mode"] = "concept"
            yield from lines(self._pass(reader, list(range(n)), chunk, overlap, stats, text=text.strip(),
                                        carry=carry))
            return
        if direction not in DIRECTIONS:
            raise PromptError(f"metadata[\"propagation_direction\"] must be one of {list(DIRECTIONS)}")
        prompts = parse_object_prompts(text, w, h)
        start = min(p.frame for p in prompts)
        if any(p.frame >= n for p in prompts):
            raise PromptError(f"a prompt frame is outside the video (0-{n - 1})")
        if direction == "backward" and start == 0:
            raise PromptError("backward propagation needs an object prompt on a frame after the first")
        stats["mode"] = f"prompts/{direction}"
        forward_last: dict[int, object] = {}
        if direction in ("both", "forward"):
            for g, ids, masks, conf in self._pass(reader, list(range(start, n)), chunk, overlap, stats, prompts=prompts):
                if g == start:
                    forward_last = {int(i): m for i, m in zip(ids.tolist(), masks)}
                line = format_line(g, ids, masks, conf, w, h)
                if line is not None:
                    stats["lines"] += 1
                    yield line
        if direction in ("both", "backward"):
            from sam_wire import carry_prompt

            seeds = [p for p in prompts if p.frame == start]
            seeded = {p.obj_id for p in seeds}
            # objects first prompted later were tracked onto `start` by the forward pass
            seeds += [q for oid, m in forward_last.items() if oid not in seeded
                      if (q := carry_prompt(start, oid, m)) is not None]
            order = list(range(start, -1, -1))
            yield from lines(self._pass(reader, order, chunk, overlap, stats, prompts=seeds,
                                        emit_first=direction == "backward"))

    @modal.method()
    def track(
        self,
        video: str | bytes,
        text: str,
        propagation_direction: str = "both",
        chunk_frames: int = CHUNK_FRAMES,
        overlap: int = OVERLAP_FRAMES,
        job: str | None = None,
        carry: str = CARRY,
        service_version: int | None = None,
    ) -> Iterator[str]:
        """Stream the Model API's output lines for ``video`` and ``text``, one per frame."""
        if service_version is not None and service_version != SERVICE_VERSION:
            raise ValueError(f"this kappa-sam container serves version {SERVICE_VERSION}, the caller expects "
                             f"{service_version}; redeploy tracker/app.py or retry once the rollout is done")
        t0 = time.time()
        if self.gpu_gib < 30:
            chunk_frames = min(chunk_frames, SMALL_GPU_CHUNK)
        path = self._resolve(video)
        job = job or uuid.uuid4().hex
        stats: dict = {"job": job, "gpu": self.gpu_name, "lines": 0, "chunks": [], "chunks_matched": [],
                       "chunk_frames": chunk_frames, "overlap": overlap, "carry": carry,
                       "service_version": SERVICE_VERSION, "sam3_commit": SAM3_COMMIT, "checkpoint_sha256": CKPT_SHA256}
        q: queue.Queue = queue.Queue(maxsize=4096)
        done = object()

        def produce():
            import torch

            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for line in self._run(path, text, propagation_direction, chunk_frames, overlap, carry, stats):
                        q.put(line)
            except BaseException as error:  # noqa: BLE001 - re-raised in the caller's thread
                q.put(error)
            finally:
                q.put(done)

        worker = threading.Thread(target=produce, daemon=True)
        worker.start()
        error = None
        while (item := q.get()) is not done:
            if isinstance(item, BaseException):
                error = item
                continue
            yield item
        worker.join()
        wall = time.time() - t0
        rate = self.gpu_rate + CPU_CORES * CPU_USD_PER_CORE_S + MEMORY_MIB / 1024 * MEM_USD_PER_GIB_S
        stats.update(
            wall_s=round(wall, 2),
            gpu_s=round(wall, 2),
            fps=round(stats.get("frames", 0) / wall, 2) if wall > 0 else None,
            usd_estimate=round(wall * rate, 4),
            model_load_s=round(self.load_s, 1),
            error=repr(error) if error else None,
        )
        print("[kappa-sam] stats " + json.dumps(stats))
        try:
            stats_store[job] = stats
        except Exception as e:  # noqa: BLE001 - stats are best effort
            print(f"[kappa-sam] could not store stats: {e}")
        if error is not None:
            raise error

    def _resolve(self, video: str | bytes) -> str:
        import hashlib

        if isinstance(video, (bytes, bytearray)):
            path = f"/tmp/{hashlib.sha256(video).hexdigest()}.mp4"
            if not os.path.exists(path):
                with open(path, "wb") as handle:
                    handle.write(video)
            return path
        path = video if video.startswith("/") else f"{VOLUME_MOUNT}/{video.lstrip('/')}"
        if not os.path.exists(path):
            videos.reload()
        if not os.path.exists(path):
            raise FileNotFoundError(f"{video} is not in the {VOLUME_NAME} volume")
        return path


@app.local_entrypoint()
def main(video: str, text: str, out: str = "", direction: str = "both", chunk_frames: int = CHUNK_FRAMES,
         overlap: int = OVERLAP_FRAMES) -> None:
    """``modal run tracker/app.py --video clip.mp4 --text mouse --out raw.txt``: a quick check."""
    import hashlib

    local = Path(video)
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    remote = f"videos/{digest}.mp4"
    with videos.batch_upload(force=True) as batch:
        batch.put_file(str(local), remote)
    job = uuid.uuid4().hex
    sink = open(out, "w", encoding="utf-8") if out else None
    n = 0
    for line in Sam31().track.remote_gen(remote, text, direction, chunk_frames, overlap, job):
        n += 1
        if sink:
            sink.write(line)
        elif n <= 3:
            print(line[:200])
    if sink:
        sink.close()
    print(f"{n} lines; stats: {json.dumps(stats_store.get(job))}")
