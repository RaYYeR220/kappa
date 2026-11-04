# Kappa

**Every number, with the frames that prove it.**

Kappa is a measuring instrument made from any camera. Give it a video of a behavioural experiment (a mouse in an open field, five zebrafish in a tank) and ask in plain English what to measure. It drafts the assay, a researcher confirms it, and then every animal is tracked with SAM 3.1 and every number comes back in one of three states:

- **Verified**: the value, its uncertainty, and links to the frames it was computed from.
- **Locked**: the value is withheld because the track underneath it failed an integrity check (two animals touched and may have swapped identities, an animal vanished for too long, a position jumped faster than the animal can move). It unlocks when a person marks who is who on one frame and every check passes again.
- **Refused**: the question asks for something the video cannot show (heart rate from a top-down camera), and Kappa says so instead of approximating it.

The name is Cohen's κ, the statistic labs use to check that two human scorers agree. Kappa is graded the same way, against three human raters, with the evaluation pre-registered before any test video was run: [`eval/PREREGISTRATION.md`](eval/PREREGISTRATION.md).

**Live:** https://kappa-tan-three.vercel.app: two recorded runs you can replay with no account (an open-field mouse and five zebrafish), and the validation results.

## Results, as pre-registered

On the 15 held-out open-field videos (tracked once, after the configuration was frozen in commit `5280b9a`), Kappa's rearing bouts agree with the three human raters at a frame-level **κ of 0.530, 0.540 and 0.545 (mean 0.538)**. The raters agree with each other at 0.801, 0.816 and 0.823. **The pre-registered success criterion, Kappa's mean κ at or above the lowest human pair, is not met.** The numbers are in [`eval/results/summary.json`](eval/results/summary.json), per video in [`eval/results/detail.json`](eval/results/detail.json).

- The tuning-set estimate made before the test run (leave one video out, 0.515) predicted this result.
- Supported rears: κ 0.46 (humans 0.85 to 0.89). Unsupported rears: κ 0.10 (humans 0.68 to 0.74): our wall-distance split is a poor stand-in for what the raters judge.
- Rear count per video: ICC(2,1) 0.35 and Pearson r 0.73 against the mean rater (humans 0.94 to 0.97). Kappa finds more, shorter bouts than people do.
- Negative control: the same bouts shifted by 30 s score κ 0.012.
- What changed on the way, and why, is in Amendment 1. The first design asked Muse Wonder to judge each candidate clip. On the tuning videos it could not tell a rear from a walk in this footage (κ 0.10 to 0.15), so rearing is now read from the SAM 3.1 silhouettes by a small classifier trained on the five tuning videos only.

## Why

Behavioural neuroscience and pharmacology still score a large share of their video by hand. A ten-minute open-field test takes a trained person about an hour to score (Sturman et al., 2020), trained raters scoring the same videos reach a frame-level κ of only 0.67 to 0.89 depending on the behaviour, and the commercial trackers cost $3,500 to $30,000 per seat. The open-source deep-learning tools need hundreds of labelled frames and a GPU training run for every new setup.

Zero-shot segmentation changes the cost of the tracking. It does not change what a reviewer needs before they accept a number: a definition they can read, evidence they can check, and an instrument that tells them when it does not know. Kappa is built around that second part.

## How it works

```
video ──► normalise (constant frame rate, sha256)
  │
  ├─► Muse Spark Open: question + keyframes ──► draft protocol
  │       metrics only from a typed library; anything else is refused with a reason
  │       arena corners located by perception grounding
  │       a person confirms it; the confirmation is pinned to the protocol's sha256
  │
  ├─► SAM 3.1 (open weights, our endpoint): track the subject concept through the
  │       video (masks, ids, confidence)
  │
  ├─► subjects: tracklets → animals; short gaps interpolated, long ones flagged
  ├─► integrity: teleports, low confidence, contacts (identity decided by motion
  │       continuity, or left open), re-identifications, extra objects
  ├─► metrics: distance, speed, zone time/entries/latency, immobility, spacing,
  │       behaviour counts and durations; each number gated by the events under it
  ├─► behaviour: rearing read frame by frame from the SAM 3.1 silhouettes (area,
  │       perimeter, solidity, Hu moments, motion, wall distance, in context) by a
  │       small decision-tree classifier trained on the tuning videos only
  ├─► repair: a person marks the animals on one frame → SAM 3.1 re-tracks from there
  ├─► Reviewer 2: Muse Wonder reads the whole run and lists what a reviewer would doubt
  │
  └─► manifest: hashes of the video, protocol and every model output, all events and
          results, and a methods paragraph generated from those fields only
```

Model outputs are stored content-addressed and never requested twice. Analysis makes no model calls, so the same inputs always give the same manifest.

### Meta models: what each one does and where it runs

| Model | Weights | Where it runs | Role |
|---|---|---|---|
| **SAM 3.1** (`sam3.1_multiplex.pt`) | open | our own GPU endpoint on Modal (`tracker/`) | Text-prompted video tracking of every animal, with masks, stable ids and per-frame confidence; box and point prompts on any frame, propagated forward, for identity repair. The endpoint emits exactly the output text of Meta Model API's hosted `sam-3.1`, so either backend can be used. Masks are drawn in the browser with Meta's `@meta-sam/parser` and `@meta-sam/graphics`. |
| **Muse Spark Open** (816B MoE) | open | hosted inference (Venice) | Drafts the assay protocol from the question and keyframes with structured output, and locates the arena floor with perception grounding. |
| **Muse Wonder** | hosted | hosted inference (Venice) | Writes Reviewer 2: an adversarial read of the finished run (protocol, integrity events, every result) that lists what a careful methods reviewer would object to. Advice only: it never changes a number. (It no longer scores behaviour: on the tuning videos it could not tell a rear from a walk, see Amendment 1.) |

The same code also runs entirely on Meta Model API (`sam-3.1`, `muse-spark-1.3`, `muse-wonder-1.0`): the backends are switched with `KAPPA_SAM_BACKEND` and `KAPPA_LLM_BACKEND`. Every manifest records which model and backend produced each output.

## Repository

```
engine/   Python package `kappa`: pipeline, CLI, MCP server, HTTP API (uv)
tracker/  the SAM 3.1 endpoint on Modal (open weights, Model API output format)
web/      Next.js app: landing, run workspace, validation, docs
eval/     pre-registration and evaluation outputs
```

## Run it

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), ffmpeg, Node 20+ and pnpm. Model access is needed only for the steps that call a model: either a Meta Model API key, or a Modal account for the SAM 3.1 endpoint plus a Venice key for the Muse models.

```bash
cd engine
uv sync --all-extras
uv run pytest                      # the full suite runs offline

export KAPPA_SAM_BACKEND=modal KAPPA_LLM_BACKEND=venice VENICE_API_KEY=...
(cd ../tracker && modal deploy app.py)   # once: the SAM 3.1 endpoint
# or, on Meta Model API: export MODEL_API_KEY=...
uv run kappa new path/to/video.mp4 --id demo
uv run kappa draft demo "How far does the mouse travel, and how long does it spend in the centre?"
uv run kappa confirm demo --sha256 <shown by draft> --by "Your Name"
uv run kappa track demo
uv run kappa results demo
uv run kappa manifest demo --review
```

The web app replays exported runs with no server and no credentials:

```bash
cd web
pnpm install
pnpm dev                           # http://localhost:3000
```

### Other ways in

- **CLI**: `kappa new · draft · confirm · track · score · repair · results · manifest · runs · batch`. `batch` runs one confirmed protocol over a cohort of videos, so every animal is measured under the same protocol hash.
- **MCP server** (`kappa-mcp`, stdio): `create_run`, `draft_protocol`, `submit_for_confirmation`, `track`, `get_results`, `get_manifest`, `list_runs`, `propose_repair`, `score_behaviour`. An agent can draft and propose; it cannot confirm a protocol or settle an identity. Those happen through the CLI or the web workspace, pinned to the sha256 of what the person reviewed.
- **HTTP API** (`uvicorn --factory kappa.api:create_app`): read-only when `KAPPA_PUBLIC=1`; otherwise writes need `KAPPA_OPERATOR_TOKEN` and fail closed without it. Locked values never leave the server.

### Configuration

| Variable | Used by | Meaning |
|---|---|---|
| `KAPPA_SAM_BACKEND` | engine | `modal` (open-weight SAM 3.1 on our endpoint) or `api` (Meta Model API) |
| `KAPPA_LLM_BACKEND` | engine | `venice` (Muse Spark Open, Muse Wonder) or unset (Meta Model API) |
| `VENICE_API_KEY` | engine | key for the hosted Muse models |
| `MODEL_API_KEY` | engine | Meta Model API key, when that backend is used |
| `KAPPA_HOME` | engine | where runs, videos and the content-addressed store live (default `./kappa-data`) |
| `KAPPA_PUBLIC` | API | `1` serves recorded runs read-only |
| `KAPPA_OPERATOR_TOKEN` | API | bearer token for every write |
| `KAPPA_CORS` | API | allowed origins (default localhost:3000) |
| `NEXT_PUBLIC_KAPPA_API` | web | enables live mode against an API |

## Protocols as code

A protocol is a JSON document validated by a strict schema (`engine/src/kappa/protocol.py`): subject, arena, zones, entry rule, metrics, behaviours with operational definitions, integrity thresholds and smoothing. Its sha256 is printed on every result, so two labs that ran the same hash ran the same assay. Three starters live in `engine/src/kappa/library/`: open field, zebrafish novel tank (top view) and elevated plus maze.

## Honest limits

- **Rearing is below human agreement.** κ 0.538 against raters who agree at 0.80 to 0.82 (above). Kappa reports rearing counts, but read them as a first pass, not as a second rater.
- **Only rearing has a detector.** Grooming and jumping are not scored and nothing is claimed about them.
- **The rearing classifier is not a Meta model.** It is about 400 small decision trees over descriptors of the SAM 3.1 masks, roughly 1.5 MB, trained in seconds. SAM 3.1 does the seeing; the trees only weigh what it saw.
- **Most open-field sessions lock somewhere.** SAM 3.1 loses the mouse for more than 0.4 s at least once in most of the test videos. Kappa withholds the whole-session numbers that span such a gap (37 of 135 test-set values verified, 98 locked); the 5-minute bins without a gap stay verified.
- **Crowded scenes lock.** Five shoaling fish seen from above overlap often, and every overlap that motion cannot settle locks the numbers that depend on identity until a person resolves it. That is the intended behaviour, and it means social assays need more clicks.
- **Distances are 2D**, measured on the arena floor plane through a four-point homography. Lens distortion is not modelled, and fish depth is ignored.
- **The tracks themselves are public evidence.** Kappa withholds a locked number, but the per-frame positions behind it stay visible, with the doubtful stretches marked.
- **SAM 3.1 is not deterministic** run to run. Kappa never re-requests an output it has stored, and the manifest pins every output by hash.

## Data and licences

Kappa's code is MIT licensed. The rearing model in `engine/src/kappa/models/` is trained on the GPL-3.0 rater labels and is shared under GPL-3.0. The evaluation uses the open-field videos of Sturman et al. (2020, CC BY 4.0, Zenodo 3608658) and their raters' labels from `ETHZ-INS/DLCAnalyzer` (GPL-3.0); both are downloaded at run time and not redistributed here. The zebrafish example comes from 3D-ZeF (Pedersen et al., CVPR 2020).
