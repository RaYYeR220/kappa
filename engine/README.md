# kappa engine

The Python package behind Kappa: the pipeline from a video and a confirmed protocol to numbers that are verified, locked or refused, plus the three ways to drive it (CLI, MCP server, HTTP API). Everything a person or an agent can do goes through the same `Runner`, so the rules are the same everywhere: nothing is measured under a protocol nobody confirmed, and nothing is reported from a track that failed a check.

```bash
uv sync --all-extras
uv run pytest                      # the full suite runs offline, on synthetic tracks and recorded model outputs
```

## Backends

| Step | Model | Backend switch |
|---|---|---|
| Segmentation and tracking | SAM 3.1 | `KAPPA_SAM_BACKEND=modal` (open weights on the endpoint in `../tracker`) or `api` (Meta Model API, `MODEL_API_KEY`) |
| Protocol draft, arena corners, Reviewer 2 | Muse Spark Open, Muse Wonder | `KAPPA_LLM_BACKEND=venice` (`VENICE_API_KEY`) or the Model API |
| Rearing | `models/rearing_topdown.pkl`, gradient-boosted trees over SAM 3.1 silhouettes | none: local, CPU, sha256-checked before it is loaded |

Every model output is stored once, content-addressed (`store.py`), and never requested again; the manifest pins each one by hash.

## Command line

One subcommand per step. Exit status 2 means Kappa refused the step and printed why.

```
kappa new <video> [--id ID] [--start S] [--end S]     start a run from a video; prints the run id
kappa draft <run> "<question>"                        draft a protocol from an English question
kappa confirm <run> --sha256 <hash> --by "<name>"     a person accepts the protocol it read (pinned to its hash)
kappa track <run>                                     track every subject with SAM 3.1
kappa score <run> <behaviour>                         score one behaviour of the protocol
kappa repair <run> --frame F --by "<name>" --subject 1:x,y ...   settle who is who on one frame, re-track from there
kappa results <run>                                   every result with its status; locked values are withheld
kappa manifest <run> [--review]                       write the run's manifest (and Reviewer 2)
kappa runs | batch | eval | train-rearing             list runs; one protocol over many videos; the pre-registered
                                                      evaluation; train the rearing model on the tuning videos only
```

## MCP server

`kappa-mcp` (FastMCP) exposes nine tools: `create_run`, `draft_protocol`, `submit_for_confirmation`, `track`, `get_results`, `get_manifest`, `list_runs`, `propose_repair`, `score_behaviour`. An agent can propose a protocol or a repair, never apply one: both return `"status": "awaiting_person"` and the command a person runs. Locked numbers reach the agent as `null`, with the frame a person needs to look at.

## HTTP API

`uvicorn --factory kappa.api:create_app`. Reads are open; every request that changes a run needs `Authorization: Bearer $KAPPA_OPERATOR_TOKEN`, and writes are disabled when the token is not set. Long steps (track, score, repair, review) run as jobs with progress at `/runs/{id}/jobs/{name}`.

## Layout

```
model.py, protocol.py      the data model and the assay protocol schema
runner.py                  one run's lifecycle: new, draft, confirm, track, score, repair, manifest
pipeline.py, integrity.py  detections -> subject tracks -> integrity events (gaps, jumps, low confidence, contacts)
metrics/                   distance, zones, immobility, behaviour counts; gating.py decides verified / locked / refused
shape.py, behaviour_model.py   SAM 3.1 silhouette descriptors and the rearing classifier
sam/                       SAM 3.1 clients (Modal endpoint, Model API) and a fast parser for its output text
agents/                    protocol drafting, Reviewer 2
evidence.py, export.py     the methods paragraph, the manifest, the replay bundles the web app reads
eval/                      the pre-registered evaluation (Sturman et al. 2020) and its statistics
cli.py, mcp_server.py, api.py
```
