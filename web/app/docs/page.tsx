import type { Metadata } from "next";

import { Nav } from "@/components/site/Nav";
import { Code } from "@/components/ui/Code";

import styles from "./docs.module.css";

export const metadata: Metadata = {
  title: "Docs",
  description: "Quick start for the Kappa CLI, MCP server and HTTP API, the run bundle format and the protocol schema.",
};

const TOC = [
  { id: "start", label: "Quick start" },
  { id: "cli", label: "Command line" },
  { id: "mcp", label: "MCP server" },
  { id: "api", label: "HTTP API" },
  { id: "bundle", label: "Bundle format" },
  { id: "protocol", label: "Protocol schema" },
];

const INSTALL = `cd kappa/engine                   # in a checkout of the Kappa repository
uv sync --extra api --extra mcp
export KAPPA_SAM_BACKEND=modal    # SAM 3.1 open weights on the endpoint in tracker/ (modal deploy tracker/app.py)
export KAPPA_LLM_BACKEND=venice VENICE_API_KEY=...   # Muse Spark Open and Muse Wonder
# or run everything on Meta Model API instead: export MODEL_API_KEY=...
export KAPPA_HOME=~/kappa-data    # runs and the content-addressed store (default ./kappa-data)`;

const FIRST_RUN = `uv run kappa new trial.mp4 --id trial-07
uv run kappa draft trial-07 "How far does the mouse travel in each 5-minute bin, and how long is it in the centre?"
# read protocol.draft.json in the run folder, edit it, mark the arena corners
uv run kappa confirm trial-07 --by "A. Researcher"
uv run kappa track trial-07
uv run kappa results trial-07
uv run kappa manifest trial-07 --review`;

const MCP = `{
  "mcpServers": {
    "kappa": {
      "command": "uv",
      "args": ["--directory", "/path/to/kappa/engine", "run", "kappa-mcp"],
      "env": { "MODEL_API_KEY": "...", "KAPPA_HOME": "/path/to/kappa-data" }
    }
  }
}`;

const API = `export KAPPA_OPERATOR_TOKEN=$(openssl rand -hex 24)
export KAPPA_CORS=http://localhost:3000
uv run uvicorn --factory kappa.api:create_app --port 8000

# the web workspace, pointed at it
NEXT_PUBLIC_KAPPA_API=http://localhost:8000 pnpm dev`;

const EXPORT = `from pathlib import Path
from kappa.export import export_run
from kappa.runner import Runner

export_run(Runner(), "trial-07", Path("../web/public/runs"))
# then list it: web/public/runs/index.json  ->  {"runs": [{"id": "trial-07"}]}`;

const EXAMPLE = `{
  "schema_version": "kappa.protocol/1",
  "title": "Open field test, mouse, 45 x 45 cm arena",
  "question": "How far does the mouse travel, and how long is it in the centre?",
  "subject": { "concept": "mouse", "expected_count": 1, "label": "mouse" },
  "arena": { "shape": "rect", "width_cm": 45, "height_cm": 45, "corners_px": null },
  "zones": [{ "id": "center", "kind": "center_fraction", "fraction": 0.25 }],
  "entry_rule": { "mode": "body_fraction", "body_fraction": 0.7, "min_dwell_s": 0 },
  "metrics": [
    { "id": "distance", "kind": "distance", "bin_s": 300 },
    { "id": "center_time", "kind": "time_in_zone", "zone": "center" }
  ],
  "refused": [{ "request": "heart rate", "reason": "not visible on video" }]
}`;

export default function DocsPage() {
  return (
    <>
      <Nav current="/docs" />
      <main id="main" className={`wrap ${styles.page}`}>
        <header className={styles.head}>
          <h1>Docs</h1>
          <p>
            Kappa is a Python engine with three ways in: a command line, an MCP server for agents, and an HTTP API for this
            workspace. All three apply the same rules: nothing runs on an unconfirmed protocol, and a locked number never
            leaves the engine.
          </p>
        </header>

        <div className={styles.layout}>
          <nav aria-label="On this page" className={styles.toc}>
            <ul>
              {TOC.map((t) => (
                <li key={t.id}>
                  <a href={`#${t.id}`}>{t.label}</a>
                </li>
              ))}
            </ul>
          </nav>

          <div className={styles.content}>
            <section id="start" aria-labelledby="start-title">
              <h2 id="start-title">Quick start</h2>
              <p>The engine needs Python 3.11 or later and ffmpeg on the path. For the model steps it needs either the SAM 3.1 endpoint (open weights, deployed from tracker/ to Modal) and a key for hosted Muse Spark Open and Muse Wonder, or a Meta Model API key.</p>
              <Code code={INSTALL} label="install" language="shell" />
              <p>A run is one video, one confirmed protocol, and everything computed from them.</p>
              <Code code={FIRST_RUN} label="a first run" language="shell" />
            </section>

            <section id="cli" aria-labelledby="cli-title">
              <h2 id="cli-title">Command line</h2>
              <p>
                One subcommand per step. Exit status is 0 on success, 2 when Kappa refuses a step (with the reason on stderr),
                and 1 for anything else.
              </p>
              <table className={styles.table}>
                <tbody>
                  {[
                    ["kappa new VIDEO", "Normalise the video to a constant frame rate, hash it, start a run. --start, --end, --fps, --id."],
                    ["kappa draft RUN QUESTION", "Muse Spark drafts a protocol from the question and keyframes into protocol.draft.json."],
                    ["kappa confirm RUN --by NAME", "A person accepts the draft (or --protocol FILE). Only a confirmed protocol runs."],
                    ["kappa track RUN", "SAM 3.1 segments and tracks every subject; the raw output is stored by its sha256."],
                    ["kappa score RUN BEHAVIOUR", "Candidate stretches from the tracks, each judged on its clip by Muse Spark."],
                    ["kappa repair RUN --frame N --by NAME --subject ID:X,Y …", "Settle who is who on frame N; SAM 3.1 re-tracks forward and every check runs again."],
                    ["kappa results RUN [--json]", "Every number with its status. Locked values are withheld."],
                    ["kappa manifest RUN [--review]", "The evidence manifest and methods paragraph; --review adds Reviewer 2 by Muse Wonder."],
                    ["kappa batch PROTOCOL VIDEO… --by NAME", "Run one confirmed protocol on many videos."],
                  ].map(([cmd, what]) => (
                    <tr key={cmd}>
                      <th scope="row">
                        <code>{cmd}</code>
                      </th>
                      <td>{what}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>

            <section id="mcp" aria-labelledby="mcp-title">
              <h2 id="mcp-title">MCP server</h2>
              <p>
                <code>kappa-mcp</code> serves the run lifecycle over stdio. Its tools cover creating a run, drafting a
                protocol, tracking, scoring behaviours, and reading results and the manifest; the protocol library and each
                run&rsquo;s manifest are also resources.
              </p>
              <Code code={MCP} label="mcp.json" language="json" />
              <ul className={styles.rules}>
                <li>An agent can draft a protocol. A person confirms it, by name.</li>
                <li>Results reach the agent gated: a locked number arrives as null, with the events that lock it and the frame where a person can settle them.</li>
                <li>An agent is told never to estimate a locked or refused value.</li>
              </ul>
            </section>

            <section id="api" aria-labelledby="api-title">
              <h2 id="api-title">HTTP API</h2>
              <p>
                The API behind the live workspace. Reads are public. Writes call paid models, so they need{" "}
                <code>Authorization: Bearer $KAPPA_OPERATOR_TOKEN</code> and fail closed when no token is configured.{" "}
                <code>KAPPA_PUBLIC=1</code> serves recorded runs read-only.
              </p>
              <Code code={API} label="serve" language="shell" />
              <table className={styles.table}>
                <tbody>
                  {[
                    ["GET /runs, /runs/{id}", "Runs and their state."],
                    ["GET /runs/{id}/video, /sam, /tracks", "The normalised video, SAM 3.1’s raw output text, per-frame subject tracks."],
                    ["GET /runs/{id}/results, /manifest", "Gated results and integrity events; the redacted manifest."],
                    ["POST /runs/{id}/track", "Start tracking. The job streams SAM 3.1 output as it arrives."],
                    ["POST /runs/{id}/repair", "{frame, by, anchors: [{subject, x, y}]}: settle who is who and re-track."],
                    ["POST /runs/{id}/review", "Ask Reviewer 2 for a read of the run."],
                    ["GET /runs/{id}/jobs/{name}", "Server-sent events for a job; operator token required."],
                  ].map(([route, what]) => (
                    <tr key={route}>
                      <th scope="row">
                        <code>{route}</code>
                      </th>
                      <td>{what}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className={styles.small}>
                Job streams carry the Authorization header, so the workspace reads them with fetch rather than EventSource.
              </p>
            </section>

            <section id="bundle" aria-labelledby="bundle-title">
              <h2 id="bundle-title">Bundle format</h2>
              <p>
                A recorded run is a static folder this site replays with no server and no credentials. It carries exactly what
                a public reader may see: locked values are not in it.
              </p>
              <table className={styles.table}>
                <tbody>
                  {[
                    ["run.json", "Run id, question, video facts and sha256, the confirmed protocol and its sha256, who confirmed it, repairs, arena corners."],
                    ["results.json", "analysed frame range, every result (status, value when verified, evidence spans, reasons, fix frame) and every integrity event."],
                    ["tracks.json", "fps, frame count, and per subject: centroid in pixels and cm, area, interpolated flags, and source: the SAM object id behind the subject on each frame (-1 when absent)."],
                    ["sam.txt", "SAM 3.1’s raw output text, one line per frame. The browser parses it with @meta-sam/parser."],
                    ["sam-repair-N.txt", "The re-track from repair frame N, when there is one. Its object ids are subject numbers."],
                    ["video.mp4", "The normalised video every frame number refers to."],
                    ["manifest.json", "The redacted evidence manifest: models, artifacts, results, review, methods paragraph."],
                    ["verdicts.json", "Behaviour verdicts with Muse Spark’s rationale, by behaviour id."],
                  ].map(([file, what]) => (
                    <tr key={file}>
                      <th scope="row">
                        <code>{file}</code>
                      </th>
                      <td>{what}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p>Export a run from the engine into the site, then list it in the index:</p>
              <Code code={EXPORT} label="export a bundle" language="python" />
            </section>

            <section id="protocol" aria-labelledby="protocol-title">
              <h2 id="protocol-title">Protocol schema</h2>
              <p>
                <code>kappa.protocol/1</code>. Strict: unknown fields are rejected. The sha256 of its canonical JSON names the
                assay behind every number.
              </p>
              <table className={styles.table}>
                <tbody>
                  {[
                    ["subject", "concept (the text prompt for SAM 3.1), expected_count (1–16), label."],
                    ["arena", "shape rect or circle, width_cm, height_cm, corners_px: four floor corners in image pixels, marked before confirming."],
                    ["zones", "center_fraction (fraction), periphery_band (band_cm), or rect_cm ([x0, y0, x1, y1] in arena cm)."],
                    ["entry_rule", "centroid, or body_fraction of the body inside; min_dwell_s."],
                    ["metrics", "distance, mean_speed, time_in_zone, fraction_time_in_zone, zone_entries, latency_to_zone, immobility_time, immobility_bouts, mean_inter_subject_distance, behaviour_count, behaviour_duration. Optional bin_s."],
                    ["behaviours", "id, label, operational definition, detector, and where (anywhere, near the wall, away from it) with near_wall_cm."],
                    ["integrity", "max_speed_cm_s, max_interpolated_gap_s, min_confidence, low_confidence_min_s, contact_distance (body lengths), identity_margin."],
                    ["smoothing, sampling", "median, savgol or none with window_s; resampled fps, start_s, end_s."],
                    ["refused", "Requests this camera cannot answer, each with its reason."],
                  ].map(([field, what]) => (
                    <tr key={field}>
                      <th scope="row">
                        <code>{field}</code>
                      </th>
                      <td>{what}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <Code code={EXAMPLE} label="protocol.json" language="json" />
            </section>
          </div>
        </div>
      </main>
    </>
  );
}
