import Link from "next/link";

import { Hero } from "@/components/landing/Hero";
import { ProtocolTabs, type LibraryProtocol } from "@/components/landing/ProtocolTabs";
import { RefusalDemo } from "@/components/landing/RefusalDemo";
import { Code } from "@/components/ui/Code";
import { KappaScale } from "@/components/ui/KappaScale";
import { firstRunHref } from "@/lib/bundle";
import { metricName } from "@/lib/format";
import { loadProtocols } from "@/lib/protocols";
import { loadValidation, PREREGISTRATION } from "@/lib/validation";

import styles from "./page.module.css";

const MCP_CONFIG = `{
  "mcpServers": {
    "kappa": {
      "command": "uv",
      "args": ["--directory", "/path/to/kappa/engine", "run", "kappa-mcp"],
      "env": { "KAPPA_SAM_BACKEND": "modal", "KAPPA_LLM_BACKEND": "venice", "VENICE_API_KEY": "<key>" }
    }
  }
}`;

const CLI = `kappa new trial.mp4 --id trial-07
kappa draft trial-07 "How far does the mouse travel, and how long is it in the centre?"
kappa confirm trial-07 --by "A. Researcher"   # a person reads the draft first
kappa track trial-07                          # SAM 3.1, every frame
kappa results trial-07                        # verified, locked or refused
kappa repair trial-07 --frame 4680 --by "A. Researcher" --subject 1:412,233 --subject 2:470,251`;

export default async function Home() {
  const [cta, validation, library] = await Promise.all([firstRunHref(), loadValidation(), loadProtocols()]);

  const protocols: LibraryProtocol[] = library.map((p) => ({
    key: p.key,
    name: p.name,
    title: p.body.title,
    question: p.body.question,
    metrics: p.body.metrics.map((m) => ({ id: m.id, name: metricName(m, p.body, m.id) })),
    refusals: p.body.refused.map((r) => `${r.request}: ${r.reason}`),
    json: JSON.stringify(p.body, null, 2),
  }));

  return (
    <>
      <Hero ctaHref={cta} />

      <section className={styles.section} id="how" aria-labelledby="how-title">
        <div className="wrap">
          <header className={styles.head}>
            <h2 id="how-title">From a question to a number you can cite.</h2>
            <p>
              Three steps, each with a person or a check in the loop. Nothing is computed from a protocol nobody
              confirmed, and nothing is reported from a track that failed a check.
            </p>
          </header>
          <ol className={styles.steps}>
            <li>
              <span className={styles.stepNo}>1</span>
              <h3>Ask</h3>
              <p>
                Describe the measurement in plain English. Muse Spark Open drafts the assay protocol from your question
                and a few frames, and finds the arena floor: zones, metrics, thresholds, and what this camera cannot
                answer. A person confirms it, and its sha256 fixes the assay.
              </p>
              <div className={styles.visual} aria-hidden="true">
                <div className={styles.askCard}>
                  <p className={styles.askQ}>&ldquo;How far does it walk, and how long in the centre?&rdquo;</p>
                  <ul>
                    <li>
                      <span>distance</span>
                      <em>per 5 min</em>
                    </li>
                    <li>
                      <span>centre time</span>
                      <em>inner 25%</em>
                    </li>
                    <li className={styles.askRefused}>
                      <span>heart rate</span>
                      <em>refused</em>
                    </li>
                  </ul>
                  <p className={styles.askSha}>
                    confirmed by A. Researcher <code>598d6a1c</code>
                  </p>
                </div>
              </div>
            </li>
            <li>
              <span className={styles.stepNo}>2</span>
              <h3>Track</h3>
              <p>
                SAM 3.1, running from Meta's open weights on Kappa's own GPU endpoint, segments every animal from a text
                prompt and keeps a stable id on every frame. Kappa checks each track for gaps, impossible jumps, low
                confidence, and contacts where two animals could trade identities.
              </p>
              <div className={styles.visual} aria-hidden="true">
                <svg viewBox="0 0 320 180" className={styles.trackArt}>
                  <rect width="320" height="180" rx="14" fill="#121210" />
                  <rect x="12" y="12" width="296" height="156" rx="6" fill="#dcd9d0" />
                  <path d="M40 140c30-10 50-60 90-70s60 30 100 10" fill="none" stroke="#bc4a3f" strokeWidth="1.5" strokeDasharray="2 3" opacity=".7" />
                  <ellipse cx="236" cy="78" rx="21" ry="10" transform="rotate(-20 236 78)" fill="#f9786a" fillOpacity=".55" stroke="#bc4a3f" strokeWidth="2" />
                  <path d="M60 52c20 4 40 22 50 46" fill="none" stroke="#1577c8" strokeWidth="1.5" strokeDasharray="2 3" opacity=".7" />
                  <ellipse cx="112" cy="104" rx="20" ry="10" transform="rotate(64 112 104)" fill="#52a9fe" fillOpacity=".55" stroke="#1577c8" strokeWidth="2" />
                  <ellipse cx="178" cy="136" rx="19" ry="9.5" transform="rotate(10 178 136)" fill="#87b73a" fillOpacity=".55" stroke="#5b8304" strokeWidth="2" />
                  <g fontFamily="var(--mono)" fontSize="11" fontWeight="500" fill="#0e0e0c">
                    <rect x="220" y="40" width="26" height="17" rx="4" fill="#f9786a" />
                    <text x="233" y="52" textAnchor="middle">M1</text>
                    <rect x="124" y="70" width="26" height="17" rx="4" fill="#52a9fe" />
                    <text x="137" y="82" textAnchor="middle">M2</text>
                    <rect x="194" y="104" width="26" height="17" rx="4" fill="#87b73a" />
                    <text x="207" y="116" textAnchor="middle">M3</text>
                  </g>
                </svg>
              </div>
            </li>
            <li>
              <span className={styles.stepNo}>3</span>
              <h3>Prove, or refuse</h3>
              <p>
                A deterministic engine computes the numbers, and rearing is read from the SAM 3.1 silhouettes. Each
                number is verified with the frames it came from, locked when the track under it failed a check, or
                refused with the reason. Muse Wonder then reads the whole run as Reviewer 2.
              </p>
              <div className={styles.visual} aria-hidden="true">
                <ul className={styles.cells}>
                  <li data-status="verified">
                    <span>Verified</span>
                    <b>
                      22.70 <small>cm</small>
                    </b>
                  </li>
                  <li data-status="locked">
                    <span>Locked</span>
                    <b>ID uncertain</b>
                  </li>
                  <li data-status="refused">
                    <span>Refused</span>
                    <b>needs a side view</b>
                  </li>
                </ul>
              </div>
            </li>
          </ol>
        </div>
      </section>

      <section className={`${styles.section} ${styles.dots}`} id="refusal" aria-labelledby="refusal-title">
        <div className={`wrap ${styles.split}`}>
          <div className={styles.splitText}>
            <h2 id="refusal-title">A number it can&rsquo;t defend stays locked.</h2>
            <p>
              When two animals touch, a tracker can carry on with their identities swapped and every number after that
              belongs to the wrong animal. Kappa checks the motion on both sides of every contact. When the motion
              can&rsquo;t tell them apart, every number that depends on who is who is withheld.
            </p>
            <p>
              A person settles it on one frame by marking each animal. SAM 3.1 re-tracks forward from that frame, every
              integrity check runs again, and the repair is recorded with the name of the person who made it.
            </p>
            <p className={styles.note}>The panel is an illustration of that flow; recorded runs show the real one.</p>
          </div>
          <RefusalDemo />
        </div>
      </section>

      <section className={styles.section} id="validated" aria-labelledby="validated-title">
        <div className={`wrap ${styles.split}`}>
          <div className={styles.splitText}>
            <h2 id="validated-title">Graded against three human raters.</h2>
            <p>
              The accuracy test was pre-registered before any test video was analysed: rearing in 15 open-field videos,
              scored frame by frame and compared with each of three trained raters by Cohen&rsquo;s κ. The bar is the
              agreement between the raters themselves.
            </p>
            <p>
              <Link className={styles.link} href="/validation">
                Read the pre-registration and results
              </Link>
            </p>
          </div>
          <div className={styles.kappaPanel}>
            <KappaScale validation={validation} human={PREREGISTRATION.humanKappa} />
            <p className={styles.commit}>
              Pre-registered in commit <code>{PREREGISTRATION.commit.slice(0, 7)}</code>, before any test video was analysed.
            </p>
          </div>
        </div>
      </section>

      <section className={styles.section} id="protocols" aria-labelledby="protocols-title">
        <div className="wrap">
          <header className={styles.head}>
            <h2 id="protocols-title">Protocols are files you can review.</h2>
            <p>
              A protocol is plain JSON: subject, arena, zones, metrics, integrity thresholds, and the requests it
              refuses. Muse Spark Open drafts new ones; a person confirms them; every number names the sha256 of the exact
              assay it was computed under. These three ship with Kappa.
            </p>
          </header>
          <ProtocolTabs protocols={protocols} />
        </div>
      </section>

      <section className={`${styles.section} ${styles.last}`} id="agents" aria-labelledby="agents-title">
        <div className="wrap">
          <header className={styles.head}>
            <h2 id="agents-title">Agents can run it. People sign it.</h2>
            <p>
              The MCP server and the command line expose the whole run under the same rules as the workspace. An agent
              can draft a protocol, but a person confirms it. Locked numbers reach the agent as null, with the frame a
              person needs to fix.
            </p>
          </header>
          <div className={styles.agentGrid}>
            <div>
              <h3 className={styles.codeTitle}>MCP server</h3>
              <p className={styles.codeText}>Add Kappa to any MCP client. Its tools cover the run, from create_run to get_manifest.</p>
              <Code code={MCP_CONFIG} label="mcp.json" language="json" />
            </div>
            <div>
              <h3 className={styles.codeTitle}>Command line</h3>
              <p className={styles.codeText}>One subcommand per step. Exit status 2 means Kappa refused the step and said why.</p>
              <Code code={CLI} label="terminal" language="shell" />
            </div>
          </div>
          <p className={styles.more}>
            <Link className={styles.link} href="/docs">
              Read the docs: CLI, MCP, HTTP API and the bundle format
            </Link>
          </p>
        </div>
      </section>
    </>
  );
}
