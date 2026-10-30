import { readFile } from "node:fs/promises";
import path from "node:path";

import type { Metadata } from "next";

import { Nav } from "@/components/site/Nav";
import { KappaScale } from "@/components/ui/KappaScale";
import { Markdown, sections } from "@/lib/markdown";
import { loadValidation, PREREGISTRATION } from "@/lib/validation";

import styles from "./validation.module.css";

export const metadata: Metadata = {
  title: "Validation",
  description: "Kappa's accuracy test, pre-registered before any test video was analysed: rearing, frame by frame, against three human raters.",
};

function when(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "Europe/Berlin",
    timeZoneName: "short",
  });
}

export default async function ValidationPage() {
  const [validation, md] = await Promise.all([
    loadValidation(),
    readFile(path.join(process.cwd(), "content", "PREREGISTRATION.md"), "utf8").catch(() => ""),
  ]);
  const doc = sections(md);
  const published = validation.state === "published" ? validation.summary : null;

  return (
    <>
      <Nav current="/validation" />
      <main id="main" className={`wrap ${styles.page}`}>
        <header className={styles.head}>
          <h1>Validated against people.</h1>
          <div>
            <p className={styles.lede}>
              Kappa&rsquo;s accuracy is tested the way a methods reviewer would ask for: on videos it was never tuned on,
              against trained human raters, with the analysis and the bar to clear fixed in advance.
            </p>
            <p className={styles.commit}>
              Pre-registered in commit <code>{PREREGISTRATION.commit.slice(0, 7)}</code> on {when(PREREGISTRATION.date)},
              before any test video was analysed.
            </p>
          </div>
        </header>

        <section className={styles.results} aria-labelledby="results-title">
          <div className={styles.sectionHead}>
            <h2 id="results-title">Results</h2>
            {published ? (
              <p>
                {published.test_videos.length} test videos, analysed once on {when(published.generated_at)}.
                {published.config_commit ? ` Configuration frozen in commit ${published.config_commit.slice(0, 7)}.` : ""}
              </p>
            ) : (
              <p>
                The test videos have not been analysed yet. They will be analysed once, with the configuration frozen in a
                commit made before the first test run, and the results will be published here whatever they are.
              </p>
            )}
          </div>

          <div className={styles.resultBody}>
            {!published && (
              <div className={styles.pending} role="status">
                <span>Pending: results are published after the pre-registered run.</span>
              </div>
            )}
            {validation.state === "invalid" && (
              <p className={styles.problem} role="alert">
                A results file exists but could not be read ({validation.problem}), so nothing from it is shown.
              </p>
            )}
            <div className={styles.scale}>
              <KappaScale validation={validation} human={PREREGISTRATION.humanKappa} />
            </div>

            {published && (
              <>
                <table className={styles.table}>
                  <caption>Primary endpoint: frame-level κ for rearing, pooled over the test videos</caption>
                  <thead>
                    <tr>
                      <th scope="col">Comparison</th>
                      <th scope="col">κ</th>
                      <th scope="col">95% interval</th>
                    </tr>
                  </thead>
                  <tbody>
                    {published.primary.kappa_vs_raters.map((r) => (
                      <tr key={r.rater}>
                        <th scope="row">Kappa vs rater {r.rater}</th>
                        <td className="mono">{r.kappa.toFixed(3)}</td>
                        <td className="mono">{r.ci95 ? `${r.ci95[0].toFixed(3)} to ${r.ci95[1].toFixed(3)}` : ""}</td>
                      </tr>
                    ))}
                    {published.primary.human_vs_human.map((r) => (
                      <tr key={r.raters.join("-")}>
                        <th scope="row">
                          Rater {r.raters[0]} vs rater {r.raters[1]}
                        </th>
                        <td className="mono">{r.kappa.toFixed(3)}</td>
                        <td />
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr>
                      <th scope="row">Kappa&rsquo;s mean against the bar</th>
                      <td className="mono">
                        {published.primary.mean_kappa.toFixed(3)} vs {published.primary.threshold.toFixed(3)}
                      </td>
                      <td>{published.primary.passed ? "Cleared" : "Not cleared"}</td>
                    </tr>
                  </tfoot>
                </table>

                {published.secondary && published.secondary.length > 0 && (
                  <table className={styles.table}>
                    <caption>Secondary endpoints, reported without a threshold</caption>
                    <thead>
                      <tr>
                        <th scope="col">Endpoint</th>
                        <th scope="col">Kappa</th>
                        <th scope="col">Raters</th>
                      </tr>
                    </thead>
                    <tbody>
                      {published.secondary.map((r) => (
                        <tr key={r.label}>
                          <th scope="row">
                            {r.label}
                            {r.note && <span className={styles.note}>{r.note}</span>}
                          </th>
                          <td className="mono">{r.kappa}</td>
                          <td className="mono">{r.humans ?? ""}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}

                {published.negative_controls && published.negative_controls.length > 0 && (
                  <table className={styles.table}>
                    <caption>Negative controls</caption>
                    <thead>
                      <tr>
                        <th scope="col">Control</th>
                        <th scope="col">Expected</th>
                        <th scope="col">Observed</th>
                      </tr>
                    </thead>
                    <tbody>
                      {published.negative_controls.map((c) => (
                        <tr key={c.id}>
                          <th scope="row">{c.description}</th>
                          <td>{c.expected}</td>
                          <td>
                            {c.observed} {c.passed ? "" : "(failed)"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </>
            )}
          </div>
        </section>

        <section id="preregistration" className={styles.prereg} aria-labelledby="prereg-title">
          <div className={styles.sectionHead}>
            <h2 id="prereg-title">The pre-registration</h2>
            {doc.lede && (
              <div className={styles.prose}>
                <Markdown source={doc.lede} />
              </div>
            )}
          </div>
          <div className={styles.sections}>
            {doc.sections.map((s) => (
              <section key={s.title} className={styles.mdSection} data-key={s.title.toLowerCase().split(" ")[0]}>
                <h3>{s.title}</h3>
                <div className={styles.prose}>
                  <Markdown source={s.body} mark="**Success criterion" />
                </div>
              </section>
            ))}
          </div>
        </section>
      </main>
    </>
  );
}
