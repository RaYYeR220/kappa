import type { ValidationState } from "@/lib/validation";

import styles from "./KappaScale.module.css";

// 0.4: Kappa's own values (0.53-0.55) sit inside the axis instead of being clamped to its edge
const MIN = 0.4;
const MAX = 1;
const pos = (k: number) => `${(Math.min(MAX, Math.max(MIN, k)) - MIN) / (MAX - MIN) * 100}%`;
const TICKS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

/** Frame-level κ for rearing on one axis: the raters against each other, and Kappa against each rater. */
export function KappaScale({ validation, human }: { validation: ValidationState; human: number[] }) {
  const published = validation.state === "published" ? validation.summary : null;
  const humans = published ? published.primary.human_vs_human.map((h) => h.kappa) : human;
  const kappas = published ? published.primary.kappa_vs_raters : [];
  const threshold = published ? published.primary.threshold : Math.min(...human);

  return (
    <figure className={styles.scale}>
      <figcaption className={styles.caption}>
        Cohen&rsquo;s κ for rearing, frame by frame
        {published ? `, ${published.test_videos.length} test videos` : ", raters on all 20 videos"}
      </figcaption>
      <div className={styles.rows}>
        <div className={styles.row}>
          <span className={styles.label}>Rater vs rater</span>
          <div className={styles.track}>
            <span className={styles.threshold} style={{ left: pos(threshold) }} aria-hidden="true" />
            {humans.map((k, i) => (
              <span key={i} className={styles.dot} data-who="human" style={{ left: pos(k) }} title={`κ ${k.toFixed(3)}`}>
                <span className="sr-only">κ {k.toFixed(3)}</span>
              </span>
            ))}
            <span className={styles.values} style={{ left: pos(Math.max(...humans)) }}>
              {humans.map((k) => k.toFixed(3)).join("  ")}
            </span>
          </div>
        </div>
        <div className={styles.row}>
          <span className={styles.label}>Kappa vs rater</span>
          <div className={styles.track}>
            <span className={styles.threshold} style={{ left: pos(threshold) }} aria-hidden="true" />
            {published ? (
              <>
                {kappas.map((k) => (
                  <span key={k.rater} className={styles.dot} data-who="kappa" style={{ left: pos(k.kappa) }} title={`Rater ${k.rater}: κ ${k.kappa.toFixed(3)}`}>
                    <span className="sr-only">
                      Rater {k.rater}: κ {k.kappa.toFixed(3)}
                    </span>
                  </span>
                ))}
                <span className={styles.values} style={{ left: pos(Math.max(...kappas.map((k) => k.kappa))) }}>
                  {kappas.map((k) => k.kappa.toFixed(3)).join("  ")}
                </span>
              </>
            ) : (
              <span className={styles.pending}>
                <span>Pending: published after the pre-registered run</span>
              </span>
            )}
          </div>
        </div>
      </div>
      <div className={styles.axis} aria-hidden="true">
        <span className={styles.label} />
        <div className={styles.ticks}>
          {TICKS.map((t) => (
            <span key={t} style={{ left: pos(t) }}>
              {t.toFixed(1)}
            </span>
          ))}
        </div>
      </div>
      <p className={styles.legend}>
        <span className={styles.rule} aria-hidden="true" />
        {published
          ? `Bar to clear: the lowest rater-vs-rater κ on the same videos, ${threshold.toFixed(3)}. Kappa's mean: ${published.primary.mean_kappa.toFixed(3)}, ${published.primary.passed ? "cleared" : "not cleared"}.`
          : `Bar to clear: the lowest rater-vs-rater κ on the same 15 test videos. On all 20 videos it is ${Math.min(...human).toFixed(3)}.`}
      </p>
    </figure>
  );
}
