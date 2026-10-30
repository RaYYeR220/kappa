import Link from "next/link";

import { Mark } from "./Mark";
import styles from "./Footer.module.css";

export function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={`wrap ${styles.grid}`}>
        <div className={styles.brand}>
          <Link href="/" className={styles.mark} aria-label="Kappa, home">
            <Mark size={20} />
            <span>Kappa</span>
          </Link>
          <p>
            A measuring instrument made from any camera. Tracking by SAM 3.1, protocols drafted by Muse Spark, review by
            Muse Wonder, all on the Meta Model API.
          </p>
        </div>
        <nav aria-label="Footer" className={styles.cols}>
          <div>
            <h2>Product</h2>
            <ul>
              <li>
                <Link href="/runs">Recorded runs</Link>
              </li>
              <li>
                <Link href="/#how">How it works</Link>
              </li>
              <li>
                <Link href="/#protocols">Protocols</Link>
              </li>
            </ul>
          </div>
          <div>
            <h2>Evidence</h2>
            <ul>
              <li>
                <Link href="/validation">Validation</Link>
              </li>
              <li>
                <Link href="/validation#preregistration">Pre-registration</Link>
              </li>
            </ul>
          </div>
          <div>
            <h2>Build</h2>
            <ul>
              <li>
                <Link href="/docs">Docs</Link>
              </li>
              <li>
                <Link href="/docs#mcp">MCP server</Link>
              </li>
              <li>
                <Link href="/docs#bundle">Bundle format</Link>
              </li>
            </ul>
          </div>
        </nav>
      </div>
      <div className={`wrap ${styles.base}`}>
        <span>MIT licence</span>
        <span>Meta Global AI Developer Hackathon, 2026</span>
      </div>
    </footer>
  );
}
