import Link from "next/link";

import { Mark } from "./Mark";
import styles from "./Nav.module.css";

const LINKS = [
  { href: "/runs", label: "Runs" },
  { href: "/validation", label: "Validation" },
  { href: "/docs", label: "Docs" },
];

export function Nav({ current, inset = false }: { current?: string; inset?: boolean }) {
  return (
    <header className={`${styles.nav} ${inset ? styles.inset : ""}`}>
      <Link className={styles.mark} href="/" aria-label="Kappa, home">
        <Mark />
        <span>Kappa</span>
      </Link>
      <nav aria-label="Primary">
        <ul className={styles.links}>
          {LINKS.map((l) => (
            <li key={l.href}>
              <Link href={l.href} aria-current={current === l.href ? "page" : undefined}>
                {l.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}
