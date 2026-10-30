import { CopyButton } from "./CopyButton";
import styles from "./Code.module.css";

/** A code block on the ink well, with a copy button. */
export function Code({ code, label, language }: { code: string; label: string; language?: string }) {
  return (
    <figure className={styles.code}>
      <figcaption className={styles.bar}>
        <span>{label}</span>
        <CopyButton text={code} tone="dark" />
      </figcaption>
      <pre tabIndex={0} aria-label={label} data-language={language}>
        <code>{code}</code>
      </pre>
    </figure>
  );
}
