"use client";

import { useId, useRef, useState } from "react";

import { Code } from "@/components/ui/Code";

import styles from "./ProtocolTabs.module.css";

export interface LibraryProtocol {
  key: string;
  name: string;
  title: string;
  question: string;
  metrics: { id: string; name: string }[];
  refusals: string[];
  json: string;
}

export function ProtocolTabs({ protocols }: { protocols: LibraryProtocol[] }) {
  const [active, setActive] = useState(0);
  const base = useId();
  const tabs = useRef<(HTMLButtonElement | null)[]>([]);
  const current = protocols[active];

  function onKey(e: React.KeyboardEvent, i: number) {
    let next = i;
    if (e.key === "ArrowRight") next = (i + 1) % protocols.length;
    else if (e.key === "ArrowLeft") next = (i - 1 + protocols.length) % protocols.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = protocols.length - 1;
    else return;
    e.preventDefault();
    setActive(next);
    tabs.current[next]?.focus();
  }

  if (!current) return null;

  return (
    <div className={styles.tabs}>
      <div role="tablist" aria-label="Protocol library" className={styles.list}>
        {protocols.map((p, i) => (
          <button
            key={p.key}
            ref={(el) => {
              tabs.current[i] = el;
            }}
            role="tab"
            type="button"
            id={`${base}-tab-${i}`}
            aria-selected={i === active}
            aria-controls={`${base}-panel`}
            tabIndex={i === active ? 0 : -1}
            className={styles.tab}
            onClick={() => setActive(i)}
            onKeyDown={(e) => onKey(e, i)}
          >
            {p.name}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`${base}-panel`} aria-labelledby={`${base}-tab-${active}`} className={styles.panel}>
        <div className={styles.summary}>
          <h3>{current.title}</h3>
          <blockquote>{current.question}</blockquote>
          <h4>Measures</h4>
          <ul className={styles.metrics}>
            {current.metrics.map((m) => (
              <li key={m.id}>
                <span>{m.name}</span>
                <code>{m.id}</code>
              </li>
            ))}
          </ul>
          {current.refusals.length > 0 && (
            <>
              <h4>Refuses</h4>
              <ul className={styles.refusals}>
                {current.refusals.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>
            </>
          )}
        </div>
        <div className={styles.json}>
          <Code code={current.json} label={`${current.key}.json`} language="json" />
        </div>
      </div>
    </div>
  );
}
