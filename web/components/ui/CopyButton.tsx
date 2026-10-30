"use client";

import { useRef, useState } from "react";

import styles from "./CopyButton.module.css";

export function CopyButton({
  text,
  label = "Copy",
  done = "Copied",
  tone = "light",
  className,
}: {
  text: string;
  label?: string;
  done?: string;
  tone?: "light" | "dark";
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number>(0);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    setCopied(true);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setCopied(false), 1800);
  }

  return (
    <button type="button" className={`${styles.button} ${styles[tone]} ${className ?? ""}`} onClick={copy}>
      <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
        {copied ? (
          <path d="M2.5 7.4 5.6 10.4 11.5 3.8" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        ) : (
          <g fill="none" stroke="currentColor" strokeWidth="1.3">
            <rect x="4.5" y="4.5" width="7.5" height="7.5" rx="1.8" />
            <path d="M9.5 2.6V2.5a1.5 1.5 0 0 0-1.5-1.5H3A1.5 1.5 0 0 0 1.5 2.5V8A1.5 1.5 0 0 0 3 9.5h.1" />
          </g>
        )}
      </svg>
      <span aria-live="polite">{copied ? done : label}</span>
    </button>
  );
}
