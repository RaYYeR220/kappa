"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";

import { Nav } from "@/components/site/Nav";

import { createField } from "./field";
import styles from "./Hero.module.css";

export function Hero({ ctaHref }: { ctaHref: string }) {
  const hero = useRef<HTMLElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const arena = useRef<HTMLDivElement>(null);
  const copy = useRef<HTMLDivElement>(null);
  const clock = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!hero.current || !canvas.current || !arena.current || !copy.current) return;
    const mono = getComputedStyle(document.documentElement).getPropertyValue("--font-geist-mono").trim();
    return createField(
      { hero: hero.current, canvas: canvas.current, arena: arena.current, copy: copy.current, clock: clock.current },
      `${mono ? `${mono}, ` : ""}ui-monospace, monospace`,
    );
  }, []);

  return (
    <section ref={hero} className={styles.hero} aria-labelledby="claim">
      <div ref={arena} className={styles.arena} aria-hidden="true">
        <span className={styles.arenaTag}>
          <b>open field</b> 40 × 40 cm
          <span className={styles.session}>
            session <span ref={clock}>03:09</span>
          </span>
        </span>
      </div>
      <canvas ref={canvas} className={styles.field} aria-hidden="true" />
      <Nav inset />
      <div className={styles.copy} id="main">
        <div ref={copy} className={styles.copyBox}>
          <p className={`${styles.eyebrow} ${styles.reveal}`} style={{ "--i": 0 } as React.CSSProperties}>
            <i aria-hidden="true">
              <span style={{ background: "#f9786a" }} />
              <span style={{ background: "#52a9fe" }} />
              <span style={{ background: "#87b73a" }} />
              <span style={{ background: "#e579c4" }} />
            </i>
            Behavioural measurement built on SAM 3.1
          </p>
          <h1 id="claim" className={styles.reveal} style={{ "--i": 1 } as React.CSSProperties}>
            <span>Every number,</span> <span>with the frames</span> <span>that prove it.</span>
          </h1>
          <p className={`${styles.sub} ${styles.reveal}`} style={{ "--i": 2 } as React.CSSProperties}>
            Kappa turns any experiment video into measurements you can defend. Ask in plain English. It tracks every animal,
            shows its work frame by frame, and refuses to report what it can&rsquo;t prove.
          </p>
          <div className={`${styles.actions} ${styles.reveal}`} style={{ "--i": 3 } as React.CSSProperties}>
            <Link className={styles.cta} href={ctaHref}>
              Open a recorded run
              <span className={styles.play} aria-hidden="true">
                <svg width="10" height="12" viewBox="0 0 10 12">
                  <path d="M1 1.2v9.6L9 6z" fill="#fff" />
                </svg>
              </span>
            </Link>
            <Link className={styles.more} href="/validation">
              How it&rsquo;s validated
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
