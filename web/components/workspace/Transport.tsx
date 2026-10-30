"use client";

import { useSyncExternalStore } from "react";

import { clock } from "@/lib/format";
import type { Player } from "@/lib/player";

import type { Layers } from "./VideoStage";
import styles from "./Workspace.module.css";

const RATES = [0.25, 0.5, 1];

export function Transport({
  player,
  layers,
  onLayers,
  status,
}: {
  player: Player;
  layers: Layers;
  onLayers: (l: Layers) => void;
  status: React.ReactNode;
}) {
  const frame = useSyncExternalStore(player.subscribe, player.getFrame, player.getFrame);
  const playing = useSyncExternalStore(player.subscribe, player.isPlaying, player.isPlaying);
  const rate = useSyncExternalStore(player.subscribe, player.getRate, player.getRate);

  const toggle = (key: keyof Layers) => onLayers({ ...layers, [key]: !layers[key] });

  return (
    <div className={styles.transport}>
      <div className={styles.transportMain}>
        <button type="button" className={styles.play} onClick={() => player.toggle()} aria-label={playing ? "Pause" : "Play"}>
          {playing ? (
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
              <rect x="2.5" y="1.5" width="3.2" height="11" rx="1" fill="currentColor" />
              <rect x="8.3" y="1.5" width="3.2" height="11" rx="1" fill="currentColor" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
              <path d="M3 1.6v10.8L12.2 7z" fill="currentColor" />
            </svg>
          )}
        </button>
        <button type="button" className={styles.icon} onClick={() => player.step(-1)} aria-label="Previous frame">
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <path d="M10.5 2.5 5 7l5.5 4.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M3 2.5v9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
        <button type="button" className={styles.icon} onClick={() => player.step(1)} aria-label="Next frame">
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <path d="M3.5 2.5 9 7l-5.5 4.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M11 2.5v9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
        <p className={styles.readout} aria-live="off">
          <span className={styles.readoutTime}>{clock(frame, player.fps)}</span>
          <span className={styles.readoutFrame}>
            frame {frame} <span>/ {player.nFrames - 1}</span>
          </span>
        </p>
      </div>
      <div className={styles.transportSide}>
        <div className={styles.segmented} role="group" aria-label="Playback speed">
          {RATES.map((r) => (
            <button key={r} type="button" aria-pressed={rate === r} onClick={() => player.setRate(r)}>
              {r === 1 ? "1×" : `${r}×`}
            </button>
          ))}
        </div>
        <div className={styles.segmented} role="group" aria-label="Overlay layers">
          <button type="button" aria-pressed={layers.masks} onClick={() => toggle("masks")}>
            Masks
          </button>
          <button type="button" aria-pressed={layers.trails} onClick={() => toggle("trails")}>
            Trails
          </button>
          <button type="button" aria-pressed={layers.tags} onClick={() => toggle("tags")}>
            IDs
          </button>
        </div>
      </div>
      <div className={styles.samStatus}>{status}</div>
    </div>
  );
}
