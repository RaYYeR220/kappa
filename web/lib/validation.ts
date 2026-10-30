import "server-only";

import { readFile } from "node:fs/promises";
import path from "node:path";

import prereg from "@/content/prereg.json";

/**
 * public/validation/summary.json, written by the engine's evaluation once the
 * pre-registered test-set run is done (eval/PREREGISTRATION.md). Until the file
 * exists the site shows the pending state. Schema "kappa.validation/1":
 *
 * {
 *   "schema": "kappa.validation/1",
 *   "generated_at": "2026-11-02T18:00:00Z",          // ISO 8601
 *   "preregistration_commit": "67009b9…",            // commit that added PREREGISTRATION.md
 *   "config_commit": "…",                            // commit that froze the configuration
 *   "behaviour": "rearing",
 *   "fps": 25,
 *   "test_videos": ["OFT_16", …],                    // the 15 test videos, analysed once
 *   "primary": {
 *     "kappa_vs_raters": [{ "rater": "A", "kappa": 0.0, "ci95": [0.0, 0.0] }, …],  // ci95 optional
 *     "human_vs_human": [{ "raters": ["A", "B"], "kappa": 0.0 }, …],             // same 15 videos
 *     "mean_kappa": 0.0,                             // mean of kappa_vs_raters
 *     "threshold": 0.0,                              // lowest human_vs_human kappa
 *     "passed": true                                 // mean_kappa >= threshold
 *   },
 *   "secondary": [{ "label": "Rear count, ICC(2,1)", "kappa": "0.91", "humans": "0.88–0.93", "note": "…" }],
 *   "negative_controls": [{ "id": "shifted", "description": "…", "expected": "κ near 0", "observed": "κ 0.02", "passed": true }],
 *   "integrity": { "verified": 0, "locked": 0, "refused": 0 }  // optional, across the test runs
 * }
 */

export interface RaterKappa {
  rater: string;
  kappa: number;
  ci95?: [number, number];
}

export interface ValidationSummary {
  schema: "kappa.validation/1";
  generated_at: string;
  preregistration_commit: string;
  config_commit?: string;
  behaviour: string;
  fps?: number;
  test_videos: string[];
  primary: {
    kappa_vs_raters: RaterKappa[];
    human_vs_human: { raters: [string, string]; kappa: number }[];
    mean_kappa: number;
    threshold: number;
    passed: boolean;
  };
  secondary?: { label: string; kappa: string; humans?: string; note?: string }[];
  negative_controls?: { id: string; description: string; expected: string; observed: string; passed: boolean }[];
  integrity?: { verified: number; locked: number; refused: number };
}

export const PREREGISTRATION = {
  commit: prereg.commit,
  date: prereg.date,
  /** Rater-vs-rater frame κ for rearing on all 20 Sturman videos, from the pre-registration. */
  humanKappa: [0.801, 0.815, 0.825],
};

const num = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

function valid(s: unknown): s is ValidationSummary {
  if (!s || typeof s !== "object") return false;
  const v = s as Partial<ValidationSummary>;
  const p = v.primary;
  return (
    v.schema === "kappa.validation/1" &&
    typeof v.generated_at === "string" &&
    Array.isArray(v.test_videos) &&
    !!p &&
    Array.isArray(p.kappa_vs_raters) &&
    p.kappa_vs_raters.every((r) => typeof r?.rater === "string" && num(r.kappa)) &&
    Array.isArray(p.human_vs_human) &&
    p.human_vs_human.every((r) => Array.isArray(r?.raters) && num(r.kappa)) &&
    num(p.mean_kappa) &&
    num(p.threshold) &&
    typeof p.passed === "boolean"
  );
}

export type ValidationState =
  | { state: "pending" }
  | { state: "invalid"; problem: string }
  | { state: "published"; summary: ValidationSummary };

export async function loadValidation(): Promise<ValidationState> {
  let raw: string;
  try {
    raw = await readFile(path.join(process.cwd(), "public", "validation", "summary.json"), "utf8");
  } catch {
    return { state: "pending" };
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    return valid(parsed)
      ? { state: "published", summary: parsed }
      : { state: "invalid", problem: "summary.json does not match the kappa.validation/1 schema" };
  } catch {
    return { state: "invalid", problem: "summary.json is not valid JSON" };
  }
}
