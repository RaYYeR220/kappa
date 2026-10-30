import { subjectTag } from "./palette";
import type { EventView, MetricSpec, Protocol, ResultView } from "./types";

const pad = (n: number, w = 2) => String(Math.floor(n)).padStart(w, "0");

/** 00:02.45 for frame 147 at 60 fps. */
export function clock(frame: number, fps: number): string {
  const s = Math.max(0, frame / fps);
  return `${pad(s / 60)}:${pad(s % 60)}.${pad((s * 100) % 100)}`;
}

/** 0:05 */
export function clockShort(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${pad(s % 60)}`;
}

export function frames(n: number): string {
  return n.toLocaleString("en-GB");
}

export function plural(label: string): string {
  const l = label.trim().toLowerCase();
  if (l === "mouse") return "mice";
  if (l === "fish" || l === "sheep" || l === "zebrafish") return l;
  if (/(s|x|ch|sh)$/.test(l)) return `${l}es`;
  if (/[^aeiou]y$/.test(l)) return `${l.slice(0, -1)}ies`;
  return `${l}s`;
}

function words(id: string): string {
  return id.replace(/_/g, " ");
}

function zonePhrase(protocol: Protocol | null, zoneId: string | null | undefined): string {
  const zone = protocol?.zones.find((z) => z.id === zoneId);
  if (!zone) return zoneId ? words(zoneId) : "the zone";
  if (zone.kind === "periphery_band") return `within ${zone.band_cm ?? "?"} cm of the wall`;
  if (zone.kind === "center_fraction") return "the centre";
  return words(zone.id);
}

function behaviourLabel(protocol: Protocol | null, id: string | null | undefined): string {
  return protocol?.behaviours.find((b) => b.id === id)?.label ?? (id ? words(id) : "behaviour");
}

export function metricName(spec: MetricSpec | undefined, protocol: Protocol | null, fallbackId: string): string {
  if (!spec) return words(fallbackId);
  const label = protocol?.subject.label ?? "subject";
  const zone = zonePhrase(protocol, spec.zone);
  const zoneIn = zone.startsWith("within") ? zone : `in ${zone}`;
  switch (spec.kind) {
    case "distance":
      return "Distance travelled";
    case "mean_speed":
      return "Mean speed";
    case "time_in_zone":
      return `Time ${zoneIn}`;
    case "fraction_time_in_zone":
      return `Share of time ${zoneIn}`;
    case "zone_entries":
      return `Entries into ${zone.replace(/^within/, "the band within")}`;
    case "latency_to_zone":
      return `Latency to enter ${zone.replace(/^within/, "the band within")}`;
    case "immobility_time":
      return "Time immobile";
    case "immobility_bouts":
      return "Immobile bouts";
    case "mean_inter_subject_distance":
      return `Mean distance between ${plural(label)}`;
    case "behaviour_count":
      return `${behaviourLabel(protocol, spec.behaviour)} count`;
    case "behaviour_duration":
      return `${behaviourLabel(protocol, spec.behaviour)} time`;
    default:
      return words(spec.id);
  }
}

/** One plain sentence on how the number is computed. */
export function metricHow(spec: MetricSpec | undefined, protocol: Protocol | null): string {
  if (!spec) return "";
  const label = protocol?.subject.label ?? "subject";
  const by = protocol?.entry_rule?.mode === "body_fraction"
    ? `with at least ${Math.round((protocol.entry_rule.body_fraction ?? 0.7) * 100)}% of the body inside`
    : "by the body centre";
  const bins = spec.bin_s ? ` Also reported per ${spec.bin_s >= 60 ? `${spec.bin_s / 60} min` : `${spec.bin_s} s`}.` : "";
  switch (spec.kind) {
    case "distance":
      return `Path length of each ${label}, smoothed, in centimetres on the arena floor.${bins}`;
    case "mean_speed":
      return `Distance over the whole window, unseen stretches included.${bins}`;
    case "time_in_zone":
    case "fraction_time_in_zone":
      return `Seconds ${zonePhrase(protocol, spec.zone).startsWith("within") ? zonePhrase(protocol, spec.zone) : `in ${zonePhrase(protocol, spec.zone)}`}, judged ${by}.${bins}`;
    case "zone_entries":
      return `Separate visits, judged ${by}.${bins}`;
    case "latency_to_zone":
      return `Seconds until the first entry, judged ${by}.`;
    case "immobility_time":
    case "immobility_bouts":
      return `Slower than ${spec.immobile_below_cm_s ?? 2} cm/s for at least ${spec.immobile_min_s ?? 1} s.${bins}`;
    case "mean_inter_subject_distance":
      return `Mean distance between every pair, one value for the group, on frames where every ${label} is visible.`;
    case "behaviour_count":
    case "behaviour_duration":
      return `Bouts confirmed on the video by Muse Spark from candidate stretches.${bins}`;
    default:
      return "";
  }
}

function decimals(value: number, unit: string): number {
  if (unit === "count" || unit === "bouts" || unit === "entries") return 0;
  const a = Math.abs(value);
  if (a >= 1000) return 0;
  if (a >= 100) return 1;
  return 2;
}

export function formatValue(value: number, unit: string): string {
  return value.toLocaleString("en-GB", {
    minimumFractionDigits: decimals(value, unit),
    maximumFractionDigits: decimals(value, unit),
  });
}

export function unitLabel(unit: string): string {
  return unit === "count" ? "" : unit;
}

/** The largest single uncertainty term, the ± shown next to a verified value. */
export function plusMinus(result: ResultView): number | null {
  const v = Object.values(result.uncertainty).filter((x) => Number.isFinite(x));
  return v.length ? Math.max(...v) : null;
}

export function uncertaintyLines(result: ResultView): string[] {
  return Object.entries(result.uncertainty).map(([k, v]) => {
    if (k === "interpolated") {
      return `${formatValue(v, result.unit)} ${unitLabel(result.unit)} of this value comes from frames where the track was filled in between sightings`.replace(/\s+/g, " ");
    }
    return `${words(k)}: ${formatValue(v, result.unit)} ${unitLabel(result.unit)}`.trim();
  });
}

function names(subjects: number[], label: string): string {
  const tags = subjects.map((s) => subjectTag(s, label));
  if (tags.length <= 1) return tags[0] ?? `every ${label}`;
  return `${tags.slice(0, -1).join(", ")} and ${tags.at(-1)}`;
}

/** How long an event lasted: the engine's own figure from its detail when it gives one, else the span. */
function seconds(e: EventView, fps: number): string {
  const said = /\(([\d.]+) s\)/.exec(e.detail)?.[1];
  const s = said ? Number(said) : (e.span[1] - e.span[0]) / fps;
  return `${s < 10 ? s.toFixed(2) : s.toFixed(1)} s`;
}

/** What an integrity event means, in one or two plain sentences. */
export function eventSentence(e: EventView, label: string, fps: number): string {
  const who = names(e.subjects, label);
  const at = clock(e.span[0], fps);
  const long = seconds(e, fps);
  switch (e.kind) {
    case "contact":
      return e.severity === "blocking"
        ? `${who} touched at ${at} for ${long}. Afterwards their motion fits either identity, so who is who is uncertain.`
        : `${who} touched at ${at} for ${long}. Their motion afterwards settled who is who.`;
    case "id_churn":
      return `SAM 3.1 lost ${who} at ${at} and picked up a track that may be a different animal.`;
    case "gap":
      return `${who} was out of sight for ${long} from ${at}.`;
    case "teleport":
      return `${who} moved faster than the protocol allows at ${at}; the track probably jumped to something else.`;
    case "low_confidence":
      return `SAM 3.1 was unsure of ${who} for ${long} from ${at}.`;
    case "count_mismatch":
      return `SAM 3.1 saw a different number of ${plural(label)} than the protocol expects for ${long} from ${at}.`;
    case "propagation_disagreement":
      return `Tracking forward and backward disagree about ${who} from ${at}.`;
    case "scoring_incomplete":
      return `The behaviour review of ${who} from ${at} did not finish.`;
    default:
      return e.detail;
  }
}

export function eventTitle(e: EventView, label: string): string {
  const who = names(e.subjects, label);
  switch (e.kind) {
    case "contact":
      return e.severity === "blocking" ? `ID uncertain: ${who} touched` : `${who} touched`;
    case "id_churn":
      return `Track change: ${who}`;
    case "gap":
      return `Out of sight: ${who}`;
    case "teleport":
      return `Implausible jump: ${who}`;
    case "low_confidence":
      return `Low confidence: ${who}`;
    case "count_mismatch":
      return "Wrong number of animals";
    case "propagation_disagreement":
      return `Tracks disagree: ${who}`;
    case "scoring_incomplete":
      return `Review incomplete: ${who}`;
    default:
      return e.kind;
  }
}

export function resolvedBy(e: EventView): string | null {
  if (!e.resolved_by) return null;
  if (e.resolved_by === "motion-continuity") return "settled by motion continuity";
  return `settled by ${e.resolved_by}`;
}

/** The first blocker of a locked result, in a short line. */
export function lockLine(result: ResultView, events: Map<string, EventView>, label: string, fps: number): string {
  const first = result.blocked_by.map((id) => events.get(id)).find(Boolean);
  if (!first) return result.reasons[0] ?? "An integrity check failed under this number.";
  const more = result.blocked_by.length - 1;
  const base =
    first.kind === "contact"
      ? `ID uncertain after ${names(first.subjects, label)} touched at ${clock(first.span[0], fps)}`
      : eventTitle(first, label);
  return more > 0 ? `${base}, and ${more} more` : base;
}

export function binLabel(result: ResultView): string {
  if (!result.bin_s) return "Whole session";
  return `${clockShort(result.bin_s[0])}–${clockShort(result.bin_s[1])}`;
}

export function shortSha(sha: string | null | undefined, n = 12): string {
  return sha ? sha.slice(0, n) : "none";
}

export function modelName(id: string): string {
  if (id.startsWith("sam-3.1")) return "SAM 3.1";
  if (id.startsWith("muse-spark")) return "Muse Spark";
  if (id.startsWith("muse-wonder")) return "Muse Wonder";
  if (id.startsWith("muse-glimmer")) return "Muse Glimmer";
  return id;
}
