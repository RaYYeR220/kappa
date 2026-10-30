// The shared ID palette: eight hues, OKLCH-equidistant, in three lightness tiers.
// Assignment order for small counts: coral, blue, green, pink, amber, cyan, violet, jade.

const ORDER = [0, 5, 2, 7, 1, 4, 6, 3] as const;

/** L .72, for masks and chips on video (≥ 7 : 1 against ink). */
const VIDEO = ["#f9786a", "#da950b", "#87b73a", "#0bc18d", "#17bac8", "#52a9fe", "#ad8dfd", "#e579c4"];
/** L .56, for lines and text on the light ground (≥ 3.9 : 1 on #f1f0ec). */
const LIGHT = ["#bc4a3f", "#9b690b", "#5b8304", "#0d8963", "#04848f", "#1577c8", "#7c5ec2", "#ab4b90"];

export const NEUTRAL = { video: "#c5c3bb", light: "#8b8983" };

function slot(subject: number): number {
  const i = (((subject - 1) % ORDER.length) + ORDER.length) % ORDER.length;
  return ORDER[i] ?? 0;
}

export function subjectColor(subject: number, tier: "video" | "light" = "video"): string {
  const table = tier === "video" ? VIDEO : LIGHT;
  return table[slot(subject)] ?? NEUTRAL[tier];
}

/** "F2" for fish 2, "M1" for mouse 1: the protocol's label initial and the subject number. */
export function subjectTag(subject: number | null, label: string | undefined): string {
  if (subject == null) return "All";
  const initial = (label ?? "subject").trim().charAt(0).toUpperCase() || "S";
  return `${initial}${subject}`;
}
