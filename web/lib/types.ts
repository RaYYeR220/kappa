// Shapes of a Kappa run as the engine exports it (engine/src/kappa/export.py,
// views.py, public.py) and as the HTTP API serves it (engine/src/kappa/api.py).

export type Status = "verified" | "locked" | "refused";

export type EventKind =
  | "gap"
  | "teleport"
  | "low_confidence"
  | "count_mismatch"
  | "contact"
  | "id_churn"
  | "propagation_disagreement"
  | "scoring_incomplete";

export type Span = [number, number];

export interface Evidence {
  span: Span;
  subject: number | null;
  note: string;
}

export interface ResultView {
  metric_id: string;
  subject: number | null;
  bin_frames: Span | null;
  bin_s: Span | null;
  status: Status;
  value: number | null;
  unit: string;
  uncertainty: Record<string, number>;
  reasons: string[];
  blocked_by: string[];
  fix_frame: number | null;
  evidence: Evidence[];
  params: Record<string, unknown>;
}

export interface EventView {
  id: string;
  kind: EventKind;
  severity: "minor" | "blocking";
  subjects: number[];
  span: Span;
  detail: string;
  fix_frame: number | null;
  resolved_by: string | null;
}

export interface TrackView {
  cx: (number | null)[];
  cy: (number | null)[];
  x_cm: (number | null)[];
  y_cm: (number | null)[];
  area_px: (number | null)[];
  interpolated: number[];
  source: number[];
}

export interface TracksFile {
  fps: number;
  n_frames: number;
  subjects: Record<string, TrackView>;
}

export interface VideoInfo {
  width: number;
  height: number;
  fps: number;
  n_frames: number;
  duration_s: number;
  codec?: string;
}

export interface Zone {
  id: string;
  kind: "center_fraction" | "periphery_band" | "rect_cm";
  fraction?: number | null;
  band_cm?: number | null;
  rect_cm?: [number, number, number, number] | null;
}

export interface MetricSpec {
  id: string;
  kind: string;
  zone?: string | null;
  behaviour?: string | null;
  bin_s?: number | null;
  immobile_below_cm_s?: number;
  immobile_min_s?: number;
}

export interface BehaviourSpec {
  id: string;
  label: string;
  definition: string;
  detector: string;
  adjudicate?: boolean;
  near_wall_cm?: number | null;
}

export interface Protocol {
  schema_version: string;
  title: string;
  question: string;
  subject: { concept: string; expected_count: number; label: string };
  arena: { shape: "rect" | "circle"; width_cm: number; height_cm: number; corners_px: [number, number][] | null };
  zones: Zone[];
  entry_rule?: { mode: "centroid" | "body_fraction"; body_fraction: number; min_dwell_s: number };
  metrics: MetricSpec[];
  behaviours: BehaviourSpec[];
  integrity?: {
    max_speed_cm_s: number;
    max_interpolated_gap_s: number;
    min_confidence: number;
    low_confidence_min_s: number;
    contact_distance: number;
    identity_margin: number;
  };
  smoothing?: { method: string; window_s: number; min_step_cm: number };
  sampling?: { fps: number | null; start_s: number; end_s: number | null };
  refused: { request: string; reason: string }[];
  notes: string;
}

export interface Anchor {
  subject: number;
  frame: number;
  box: [number, number, number, number] | null;
  point: [number, number] | null;
}

export interface RepairView {
  frame: number;
  by: string;
  anchors: Anchor[];
  /** URL of the segmenter's raw re-track from this frame, when it is available. */
  sam: string | null;
}

export interface ReviewFinding {
  severity: "note" | "concern" | "major";
  topic: "design" | "definition" | "tracking" | "statistics" | "reporting";
  claim: string;
  refs: string[];
  suggestion: string;
}

export interface Review {
  verdict: "no_concerns" | "minor_concerns" | "major_concerns";
  findings: ReviewFinding[];
  model?: string;
}

export interface Verdict {
  subject: number;
  candidate: Span;
  present: boolean | null;
  confidence: number | null;
  span: Span | null;
  rationale: string;
  error: string | null;
}

export interface ManifestSummary {
  createdAt: string | null;
  engineVersion: string | null;
  models: Record<string, string>;
  review: Review[];
  methods: string | null;
  withheld: string | null;
}

export type RunOrigin = "published" | "dev" | "live";

/** One run, normalised from a static bundle or from the live API. */
export interface RunData {
  id: string;
  origin: RunOrigin;
  title: string;
  question: string | null;
  sourceName: string;
  video: VideoInfo;
  videoSha256: string;
  protocol: Protocol | null;
  protocolSha256: string | null;
  confirmedBy: string | null;
  assumptions: string[];
  repairs: RepairView[];
  corners: [number, number][] | null;
  analysed: Span | null;
  results: ResultView[];
  events: EventView[];
  verdicts: Record<string, Verdict[]>;
  manifest: ManifestSummary | null;
  tracked: boolean;
  urls: {
    video: string;
    sam: string | null;
    tracks: string | null;
    manifest: string | null;
  };
}

/** What the run list shows about a run, without its heavy files. */
export interface RunCard {
  id: string;
  origin: RunOrigin;
  title: string;
  question: string | null;
  subjectLabel: string;
  subjectCount: number;
  durationS: number;
  fps: number;
  frames: number;
  summary: Record<Status, number>;
  confirmedBy: string | null;
  createdAt: string | null;
  /** Downsampled paths per subject, in video pixels, for the run's thumbnail. */
  paths: { subject: number; points: [number, number][] }[];
  width: number;
  height: number;
}
