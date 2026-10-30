/** One mask as the segmenter emitted it, cut down to what the overlay needs. */
export interface MaskRec {
  frame: number;
  /** SAM 3.1's object id on this frame (the wire's integer id). */
  objectId: number;
  /** Half-open source-pixel box the raster covers. */
  left: number;
  top: number;
  right: number;
  bottom: number;
  encoding: "one_bit" | "lossless";
  payload: string;
  width: number;
  height: number;
  confidence?: number;
}

export type ToWorker =
  | { type: "load"; key: string; url: string }
  | { type: "stream-start"; key: string }
  | { type: "stream-delta"; key: string; text: string }
  | { type: "stream-end"; key: string };

export type FromWorker =
  | { type: "progress"; key: string; lines: number; total: number }
  | { type: "masks"; key: string; masks: MaskRec[]; done: boolean; diagnostics: number }
  | { type: "error"; key: string; message: string };
