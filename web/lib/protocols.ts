import "server-only";

import { readFile } from "node:fs/promises";
import path from "node:path";

import type { Protocol } from "./types";

const NAMES: Record<string, string> = {
  open_field: "Open field",
  zebrafish_novel_tank: "Zebrafish novel tank",
  elevated_plus_maze: "Elevated plus maze",
};

const ORDER = ["open_field", "zebrafish_novel_tank", "elevated_plus_maze"];

/** The engine's protocol library, as copied into content/protocols by scripts/sync-content.mjs. */
export async function loadProtocols(): Promise<{ key: string; name: string; body: Protocol }[]> {
  const dir = path.join(process.cwd(), "content", "protocols");
  let keys: string[];
  try {
    keys = JSON.parse(await readFile(path.join(dir, "index.json"), "utf8")) as string[];
  } catch {
    return [];
  }
  keys.sort((a, b) => (ORDER.indexOf(a) + 1 || 99) - (ORDER.indexOf(b) + 1 || 99));
  const out: { key: string; name: string; body: Protocol }[] = [];
  for (const key of keys) {
    try {
      const body = JSON.parse(await readFile(path.join(dir, `${key}.json`), "utf8")) as Protocol;
      out.push({
        key,
        name: NAMES[key] ?? key.replace(/_/g, " "),
        body: { ...body, zones: body.zones ?? [], behaviours: body.behaviours ?? [], refused: body.refused ?? [], notes: body.notes ?? "" },
      });
    } catch {
      // a file that doesn't parse is left out rather than shown broken
    }
  }
  return out;
}
