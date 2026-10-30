// Copies the engine's protocol library and the pre-registration into web/content/,
// so the site renders exactly what the engine ships. Runs before dev and build.
// When the engine folder is absent (a web-only deploy), the committed copies stay.
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repo = resolve(web, "..");
const library = join(repo, "engine", "src", "kappa", "library");
const prereg = join(repo, "eval", "PREREGISTRATION.md");
const content = join(web, "content");
const protocols = join(content, "protocols");

mkdirSync(protocols, { recursive: true });

if (existsSync(library)) {
  const names = readdirSync(library).filter((n) => n.endsWith(".json")).sort();
  for (const name of names) {
    const body = JSON.parse(readFileSync(join(library, name), "utf8"));
    writeFileSync(join(protocols, name), JSON.stringify(body, null, 2) + "\n");
  }
  writeFileSync(join(protocols, "index.json"), JSON.stringify(names.map((n) => n.replace(/\.json$/, "")), null, 2) + "\n");
  console.log(`content: ${names.length} protocols from the engine library`);
} else {
  console.log("content: engine library not found, keeping committed protocols");
}

if (existsSync(prereg)) {
  copyFileSync(prereg, join(content, "PREREGISTRATION.md"));
  try {
    const out = execFileSync("git", ["log", "--format=%H%x09%cI", "--", "eval/PREREGISTRATION.md"], { cwd: repo, encoding: "utf8" })
      .trim()
      .split("\n")
      .filter(Boolean);
    const first = out.at(-1);
    if (first) {
      const [commit, date] = first.split("\t");
      writeFileSync(join(content, "prereg.json"), JSON.stringify({ commit, date }, null, 2) + "\n");
      console.log(`content: pre-registration ${commit.slice(0, 7)} (${date})`);
    }
  } catch {
    console.log("content: git unavailable, keeping committed pre-registration commit");
  }
}
