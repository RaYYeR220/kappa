// The landing's open field: a halftone lattice that animals walk through.
// Each animal is a mask contour (marching squares over a metaball body) in its ID
// hue, with a fading trail and a chip that counts its distance. On a loop two
// animals meet, their identities become uncertain and their numbers lock; a
// cursor marks one of them and the lock resolves.

type Pt = [number, number];

interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

interface Walker {
  i: number;
  id: string;
  x: number;
  y: number;
  th: number;
  v: number;
  hist: Pt[];
  cx: number;
  cy: number;
  f1: number;
  f2: number;
  p1: number;
  p2: number;
  dist: number;
  follow: number;
  pause: number;
  state: "free" | "locked" | "resolved";
  mode: "free" | "seek";
  pass: number;
  mate: Walker | null;
}

interface Chip {
  lines: string[];
  st: Walker["state"];
  lead: number;
  hue: number;
  cw: number;
  ch: number;
  x: number;
  y: number;
}

export interface FieldElements {
  hero: HTMLElement;
  canvas: HTMLCanvasElement;
  arena: HTMLElement;
  copy: HTMLElement;
  clock: HTMLElement | null;
}

const TAU = Math.PI * 2;
const P = 8;
const LEVELS = 4;
const CYCLE = 11;

// brief §3: L .56 contours on the light ground, L .72 chips with ink text
const LINE = ["#bc4a3f", "#1577c8", "#5b8304", "#ab4b90", "#9b690b"];
const CHIP = ["#f9786a", "#52a9fe", "#87b73a", "#e579c4", "#da950b"];
const GREY = "#8b8983";

const hex = (h: string): number[] => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const mix = (a: number[], b: number[], t: number) => a.map((v, i) => Math.round(v + ((b[i] ?? 0) - v) * t));
const rgb = (c: number[]) => `rgb(${c[0]},${c[1]},${c[2]})`;
const GROUND = [241, 240, 236];
const DOTC = [200, 197, 189];
const FILL = LINE.map((h) => rgb(mix(GROUND, hex(h), 0.38)));
const FILL_GREY = rgb(mix(GROUND, hex(GREY), 0.24));
const DOTCOL = [GREY, ...LINE].map((h) =>
  Array.from({ length: LEVELS }, (_, l) => rgb(mix(DOTC, hex(h), 0.2 + (0.62 * (l + 1)) / LEVELS))),
);
const DOTBASE = rgb(DOTC);

const rectDist = (x: number, y: number, R: Rect) => {
  const dx = Math.max(R.x0 - x, 0, x - R.x1);
  const dy = Math.max(R.y0 - y, 0, y - R.y1);
  return Math.hypot(dx, dy);
};
const angDiff = (a: number, b: number) => {
  let d = a - b;
  while (d > Math.PI) d -= TAU;
  while (d < -Math.PI) d += TAU;
  return d;
};
const gauss = () => {
  let u = 0;
  let v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
};
const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/** Marching-squares contour of a metaball body: head, ears, trunk, rump. */
function blob(cs: { x: number; y: number; r: number }[], cell: number): Pt[][] {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const c of cs) {
    x0 = Math.min(x0, c.x - c.r * 2);
    y0 = Math.min(y0, c.y - c.r * 2);
    x1 = Math.max(x1, c.x + c.r * 2);
    y1 = Math.max(y1, c.y + c.r * 2);
  }
  const nx = Math.ceil((x1 - x0) / cell) + 1;
  const ny = Math.ceil((y1 - y0) / cell) + 1;
  const f = new Float32Array(nx * ny);
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const px = x0 + i * cell;
      const py = y0 + j * cell;
      let s = 0;
      for (const c of cs) {
        const dx = px - c.x;
        const dy = py - c.y;
        const q = (c.r * c.r) / (dx * dx + dy * dy + 1e-6);
        s += q * q;
      }
      f[j * nx + i] = s - 1;
    }
  const pts = new Map<string, Pt>();
  const adj = new Map<string, string[]>();
  const ep = (a: number, b: number, ax: number, ay: number, bx: number, by: number, key: string) => {
    if (!pts.has(key)) {
      const u = a / (a - b);
      pts.set(key, [ax + (bx - ax) * u, ay + (by - ay) * u]);
    }
    return key;
  };
  const link = (p: string, q: string) => {
    if (!adj.has(p)) adj.set(p, []);
    if (!adj.has(q)) adj.set(q, []);
    adj.get(p)!.push(q);
    adj.get(q)!.push(p);
  };
  for (let j = 0; j < ny - 1; j++)
    for (let i = 0; i < nx - 1; i++) {
      const a = f[j * nx + i]!;
      const b = f[j * nx + i + 1]!;
      const c = f[(j + 1) * nx + i + 1]!;
      const d = f[(j + 1) * nx + i]!;
      const idx = (a > 0 ? 8 : 0) | (b > 0 ? 4 : 0) | (c > 0 ? 2 : 0) | (d > 0 ? 1 : 0);
      if (idx === 0 || idx === 15) continue;
      const X = x0 + i * cell;
      const Y = y0 + j * cell;
      const T = () => ep(a, b, X, Y, X + cell, Y, `h${i},${j}`);
      const Rr = () => ep(b, c, X + cell, Y, X + cell, Y + cell, `v${i + 1},${j}`);
      const B = () => ep(d, c, X, Y + cell, X + cell, Y + cell, `h${i},${j + 1}`);
      const L = () => ep(a, d, X, Y, X, Y + cell, `v${i},${j}`);
      switch (idx) {
        case 1:
        case 14:
          link(L(), B());
          break;
        case 2:
        case 13:
          link(B(), Rr());
          break;
        case 3:
        case 12:
          link(L(), Rr());
          break;
        case 4:
        case 11:
          link(T(), Rr());
          break;
        case 6:
        case 9:
          link(T(), B());
          break;
        case 7:
        case 8:
          link(L(), T());
          break;
        case 5:
          link(L(), T());
          link(B(), Rr());
          break;
        case 10:
          link(T(), Rr());
          link(L(), B());
          break;
      }
    }
  const loops: Pt[][] = [];
  const seen = new Set<string>();
  for (const k of adj.keys()) {
    if (seen.has(k)) continue;
    const loop: Pt[] = [];
    let prev: string | null = null;
    let cur: string | undefined = k;
    while (cur && !seen.has(cur)) {
      seen.add(cur);
      loop.push(pts.get(cur)!);
      const nb: string[] = adj.get(cur) ?? [];
      const nxt: string | undefined = nb.find((q) => q !== prev && !seen.has(q));
      prev = cur;
      cur = nxt;
    }
    if (loop.length > 2) loops.push(loop);
  }
  return loops;
}

export function createField(el: FieldElements, monoFamily: string): () => void {
  const { hero, canvas: cv, arena: arenaEl, copy: copyBox, clock: clockEl } = el;
  const ctx = cv.getContext("2d");
  if (!ctx) return () => {};
  const base = document.createElement("canvas");
  const bctx = base.getContext("2d")!;
  const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
  const FONT = `500 12px ${monoFamily}`;

  // hatch pattern: the refusal treatment, never red
  const hp = document.createElement("canvas");
  hp.width = hp.height = 6;
  const hc = hp.getContext("2d")!;
  hc.strokeStyle = "rgba(14,14,12,.22)";
  hc.lineWidth = 1;
  hc.beginPath();
  hc.moveTo(0, 6);
  hc.lineTo(6, 0);
  hc.moveTo(-1, 1);
  hc.lineTo(1, -1);
  hc.moveTo(5, 7);
  hc.lineTo(7, 5);
  hc.stroke();
  const hatch = ctx.createPattern(hp, "repeat");

  let W = 0;
  let H = 0;
  let dpr = 1;
  let scale = 1;
  let small = false;
  let arena: Rect & { r: number } = { x0: 0, y0: 0, x1: 0, y1: 0, r: 32 };
  let excl: Rect = { x0: 0, y0: 0, x1: 0, y1: 0 };
  let cols = 0;
  let rows = 0;
  let ox = 0;
  let oy = 0;
  let rb = new Float32Array(0);
  let dark = new Float32Array(0);
  let dxs = new Float32Array(0);
  let dys = new Float32Array(0);
  let own = new Int8Array(0);
  let act = new Uint8Array(0);
  let alist: number[] = [];
  let walkers: Walker[] = [];
  let t = 0;
  let phase: "roam" | "approach" | "locked" | "fixing" | "resolved" = "roam";
  let ph = 0;
  let cyc = 0;
  let pair: [Walker, Walker] | null = null;
  let cursor: { x: number; y: number; a: number; press: number } | null = null;
  let ripples: { x: number; y: number; t: number }[] = [];
  let stampTxt = "03:12";

  const free = (x: number, y: number, m: number) =>
    x > arena.x0 + m && x < arena.x1 - m && y > arena.y0 + m && y < arena.y1 - m && rectDist(x, y, excl) > m;

  /** A free spot; on a wide field, walkers alternate between the lanes left and right of the claim. */
  function spawnPos(i = -1): { x: number; y: number } {
    const lanes = excl.x0 - arena.x0 > 160 && arena.x1 - excl.x1 > 160;
    const [lo, hi] =
      i < 0 || !lanes ? [arena.x0, arena.x1] : i % 2 === 0 ? [arena.x0, excl.x0] : [excl.x1, arena.x1];
    for (let k = 0; k < 600; k++) {
      const x = lo + Math.random() * (hi - lo);
      const y = arena.y0 + Math.random() * (arena.y1 - arena.y0);
      if (free(x, y, 60 * scale)) return { x, y };
    }
    return { x: arena.x0 + 60, y: arena.y1 - 60 };
  }

  function makeWalker(i: number): Walker {
    const p = spawnPos(i);
    return {
      i,
      id: `M${i + 1}`,
      x: p.x,
      y: p.y,
      th: Math.random() * TAU,
      v: 0,
      hist: [],
      cx: p.x,
      cy: p.y - 40,
      f1: 0.35 + Math.random() * 0.5,
      f2: 0.13 + Math.random() * 0.3,
      p1: Math.random() * TAU,
      p2: Math.random() * TAU,
      dist: 24 + Math.random() * 18,
      follow: (46 + Math.random() * 46) * scale,
      pause: 0,
      state: "free",
      mode: "free",
      pass: 0,
      mate: null,
    };
  }

  function build() {
    const hr = hero.getBoundingClientRect();
    W = hr.width;
    H = hr.height;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    base.width = cv.width;
    base.height = cv.height;
    small = W < 760;
    scale = small ? 0.78 : 1;
    const a = arenaEl.getBoundingClientRect();
    arena = { x0: a.left - hr.left, y0: a.top - hr.top, x1: a.right - hr.left, y1: a.bottom - hr.top, r: small ? 24 : 32 };
    const c = copyBox.getBoundingClientRect();
    const pad = small ? 18 : 40;
    excl = { x0: c.left - hr.left - pad, y0: c.top - hr.top - pad * 0.7, x1: c.right - hr.left + pad, y1: c.bottom - hr.top + pad * 0.6 };

    cols = Math.max(1, Math.floor((arena.x1 - arena.x0 - 12) / P));
    rows = Math.max(1, Math.floor((arena.y1 - arena.y0 - 12) / P));
    ox = arena.x0 + (arena.x1 - arena.x0 - (cols - 1) * P) / 2;
    oy = arena.y0 + (arena.y1 - arena.y0 - (rows - 1) * P) / 2;
    const N = cols * rows;
    rb = new Float32Array(N);
    dark = new Float32Array(N);
    dxs = new Float32Array(N);
    dys = new Float32Array(N);
    own = new Int8Array(N);
    act = new Uint8Array(N);
    alist = [];
    bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    bctx.clearRect(0, 0, W, H);
    bctx.fillStyle = DOTBASE;
    bctx.beginPath();
    for (let j = 0; j < rows; j++)
      for (let i = 0; i < cols; i++) {
        const x = ox + i * P;
        const y = oy + j * P;
        const k = j * cols + i;
        const de = rectDist(x, y, excl);
        // halftone: dots vanish around the claim and swell toward the walls
        let r = de <= 0 ? 0 : 0.28 + 0.92 * Math.pow(Math.min(1, de / 240), 0.9);
        const e = Math.min(x - arena.x0, arena.x1 - x, y - arena.y0, arena.y1 - y);
        r *= Math.min(1, Math.max(0, (e - 4) / 18));
        const qx = Math.min(Math.max(x, arena.x0 + arena.r), arena.x1 - arena.r);
        const qy = Math.min(Math.max(y, arena.y0 + arena.r), arena.y1 - arena.r);
        if (Math.hypot(x - qx, y - qy) > arena.r - 8) r = 0;
        rb[k] = r;
        if (r > 0.15) {
          bctx.moveTo(x + r, y);
          bctx.arc(x, y, r, 0, TAU);
        }
      }
    bctx.fill();

    const n = small ? 3 : 4;
    if (walkers.length !== n) {
      walkers = Array.from({ length: n }, (_, i) => makeWalker(i));
      phase = "roam";
      pair = null;
      cursor = null;
      cyc = CYCLE - 2.2;
    }
    for (const w of walkers) {
      if (!free(w.x, w.y, 30 * scale)) Object.assign(w, spawnPos(w.i));
      w.hist.length = 0;
      w.cx = w.x;
      w.cy = w.y - 40;
      w.follow = (46 + Math.random() * 46) * scale;
    }
  }

  // nearest boundary (arena wall or the claim), with the normal pointing into free space
  function wall(x: number, y: number) {
    let best = { d: x - arena.x0, nx: 1, ny: 0 };
    const cand: [number, number, number][] = [
      [arena.x1 - x, -1, 0],
      [y - arena.y0, 0, 1],
      [arena.y1 - y, 0, -1],
    ];
    for (const [d, nx, ny] of cand) if (d < best.d) best = { d, nx, ny };
    const qx = Math.min(Math.max(x, excl.x0), excl.x1);
    const qy = Math.min(Math.max(y, excl.y0), excl.y1);
    const dx = x - qx;
    const dy = y - qy;
    const d = Math.hypot(dx, dy);
    if (d > 0 && d < best.d) best = { d, nx: dx / d, ny: dy / d };
    return best;
  }

  function stepWalker(w: Walker, dt: number) {
    const s = scale;
    const wl = wall(w.x, w.y);
    let turn = gauss() * 1.7 * Math.sqrt(dt);
    if (w.pass > 0) {
      w.pass -= dt;
      turn = 0;
    } else if (w.mode === "seek" && w.mate) {
      const want = Math.atan2(w.mate.y - w.y, w.mate.x - w.x);
      turn += angDiff(want, w.th) * Math.min(1, dt * 6);
    } else if (wl.d < w.follow * 1.7) {
      // keep a walker's own distance from the nearest boundary: present, not glued to the wall
      let tx = -wl.ny;
      let ty = wl.nx;
      if (tx * Math.cos(w.th) + ty * Math.sin(w.th) < 0) {
        tx = -tx;
        ty = -ty;
      }
      const err = (wl.d - w.follow) / (w.follow * 0.9);
      const want = Math.atan2(ty - wl.ny * err * 0.8, tx - wl.nx * err * 0.8);
      turn += angDiff(want, w.th) * Math.min(1, dt * 2.6);
    }
    w.th += turn;
    // mice stop and start: a pause now and then, outside the scripted meeting
    if (w.mode === "free" && w.state === "free") {
      if (w.pause > 0) w.pause -= dt;
      else if (Math.random() < dt * 0.06) w.pause = 0.6 + Math.random() * 1.1;
    } else w.pause = 0;
    const g = 0.5 + 0.5 * Math.sin(t * w.f1 + w.p1) * Math.sin(t * w.f2 + w.p2) * 1.6;
    const cruise = s * (22 + 110 * Math.max(0, Math.min(1, g)) ** 2);
    const target = w.mode === "seek" ? 135 * s : w.pause > 0 ? 0 : cruise;
    w.v += (target - w.v) * Math.min(1, dt * 3);
    let nx = w.x + Math.cos(w.th) * w.v * dt;
    let ny = w.y + Math.sin(w.th) * w.v * dt;
    const m = 32 * s;
    if (nx < arena.x0 + m || nx > arena.x1 - m) {
      w.th = Math.PI - w.th;
      nx = Math.min(Math.max(nx, arena.x0 + m), arena.x1 - m);
    }
    if (ny < arena.y0 + m || ny > arena.y1 - m) {
      w.th = -w.th;
      ny = Math.min(Math.max(ny, arena.y0 + m), arena.y1 - m);
    }
    const de = rectDist(nx, ny, excl);
    if (de < m) {
      const qx = Math.min(Math.max(nx, excl.x0), excl.x1);
      const qy = Math.min(Math.max(ny, excl.y0), excl.y1);
      let ex = nx - qx;
      let ey = ny - qy;
      let d = Math.hypot(ex, ey);
      if (d < 0.01) {
        const opts: [number, number, number][] = [
          [nx - excl.x0, -1, 0],
          [excl.x1 - nx, 1, 0],
          [ny - excl.y0, 0, -1],
          [excl.y1 - ny, 0, 1],
        ];
        opts.sort((p, q) => p[0] - q[0]);
        const o = opts[0]!;
        ex = o[1];
        ey = o[2];
        d = 1;
        if (ex) nx = ex < 0 ? excl.x0 : excl.x1;
        if (ey) ny = ey < 0 ? excl.y0 : excl.y1;
      }
      const ux = ex / d;
      const uy = ey / d;
      nx = Math.min(Math.max(nx, excl.x0), excl.x1) + ux * m;
      ny = Math.min(Math.max(ny, excl.y0), excl.y1) + uy * m;
      const vx = Math.cos(w.th);
      const vy = Math.sin(w.th);
      const dot = vx * ux + vy * uy;
      if (dot < 0) w.th = Math.atan2(vy - 2 * dot * uy, vx - 2 * dot * ux);
    }
    const moved = Math.hypot(nx - w.x, ny - w.y);
    w.x = nx;
    w.y = ny;
    if (w.state !== "locked") w.dist += moved * 0.0011;
    const last = w.hist[w.hist.length - 1];
    if (!last || Math.hypot(last[0] - w.x, last[1] - w.y) > 1.5) {
      w.hist.push([w.x, w.y]);
      if (w.hist.length > 110) w.hist.shift();
    }
    // the chip drifts to the open side of the nearest boundary
    const n = wl.d < 150 * s ? wl : { nx: 0, ny: -1 };
    const off = (w.state === "locked" && pair && w === pair[0] ? 78 : 62) * s;
    const k = 1 - Math.exp(-dt * 4);
    w.cx += (w.x + n.nx * off - w.cx) * k;
    w.cy += (w.y + n.ny * off - w.cy) * k;
  }

  function back(w: Walker, L: number): Pt {
    let px = w.x;
    let py = w.y;
    let acc = 0;
    for (let n = w.hist.length - 1; n >= 0; n--) {
      const q = w.hist[n]!;
      const d = Math.hypot(px - q[0], py - q[1]);
      if (d < 0.001) continue;
      if (acc + d >= L) {
        const f = (L - acc) / d;
        return [px + (q[0] - px) * f, py + (q[1] - py) * f];
      }
      acc += d;
      px = q[0];
      py = q[1];
    }
    const r = L - acc;
    return [px - Math.cos(w.th) * r, py - Math.sin(w.th) * r];
  }

  // animals push the dots aside and leave a fading halftone trail in their hue
  function stamp(w: Walker) {
    const R = 56 * scale;
    const R2 = R * R;
    const i0 = Math.max(0, Math.floor((w.x - R - ox) / P));
    const i1 = Math.min(cols - 1, Math.ceil((w.x + R - ox) / P));
    const j0 = Math.max(0, Math.floor((w.y - R - oy) / P));
    const j1 = Math.min(rows - 1, Math.ceil((w.y + R - oy) / P));
    const o = w.state === "locked" ? 0 : w.i + 1;
    for (let j = j0; j <= j1; j++)
      for (let i = i0; i <= i1; i++) {
        const k = j * cols + i;
        if (rb[k]! <= 0) continue;
        const dx = ox + i * P - w.x;
        const dy = oy + j * P - w.y;
        const d2 = dx * dx + dy * dy;
        if (d2 > R2) continue;
        const d = Math.sqrt(d2) || 0.01;
        const f = 1 - d / R;
        const push = f * f * 13 * scale;
        if (push * push > dxs[k]! * dxs[k]! + dys[k]! * dys[k]!) {
          dxs[k] = (dx / d) * push;
          dys[k] = (dy / d) * push;
        }
        const dk = Math.min(1, f * 1.15);
        if (dk > dark[k]!) {
          dark[k] = dk;
          own[k] = o;
        }
        if (!act[k]) {
          act[k] = 1;
          alist.push(k);
        }
      }
  }

  function closestPair(): [Walker, Walker] | null {
    let best: [Walker, Walker] | null = null;
    let bd = Infinity;
    for (let a = 0; a < walkers.length; a++)
      for (let b = a + 1; b < walkers.length; b++) {
        const A = walkers[a]!;
        const B = walkers[b]!;
        const d = Math.hypot(A.x - B.x, A.y - B.y);
        let clear = true;
        for (let s = 1; s < 16; s++) {
          const f = s / 16;
          if (rectDist(A.x + (B.x - A.x) * f, A.y + (B.y - A.y) * f, excl) < 10) {
            clear = false;
            break;
          }
        }
        if (clear && d < bd) {
          bd = d;
          best = [A, B];
        }
      }
    return best && bd < (small ? 420 : 620) ? best : null;
  }

  function story(dt: number) {
    cyc += dt;
    ph += dt;
    if (phase === "roam" && cyc >= CYCLE) {
      cyc = 0;
      const p = closestPair();
      if (p) {
        pair = p.sort((a, b) => a.i - b.i) as [Walker, Walker];
        phase = "approach";
        ph = 0;
        pair[0].mode = pair[1].mode = "seek";
        pair[0].mate = pair[1];
        pair[1].mate = pair[0];
      } else cyc = CYCLE - 1.2;
    } else if (phase === "approach" && pair) {
      const [a, b] = pair;
      if (Math.hypot(a.x - b.x, a.y - b.y) < 32 * scale) {
        phase = "locked";
        ph = 0;
        a.mode = b.mode = "free";
        a.pass = b.pass = 0.7;
        a.state = b.state = "locked";
        stampTxt = fmt(189 + t);
      } else if (ph > 6) {
        phase = "roam";
        a.mode = b.mode = "free";
        cyc = CYCLE - 2;
      }
    } else if (phase === "locked" && pair && ph > 3.2) {
      phase = "fixing";
      ph = 0;
      const a = pair[0];
      cursor = { x: a.x + (a.x < W / 2 ? 140 : -140), y: a.y + 100, a: 0, press: 0 };
    } else if (phase === "fixing" && pair && cursor) {
      const a = pair[0];
      const k = 1 - Math.exp(-dt * 6.5);
      cursor.x += (a.x - cursor.x) * k;
      cursor.y += (a.y - cursor.y) * k;
      cursor.a = Math.min(1, cursor.a + dt * 4);
      cursor.press = ph > 0.85 ? Math.min(1, (ph - 0.85) / 0.12) : 0;
      if (ph > 1.05) {
        phase = "resolved";
        ph = 0;
        ripples.push({ x: cursor.x, y: cursor.y, t: 0 });
        pair.forEach((w) => (w.state = "resolved"));
      }
    } else if (phase === "resolved" && pair && cursor) {
      const a = pair[0];
      const k = 1 - Math.exp(-dt * 5);
      cursor.x += (a.x - cursor.x) * k;
      cursor.y += (a.y - cursor.y) * k;
      cursor.press = Math.max(0, cursor.press - dt * 6);
      cursor.a = ph > 0.8 ? Math.max(0, cursor.a - dt * 2) : cursor.a;
      if (ph > 2.2) {
        phase = "roam";
        pair.forEach((w) => {
          w.state = "free";
          w.mate = null;
        });
        cursor = null;
      }
    }
    ripples.forEach((r) => (r.t += dt));
    ripples = ripples.filter((r) => r.t < 0.7);
  }

  function update(dt: number) {
    t += dt;
    story(dt);
    for (const w of walkers) {
      stepWalker(w, dt);
      stamp(w);
    }
    const dk = Math.exp(-dt / 1.9);
    const mk = Math.exp(-dt * 2.6);
    let n = 0;
    for (let q = 0; q < alist.length; q++) {
      const k = alist[q]!;
      dark[k]! *= dk;
      dxs[k]! *= mk;
      dys[k]! *= mk;
      if (dark[k]! < 0.015 && Math.abs(dxs[k]!) + Math.abs(dys[k]!) < 0.06) {
        act[k] = 0;
        dark[k] = dxs[k] = dys[k] = 0;
      } else alist[n++] = k;
    }
    alist.length = n;
  }

  function drawWalker(w: Walker) {
    const s = scale * 2.45;
    const c = Math.cos(w.th);
    const sn = Math.sin(w.th);
    const locked = w.state === "locked";
    const rp = back(w, 9 * s);
    const at = (f: number, l: number, r: number) => ({ x: w.x + c * f * s - sn * l * s, y: w.y + sn * f * s + c * l * s, r: r * s });
    const mid = back(w, 4.5 * s);
    const cs = [at(14.5, 0, 2.3), at(10, 0, 4.1), at(7.4, 5.1, 2.7), at(7.4, -5.1, 2.7), at(4.5, 0, 5.4), at(0, 0, 6.6), { x: mid[0], y: mid[1], r: 6.8 * s }, { x: rp[0], y: rp[1], r: 7 * s }];
    // the tail follows the path
    ctx!.beginPath();
    let p = back(w, 15 * s);
    ctx!.moveTo(p[0], p[1]);
    for (let L = 18; L <= 40; L += 2.5) {
      p = back(w, L * s);
      ctx!.lineTo(p[0], p[1]);
    }
    ctx!.strokeStyle = locked ? GREY : (LINE[w.i] ?? GREY);
    ctx!.globalAlpha = 0.85;
    ctx!.lineWidth = 2 * scale;
    ctx!.lineCap = "round";
    ctx!.lineJoin = "round";
    ctx!.stroke();
    ctx!.globalAlpha = 1;
    const path = new Path2D();
    for (const loop of blob(cs, 1.7 * s)) {
      path.moveTo(loop[0]![0], loop[0]![1]);
      for (let q = 1; q < loop.length; q++) path.lineTo(loop[q]![0], loop[q]![1]);
      path.closePath();
    }
    if (locked) {
      ctx!.fillStyle = FILL_GREY;
      ctx!.fill(path);
      ctx!.setLineDash([4, 3]);
      ctx!.strokeStyle = "#6b6963";
      ctx!.lineWidth = 1.6;
      ctx!.stroke(path);
      ctx!.setLineDash([]);
    } else {
      ctx!.strokeStyle = "rgba(11,11,10,.75)";
      ctx!.lineWidth = 4.5;
      ctx!.stroke(path);
      ctx!.fillStyle = FILL[w.i] ?? FILL_GREY;
      ctx!.fill(path);
      ctx!.strokeStyle = LINE[w.i] ?? GREY;
      ctx!.lineWidth = 2.4;
      ctx!.stroke(path);
    }
  }

  function lockGlyph(x: number, y: number) {
    ctx!.fillStyle = "#0e0e0c";
    ctx!.fillRect(x, y - 1, 8, 6);
    ctx!.beginPath();
    ctx!.arc(x + 4, y - 1, 2.6, Math.PI, 0);
    ctx!.strokeStyle = "#0e0e0c";
    ctx!.lineWidth = 1.3;
    ctx!.stroke();
  }

  function tick(x: number, y: number) {
    ctx!.beginPath();
    ctx!.moveTo(x, y + 0.5);
    ctx!.lineTo(x + 2.8, y + 3.2);
    ctx!.lineTo(x + 8, y - 2.6);
    ctx!.strokeStyle = "#0e0e0c";
    ctx!.lineWidth = 1.6;
    ctx!.lineCap = "round";
    ctx!.stroke();
  }

  function chipSpecs(): Chip[] {
    ctx!.font = FONT;
    const specs: Chip[] = [];
    const add = (lines: string[], st: Walker["state"], ax: number, ay: number, hue: number) => {
      const lead = st === "free" ? 0 : 15;
      const cw = Math.max(...lines.map((l, n) => ctx!.measureText(l).width + (n === 0 ? lead : 0))) + 18;
      const ch = lines.length * 16 + 10;
      specs.push({ lines, st, lead, hue, cw, ch, x: ax - cw / 2, y: ay - ch / 2 });
    };
    for (const w of walkers) {
      if (w.state !== "free" && pair && w === pair[1]) continue;
      if (w.state === "locked" && pair)
        add(
          [`${pair[0].id} ${pair[1].id}  ID uncertain`, `touched at ${stampTxt}, numbers locked`],
          "locked",
          (pair[0].cx + pair[1].cx) / 2,
          (pair[0].cy + pair[1].cy) / 2,
          0,
        );
      else if (w.state === "resolved" && pair)
        add([`${pair[0].id} ${pair[1].id}  marked, re-tracked`], "resolved", (pair[0].cx + pair[1].cx) / 2, (pair[0].cy + pair[1].cy) / 2, w.i);
      else add([`${w.id}  ${w.dist.toFixed(1)} m`], "free", w.cx, w.cy, w.i);
    }
    for (let it = 0; it < 4; it++)
      for (let a = 0; a < specs.length; a++)
        for (let b = a + 1; b < specs.length; b++) {
          const A = specs[a]!;
          const B = specs[b]!;
          const ox_ = Math.min(A.x + A.cw, B.x + B.cw) - Math.max(A.x, B.x) + 4;
          const oy_ = Math.min(A.y + A.ch, B.y + B.ch) - Math.max(A.y, B.y) + 4;
          if (ox_ > 0 && oy_ > 0) {
            const dir = A.y + A.ch / 2 <= B.y + B.ch / 2 ? 1 : -1;
            A.y -= (dir * oy_) / 2;
            B.y += (dir * oy_) / 2;
          }
        }
    for (const c of specs) {
      c.x = Math.min(Math.max(c.x, arena.x0 + 8), arena.x1 - c.cw - 8);
      c.y = Math.min(Math.max(c.y, arena.y0 + 8), arena.y1 - c.ch - 8);
    }
    return specs;
  }

  function drawChip(c: Chip) {
    ctx!.font = FONT;
    ctx!.textBaseline = "middle";
    const { x, y, cw, ch, st, lines, lead } = c;
    ctx!.beginPath();
    ctx!.roundRect(x, y, cw, ch, 6);
    if (st === "locked") {
      ctx!.fillStyle = "#e9e7e2";
      ctx!.fill();
      if (hatch) {
        ctx!.fillStyle = hatch;
        ctx!.fill();
      }
      ctx!.strokeStyle = "rgba(14,14,12,.45)";
      ctx!.lineWidth = 1;
      ctx!.stroke();
    } else {
      ctx!.fillStyle = CHIP[c.hue] ?? "#e9e7e2";
      ctx!.fill();
    }
    ctx!.fillStyle = "#0e0e0c";
    lines.forEach((l, n) => ctx!.fillText(l, x + 9 + (n === 0 ? lead : 0), y + 5 + 8 + n * 16));
    if (st === "locked") lockGlyph(x + 9, y + 5 + 8 - 1.5);
    if (st === "resolved") tick(x + 8, y + 5 + 8 - 0.5);
  }

  function drawCursor() {
    if (cursor && cursor.a > 0) {
      ctx!.globalAlpha = cursor.a;
      const r = 17 - cursor.press * 5;
      ctx!.beginPath();
      ctx!.arc(cursor.x, cursor.y, r, 0, TAU);
      ctx!.strokeStyle = "#2448e8";
      ctx!.lineWidth = 2;
      ctx!.stroke();
      const px = cursor.x + 8;
      const py = cursor.y + 8;
      ctx!.beginPath();
      ctx!.moveTo(px, py);
      ctx!.lineTo(px, py + 17);
      ctx!.lineTo(px + 4.5, py + 13);
      ctx!.lineTo(px + 7.6, py + 19);
      ctx!.lineTo(px + 10, py + 18);
      ctx!.lineTo(px + 7.2, py + 12);
      ctx!.lineTo(px + 12.4, py + 12);
      ctx!.closePath();
      ctx!.fillStyle = "#0e0e0c";
      ctx!.fill();
      ctx!.strokeStyle = "#f7f6f3";
      ctx!.lineWidth = 1.2;
      ctx!.stroke();
      ctx!.globalAlpha = 1;
    }
    for (const rp of ripples) {
      const k = rp.t / 0.7;
      ctx!.beginPath();
      ctx!.arc(rp.x, rp.y, 12 + k * 40, 0, TAU);
      ctx!.strokeStyle = `rgba(36,72,232,${(1 - k) * 0.8})`;
      ctx!.lineWidth = 1.8;
      ctx!.stroke();
    }
  }

  const paths: Path2D[] = Array.from({ length: (LINE.length + 1) * LEVELS + 1 }, () => new Path2D());
  function render() {
    ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx!.clearRect(0, 0, W, H);
    ctx!.drawImage(base, 0, 0, W, H);
    if (alist.length) {
      ctx!.globalCompositeOperation = "destination-out";
      ctx!.beginPath();
      for (const k of alist) {
        const x = ox + (k % cols) * P;
        const y = oy + ((k / cols) | 0) * P;
        const r = rb[k]! + 0.9;
        ctx!.moveTo(x + r, y);
        ctx!.arc(x, y, r, 0, TAU);
      }
      ctx!.fill();
      ctx!.globalCompositeOperation = "source-over";
      for (let q = 0; q < paths.length; q++) paths[q] = new Path2D();
      for (const k of alist) {
        const x = ox + (k % cols) * P + dxs[k]!;
        const y = oy + ((k / cols) | 0) * P + dys[k]!;
        const d = dark[k]!;
        const r = rb[k]! + 1.7 * d * scale;
        const b = d < 0.06 ? paths.length - 1 : own[k]! * LEVELS + Math.min(LEVELS - 1, (d * LEVELS) | 0);
        paths[b]!.moveTo(x + r, y);
        paths[b]!.arc(x, y, r, 0, TAU);
      }
      for (let o = 0; o <= LINE.length; o++)
        for (let l = 0; l < LEVELS; l++) {
          ctx!.fillStyle = DOTCOL[o]![l]!;
          ctx!.fill(paths[o * LEVELS + l]!);
        }
      ctx!.fillStyle = DOTBASE;
      ctx!.fill(paths[paths.length - 1]!);
    }
    walkers.forEach(drawWalker);
    chipSpecs().forEach(drawChip);
    drawCursor();
  }

  let last = 0;
  let raf = 0;
  let visible = true;
  let clockAcc = 0;
  function frame(now: number) {
    const dt = Math.min(0.05, (now - (last || now)) / 1000);
    last = now;
    update(dt);
    render();
    clockAcc += dt;
    if (clockAcc > 0.25 && clockEl) {
      clockAcc = 0;
      clockEl.textContent = fmt(189 + t);
    }
    raf = visible ? requestAnimationFrame(frame) : 0;
  }

  function start() {
    cancelAnimationFrame(raf);
    raf = 0;
    last = 0;
    if (mq.matches) {
      // reduced motion: one honest still, the moment the numbers lock
      cyc = CYCLE - 0.2;
      for (let n = 0; n < 60 * 40; n++) {
        update(1 / 60);
        if (phase === "locked" && ph > 0.5) break;
      }
      render();
      if (clockEl) clockEl.textContent = fmt(189 + t);
    } else raf = requestAnimationFrame(frame);
  }

  build();
  for (let n = 0; n < 120; n++) update(1 / 60); // pre-roll so trails exist on first paint
  start();
  void document.fonts?.ready.then(() => render());

  const io = new IntersectionObserver(([e]) => {
    visible = !!e?.isIntersecting;
    if (visible && !raf && !mq.matches) {
      last = 0;
      raf = requestAnimationFrame(frame);
    }
  });
  io.observe(hero);
  let rt = 0;
  const ro = new ResizeObserver(() => {
    window.clearTimeout(rt);
    rt = window.setTimeout(() => {
      build();
      if (mq.matches) start();
    }, 120);
  });
  ro.observe(hero);
  const onMotion = () => start();
  mq.addEventListener("change", onMotion);

  return () => {
    cancelAnimationFrame(raf);
    io.disconnect();
    ro.disconnect();
    window.clearTimeout(rt);
    mq.removeEventListener("change", onMotion);
  };
}
