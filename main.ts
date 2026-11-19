// Leonida Rarity Forecast — GTA VI trophy earn-rate forecaster.
//
// Reads every <id>.in case in MAKEFUN_INPUT_DIR (default /input) and writes one
// <id>.out forecast per case to MAKEFUN_OUTPUT_DIR (default /output).
// Node 22 stdlib only, deterministic, no network.
//
// How it works, in brief:
//  * Exophase refreshes a game's counts in steps, not continuously: consecutive
//    reads are often identical, then players_tracked jumps and every rate moves.
//    So the history is first collapsed into "plateaus" (runs of identical reads);
//    only plateau endpoints carry new information.
//  * Each trophy's rate is modelled on the logit scale against log-time
//    x = ln(1 + hours since release). The slope is a robust Theil-Sen fit over
//    the plateaus, shrunk toward its tier's mean slope (rare trophies are noisy),
//    floored at a small positive value and capped, then extrapolated to the
//    target with the move capped so a short-window trend cannot run away over a
//    multi-day horizon.
//  * The spread comes from the fit residuals plus the refresh jumps seen in the
//    history, so it widens honestly when the past was jumpy.
//  * When the history is completely flat (no refresh seen yet, as on launch
//    night), no trend can be fit. Then the forecast is a two-state mixture:
//    either no refresh lands before the target (rates barely move) or one does,
//    in which case a big cohort of fresh players joins and dilutes every rate
//    but the first trophy's. The mixture's quantiles are computed on a grid.
//    For a very long horizon with a flat history the mixture is replaced by a
//    wide, gently rising forecast, since a week of play lifts rates overall.

import * as fs from "node:fs";
import * as path from "node:path";

type Trophy = { id: number; name: string; tier: string };
type Read = { at: string; players_tracked: number; rates: (number | null)[] };
type Case = {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  game: { release_date: string };
  trophies: Trophy[];
  history: Read[];
};

// Standard-normal quantiles for the five levels [0.05, 0.25, 0.5, 0.75, 0.95].
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const TAU = [0.05, 0.25, 0.5, 0.75, 0.95];

// --- tuning constants (chosen from the practice backtest, see bench notes) ---
const SHRINK_K = 3; // pseudo-count pulling each slope toward its tier mean
const SLOPE_FLOOR = 0.02; // logit per log-hour: slowest believable growth
const SLOPE_CAP = 2.5;
const MAX_MOVE = 2.5; // logit cap on the extrapolated move from the last read
const RESID_FLOOR = 0.12; // minimum logit spread from fit residuals
const RESID_CAP = 0.9;
const REFRESH_INTERVAL_BASE_H = 5; // prior mean gap between Exophase refreshes
const REFRESH_INTERVAL_MIN_H = 1.5; // a page stale for hours is due one: hazard rises
const REFRESH_SHIFT = -1.5; // logit drop a refresh inflicts on non-first trophies
const MIX_NOISE = 0.18; // logit noise around each mixture state
const MIX_DRIFT = 0.03; // logit growth if no refresh lands before the target
const LONG_HORIZON_H = 24; // above this, a flat history gets the wide-rise fallback
const LONG_MEDIAN_DRIFT = 0.3;
const LONG_SPREAD = 1.2;

function clampPct(x: number): number {
  return x < 0.01 ? 0.01 : x > 99.99 ? 99.99 : x;
}
function logitPct(x: number): number {
  const p = clampPct(x) / 100;
  return Math.log(p / (1 - p));
}
function invLogitPct(l: number): number {
  return 100 / (1 + Math.exp(-l));
}
function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
function sd(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / (xs.length - 1));
}

// Theil-Sen slope: median of all pairwise slopes. Robust to the odd refresh jump.
function theilSen(xs: number[], ys: number[]): number {
  const slopes: number[] = [];
  for (let i = 0; i < xs.length; i++) {
    for (let j = i + 1; j < xs.length; j++) {
      if (xs[j] !== xs[i]) slopes.push((ys[j] - ys[i]) / (xs[j] - xs[i]));
    }
  }
  return slopes.length ? median(slopes) : 0;
}

// Quantiles of a two-state normal mixture on a fixed grid (deterministic).
// State A (weight 1-p): N(muA, s); state B (weight p): N(muB, s).
function mixtureQuantiles(muA: number, muB: number, s: number, p: number): number[] {
  const lo = Math.min(muA, muB) - 6 * s - 0.5;
  const hi = Math.max(muA, muB) + 6 * s + 0.5;
  const steps = 2400;
  const grid: number[] = [];
  for (let i = 0; i <= steps; i++) grid.push(lo + ((hi - lo) * i) / steps);
  const cdf = (x: number): number => {
    const pa = 0.5 * (1 + erf((x - muA) / (s * Math.SQRT2)));
    const pb = 0.5 * (1 + erf((x - muB) / (s * Math.SQRT2)));
    return (1 - p) * pa + p * pb;
  };
  const out: number[] = [];
  let gi = 0;
  for (const t of TAU) {
    while (gi < grid.length - 1 && cdf(grid[gi]) < t) gi++;
    out.push(grid[gi]);
  }
  return out;
}

// Abramowitz-Stegun erf approximation (accurate to ~1e-7; fine for quantiles).
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) *
      Math.exp(-ax * ax);
  return sign * y;
}

function round4(x: number): number {
  return Math.round(x * 1e4) / 1e4;
}

// Build one row of five quantiles (percent) from five logit values.
function rowFromLogits(id: number, ls: number[]): { id: number; q: number[] } {
  const q: number[] = [];
  let prev = -Infinity;
  for (const l of ls) {
    let v = round4(clampPct(invLogitPct(l)));
    if (v < prev) v = prev;
    if (v > 100) v = 100;
    q.push(v);
    prev = v;
  }
  return { id, q };
}

function forecastCase(c: Case): { trophies: { id: number; q: number[] }[] } {
  const releaseMs = Date.parse(c.game.release_date + "T00:00:00Z");
  const hoursSince = (iso: string): number => (Date.parse(iso) - releaseMs) / 3.6e6;
  const xOf = (iso: string): number => Math.log(1 + Math.max(0, hoursSince(iso)));
  const targetX = xOf(c.target);
  const horizonH = c.horizon_minutes / 60;

  // Collapse consecutive identical reads into plateaus; keep the last of each.
  const plateaus: Read[] = [];
  for (const r of c.history) {
    const prev = plateaus[plateaus.length - 1];
    if (
      prev &&
      prev.players_tracked === r.players_tracked &&
      prev.rates.length === r.rates.length &&
      prev.rates.every((v, i) => v === r.rates[i])
    ) {
      plateaus[plateaus.length - 1] = r;
    } else {
      plateaus.push(r);
    }
  }

  const last = c.history.length ? c.history[c.history.length - 1] : null;
  const lastRates: (number | null)[] = last ? last.rates : c.trophies.map(() => null);
  const lastX = plateaus.length ? xOf(plateaus[plateaus.length - 1].at) : 0;

  // Per-trophy plateau series on the logit scale.
  const series: { xs: number[]; ys: number[] }[] = c.trophies.map((_, i) => {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const p of plateaus) {
      const r = p.rates[i];
      if (r === null || r === undefined) continue;
      xs.push(xOf(p.at));
      ys.push(logitPct(r));
    }
    return { xs, ys };
  });

  const usable = series.filter((s) => s.xs.length >= 3).length;
  const rows: { id: number; q: number[] }[] = [];

  if (plateaus.length >= 3 && usable >= 5) {
    // ---------------- trend regime: real history to learn from ----------------
    const slopes: (number | null)[] = series.map((s) =>
      s.xs.length >= 3 ? theilSen(s.xs, s.ys) : null
    );
    // Tier means (fall back to the global mean for a tier with no fitted slope).
    const tierSlopes = new Map<string, number>();
    const global = slopes.filter((s): s is number => s !== null);
    const globalMean = global.length ? mean(global) : 0;
    for (const tier of new Set(c.trophies.map((t) => t.tier))) {
      const vals = c.trophies
        .map((t, i) => (t.tier === tier ? slopes[i] : null))
        .filter((s): s is number => s !== null);
      tierSlopes.set(tier, vals.length ? mean(vals) : globalMean);
    }
    const globalMedianSlope = global.length ? median(global) : 0;

    // Pooled residual spread around each trophy's fitted line.
    const residuals: number[] = [];
    const jumpSizes: number[] = []; // median logit jump per refresh (N increase)
    for (let p = 1; p < plateaus.length; p++) {
      if (plateaus[p].players_tracked > plateaus[p - 1].players_tracked * 1.05) {
        const jumps: number[] = [];
        for (let i = 0; i < c.trophies.length; i++) {
          const a = plateaus[p - 1].rates[i];
          const b = plateaus[p].rates[i];
          if (a !== null && b !== null && a > 0 && a < 100) jumps.push(logitPct(b) - logitPct(a));
        }
        if (jumps.length) jumpSizes.push(median(jumps));
      }
    }
    series.forEach((s, i) => {
      const b = slopes[i];
      if (b === null || s.xs.length < 3) return;
      const anchor = s.ys[s.ys.length - 1];
      const ax = s.xs[s.xs.length - 1];
      for (let k = 0; k < s.xs.length; k++) {
        residuals.push(s.ys[k] - (anchor + b * (s.xs[k] - ax)));
      }
    });
    const sRes = Math.min(RESID_CAP, Math.max(RESID_FLOOR, residuals.length > 5 ? sd(residuals) : RESID_FLOOR));
    const jumpSd = jumpSizes.length >= 2 ? sd(jumpSizes) : 0.25;
    const refreshGapH = 12; // prior mean gap between refreshes once some are seen
    const expectedJumps = horizonH / refreshGapH;
    const sRefresh = Math.min(1.2, Math.max(0.15, jumpSd * Math.sqrt(expectedJumps)));

    c.trophies.forEach((t, i) => {
      const r = lastRates[i];
      if (r !== null && r >= 99.99) {
        rows.push({ id: t.id, q: [99.99, 100, 100, 100, 100] });
        return;
      }
      const base = r === null || r <= 0 ? 0.05 : r; // unseen trophy: tiny positive prior
      const lastLogit = logitPct(base);
      let b = slopes[i];
      if (b === null) {
        b = tierSlopes.get(t.tier) ?? globalMean;
      } else {
        const nEff = Math.min(series[i].xs.length, 8);
        b = (nEff * b + SHRINK_K * (tierSlopes.get(t.tier) ?? globalMean)) / (nEff + SHRINK_K);
      }
      if (globalMedianSlope > 0) b = Math.max(b, SLOPE_FLOOR);
      b = Math.max(-0.4, Math.min(SLOPE_CAP, b));
      let move = b * (targetX - lastX);
      move = Math.max(-MAX_MOVE, Math.min(MAX_MOVE, move));
      const centre = lastLogit + move;
      // Slope uncertainty grows with the extrapolation distance.
      const xs = series[i].xs;
      const xSpread = xs.length > 1 ? sd(xs) : 1;
      const seSlope = sRes / (Math.sqrt(Math.max(1, xs.length)) * Math.max(0.2, xSpread));
      const spread = Math.sqrt(sRes * sRes + Math.pow(seSlope * (targetX - lastX), 2) + sRefresh * sRefresh);
      rows.push(rowFromLogits(t.id, Z.map((z) => centre + z * spread)));
    });
  } else {
    // ---------------- flat regime: no refresh observed yet ----------------
    // How stale is the page? Every read so far is identical, so the last
    // refresh predates the first read; the longer the page has sat unchanged,
    // the sooner the next refresh is due. The hazard interval shrinks from
    // the 5h prior toward 1.5h as staleness grows.
    const stalenessH = c.history.length
      ? (Date.parse(c.at) - Date.parse(c.history[0].at)) / 3.6e6
      : 0;
    const hazardIntervalH = Math.max(
      REFRESH_INTERVAL_MIN_H,
      Math.min(REFRESH_INTERVAL_BASE_H, REFRESH_INTERVAL_BASE_H - stalenessH)
    );
    const pRefresh = horizonH > LONG_HORIZON_H ? 1 : 1 - Math.exp(-horizonH / hazardIntervalH);
    c.trophies.forEach((t, i) => {
      const r = lastRates[i];
      if (r !== null && r >= 99.99) {
        rows.push({ id: t.id, q: [99.99, 100, 100, 100, 100] });
        return;
      }
      const base = r === null || r <= 0 ? 0.05 : r;
      const l0 = logitPct(base);
      if (horizonH > LONG_HORIZON_H) {
        const centre = l0 + LONG_MEDIAN_DRIFT;
        rows.push(rowFromLogits(t.id, Z.map((z) => centre + z * LONG_SPREAD)));
      } else {
        const ls = mixtureQuantiles(l0 + MIX_DRIFT, l0 + REFRESH_SHIFT, MIX_NOISE, pRefresh);
        rows.push(rowFromLogits(t.id, ls));
      }
    });
  }

  return { trophies: rows };
}

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  fs.mkdirSync(outDir, { recursive: true });
  const files = fs.readdirSync(inDir).filter((f) => f.endsWith(".in")).sort();
  for (const f of files) {
    const c = JSON.parse(fs.readFileSync(path.join(inDir, f), "utf8")) as Case;
    const out = forecastCase(c);
    fs.writeFileSync(path.join(outDir, f.replace(/\.in$/, ".out")), JSON.stringify(out) + "\n");
  }
}

main();
