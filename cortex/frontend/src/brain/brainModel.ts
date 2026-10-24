/**
 * Procedural brain-shaped point cloud.
 *
 * The surface is the zero level of a signed-distance field built from blended ellipsoids
 * (two hemispheres with frontal, parieto-occipital and temporal lobes, a longitudinal fissure,
 * a flattened base), a cerebellum and a brainstem. Particles are sampled on that surface;
 * noise-driven sulci, the lateral (Sylvian) fissure and the central sulcus are carved in by
 * displacing and thinning particles. The hippocampi are curved tubes of interior particles.
 *
 * Axes: x = left(-)/right(+), y = inferior(-)/superior(+), z = posterior(-)/anterior(+).
 * Regions are a navigation metaphor for memory categories, not anatomy claims.
 */

import type { RegionId } from "../types";
import { createNoise3D, hashString, mulberry32 } from "./noise";

export const REGION_CODES: RegionId[] = ["prefrontal", "frontal", "temporal", "hippocampus", "occipital"];
export const UNMAPPED = 5;
export const regionCode = (region: RegionId): number => REGION_CODES.indexOf(region);

export interface BrainCloud {
  count: number;
  positions: Float32Array;
  normals: Float32Array;
  brightness: Float32Array;
  region: Float32Array;
  size: Float32Array;
  /** Particle indices per region code (0..4), well-lit gyri first-class candidates for anchors. */
  regionIndices: number[][];
}

// ------------------------------------------------------------------ distance field

function sdEllipsoid(x: number, y: number, z: number, cx: number, cy: number, cz: number, rx: number, ry: number, rz: number) {
  const px = x - cx;
  const py = y - cy;
  const pz = z - cz;
  const k0 = Math.sqrt((px / rx) ** 2 + (py / ry) ** 2 + (pz / rz) ** 2);
  const k1 = Math.sqrt((px / (rx * rx)) ** 2 + (py / (ry * ry)) ** 2 + (pz / (rz * rz)) ** 2);
  return k1 === 0 ? -Math.min(rx, ry, rz) : (k0 * (k0 - 1)) / k1;
}

function sdCapsule(x: number, y: number, z: number, ax: number, ay: number, az: number, bx: number, by: number, bz: number, r: number) {
  const pax = x - ax, pay = y - ay, paz = z - az;
  const bax = bx - ax, bay = by - ay, baz = bz - az;
  const h = Math.max(0, Math.min(1, (pax * bax + pay * bay + paz * baz) / (bax * bax + bay * bay + baz * baz)));
  return Math.hypot(pax - bax * h, pay - bay * h, paz - baz * h) - r * (1 - 0.25 * h);
}

const smoothstep = (a: number, b: number, v: number) => {
  const t = Math.max(0, Math.min(1, (v - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

const smin = (a: number, b: number, k: number) => {
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
};
const smax = (a: number, b: number, k: number) => -smin(-a, -b, k);

/** Height of the cerebral underside: raised under the frontal lobe, lowest mid-way, rising toward the occipital pole. */
const baseY = (z: number) => -0.46 + 0.25 * smoothstep(0.15, 0.7, z) + 0.12 * (1 - smoothstep(-0.95, -0.45, z));

/** Half-width of the cerebrum along its length: widest over the parietal lobes, narrower toward both poles. */
export const widthScale = (z: number) => 1 - 0.25 * smoothstep(-0.15, 0.95, z) - 0.12 * smoothstep(-0.5, -1.0, z);

/** Height along the length: the occipital pole tapers, the frontal pole stays blunt. */
const heightScale = (z: number) => 1 - 0.14 * smoothstep(-0.35, -1.0, z);

/** Half the width of the longitudinal fissure between the hemispheres. */
export const FISSURE = 0.012;

/**
 * Both hemispheres as one egg-shaped body (seen from above: narrow forehead, widest behind the
 * middle), with temporal lobes below the lateral fissure, split down the midline by a thin
 * fissure so each hemisphere has a flat medial wall and a rounded lateral side.
 */
function cerebrum(x: number, y: number, z: number): number {
  const s = widthScale(z);
  const h = heightScale(z);
  // Ellipsoid evaluated in squeezed space: the distance is approximate, the surface exact.
  let d = sdEllipsoid(x / s, 0.05 + (y - 0.05) / h, z, 0, 0.05, -0.06, 0.78, 0.51, 0.9) * Math.min(s, h);
  d = smax(d, baseY(z) - y, 0.1);
  for (const side of [1, -1]) d = smin(d, sdEllipsoid(x, y, z, side * 0.46, -0.29, 0.12, 0.21, 0.2, 0.43), 0.05);
  d = smax(d, FISSURE - Math.abs(x), 0.015);
  return smax(d, -0.52 - y, 0.05);
}

const CEREBRUM = 0;
const CEREBELLUM = 1;
const STEM = 2;

function field(x: number, y: number, z: number): [number, number] {
  const cerebrumD = cerebrum(x, y, z);
  const cerebellum = smin(
    sdEllipsoid(x, y, z, 0.22, -0.46, -0.58, 0.28, 0.17, 0.25),
    sdEllipsoid(x, y, z, -0.22, -0.46, -0.58, 0.28, 0.17, 0.25),
    0.12,
  );
  const stem = sdCapsule(x, y, z, 0, -0.35, -0.3, 0, -0.9, -0.4, 0.1);
  if (cerebrumD <= cerebellum && cerebrumD <= stem) return [cerebrumD, CEREBRUM];
  if (cerebellum <= stem) return [cerebellum, CEREBELLUM];
  return [stem, STEM];
}

const fieldValue = (x: number, y: number, z: number) => field(x, y, z)[0];

function gradient(x: number, y: number, z: number, out: Float32Array): void {
  const e = 0.002;
  const gx = fieldValue(x + e, y, z) - fieldValue(x - e, y, z);
  const gy = fieldValue(x, y + e, z) - fieldValue(x, y - e, z);
  const gz = fieldValue(x, y, z + e) - fieldValue(x, y, z - e);
  const len = Math.hypot(gx, gy, gz) || 1;
  out[0] = gx / len;
  out[1] = gy / len;
  out[2] = gz / len;
}

// ------------------------------------------------------------------ landmarks and regions

/** Height of the lateral (Sylvian) fissure along the anterior-posterior axis. */
export const sylvianY = (z: number) => -0.09 + (0.4 - z) * 0.2;
/** Anterior-posterior position of the central sulcus at a given height. */
export const centralZ = (y: number) => 0.1 - 0.22 * y;

/** Region code for a point on the cerebral surface (UNMAPPED for parietal and the rest). */
export function classifyCortex(x: number, y: number, z: number): number {
  const ax = Math.abs(x);
  const sy = sylvianY(z);
  if (y < sy && z > -0.42 && z < 0.62 && ax > 0.22) return regionCode("temporal");
  if (z < -0.55 + 0.1 * y) return regionCode("occipital");
  if (z > 0.48 - 0.12 * y) return regionCode("prefrontal");
  if (z > centralZ(y) && y >= sy - 0.02) return regionCode("frontal");
  return UNMAPPED;
}

// ------------------------------------------------------------------ generation

export interface BrainOptions {
  surface?: number;
  hippocampus?: number;
  dust?: number;
  seed?: number;
}

export function generateBrain({ surface = 42000, hippocampus = 2600, dust = 2200, seed = 7 }: BrainOptions = {}): BrainCloud {
  const random = mulberry32(seed);
  const noise = createNoise3D(mulberry32(seed * 31 + 5));
  const total = surface + hippocampus + dust;
  const positions = new Float32Array(total * 3);
  const normals = new Float32Array(total * 3);
  const brightness = new Float32Array(total);
  const region = new Float32Array(total);
  const size = new Float32Array(total);
  const regionIndices: number[][] = REGION_CODES.map(() => []);
  const g = new Float32Array(3);
  let n = 0;

  const put = (x: number, y: number, z: number, nx: number, ny: number, nz: number, b: number, r: number, s: number) => {
    positions.set([x, y, z], n * 3);
    normals.set([nx, ny, nz], n * 3);
    brightness[n] = b;
    region[n] = r;
    size[n] = s;
    if (r !== UNMAPPED) regionIndices[r].push(n);
    n++;
  };

  const SHELL = 0.03;
  let tries = 0;
  while (n < surface && tries < surface * 80) {
    tries++;
    let x = -0.95 + random() * 1.9;
    let y = -1.05 + random() * 1.8;
    let z = -1.0 + random() * 2.0;
    const [d0] = field(x, y, z);
    if (Math.abs(d0) > SHELL) continue;
    // Two Newton steps onto the surface.
    for (let it = 0; it < 2; it++) {
      const d = fieldValue(x, y, z);
      gradient(x, y, z, g);
      x -= g[0] * d;
      y -= g[1] * d;
      z -= g[2] * d;
    }
    gradient(x, y, z, g);
    const [, component] = field(x, y, z);
    const [nx, ny, nz] = g;
    let groove = 0;
    let shade = 1;
    if (component === CEREBRUM) {
      const w = 0.16;
      const wx = noise(x * 1.7 + 11, y * 1.7, z * 1.7) * w;
      const wy = noise(x * 1.7, y * 1.7 + 23, z * 1.7) * w;
      const wz = noise(x * 1.7, y * 1.7, z * 1.7 + 37) * w;
      const f = 6.2;
      const n1 = noise((x + wx) * f, (y + wy) * f, (z + wz) * f);
      const n2 = noise(x * f * 2.2 + 5, y * f * 2.2, z * f * 2.2);
      groove = 1 - smoothstep(0.0, 0.17, Math.abs(n1 + 0.22 * n2));
      const ax = Math.abs(x);
      const sylvian = (1 - smoothstep(0, 0.035, Math.abs(y - sylvianY(z)))) * smoothstep(0.3, 0.42, ax) * smoothstep(-0.3, -0.15, z) * (1 - smoothstep(0.5, 0.6, z));
      const cz = centralZ(y) + 0.03 * noise(y * 6, x * 6, 3.3);
      const central = (1 - smoothstep(0, 0.022, Math.abs(z - cz))) * smoothstep(sylvianY(z), sylvianY(z) + 0.08, y) * 0.95;
      const fissure = (1 - smoothstep(0, 0.02, Math.abs(x) - FISSURE)) * smoothstep(0.2, 0.45, y) * 0.3;
      groove = Math.max(groove, sylvian, central, fissure);
      if (random() < groove * 0.68) continue;
      x -= nx * 0.032 * groove;
      y -= ny * 0.032 * groove;
      z -= nz * 0.032 * groove;
      shade = 0.36 + 0.64 * (1 - groove);
      put(x, y, z, nx, ny, nz, shade, classifyCortex(x, y, z), 0.75 + random() * 0.55);
    } else if (component === CEREBELLUM) {
      const folia = 0.5 + 0.5 * Math.sin(y * 95 + z * 18 + noise(x * 4, y * 4, z * 4) * 2.5);
      if (random() < folia * 0.45) continue;
      shade = 0.42 + 0.4 * (1 - folia);
      put(x, y, z, nx, ny, nz, shade, UNMAPPED, 0.6 + random() * 0.4);
    } else {
      put(x, y, z, nx, ny, nz, 0.42 + random() * 0.15, UNMAPPED, 0.6 + random() * 0.4);
    }
  }

  // Hippocampi: C-shaped tubes inside each temporal lobe, head anterior, tail curving up and back.
  const hippo = regionCode("hippocampus");
  const perSide = Math.floor(hippocampus / 2);
  for (const s of [1, -1]) {
    for (let i = 0; i < perSide; i++) {
      const t = random();
      const cx = s * (0.27 - 0.09 * t);
      const cy = -0.25 + 0.2 * t ** 1.6;
      const cz = 0.3 - 0.62 * t;
      const r = 0.05 * (1 - 0.55 * t) * Math.sqrt(random());
      const a = random() * Math.PI * 2;
      const b = random() * Math.PI;
      put(
        cx + r * Math.cos(a) * Math.sin(b),
        cy + r * Math.sin(a) * Math.sin(b),
        cz + r * Math.cos(b),
        0, 0, 0,
        0.5 + random() * 0.25,
        hippo,
        0.8 + random() * 0.5,
      );
    }
  }

  // Faint interior dust for depth.
  let dustTries = 0;
  const dustEnd = n + dust;
  while (n < dustEnd && dustTries < dust * 50) {
    dustTries++;
    const x = -0.8 + random() * 1.6;
    const y = -0.5 + random() * 1.1;
    const z = -0.95 + random() * 1.9;
    if (field(x, y, z)[0] < -0.06) put(x, y, z, 0, 0, 0, 0.1 + random() * 0.08, UNMAPPED, 0.5 + random() * 0.3);
  }

  return {
    count: n,
    positions: positions.subarray(0, n * 3),
    normals: normals.subarray(0, n * 3),
    brightness: brightness.subarray(0, n),
    region: region.subarray(0, n),
    size: size.subarray(0, n),
    regionIndices,
  };
}

// ------------------------------------------------------------------ placing clusters and records

export interface Placement {
  anchors: Map<string, [number, number, number]>;
  nodes: Map<string, [number, number, number]>;
}

const dist2 = (p: Float32Array, i: number, q: [number, number, number]) => {
  const dx = p[i * 3] - q[0];
  const dy = p[i * 3 + 1] - q[1];
  const dz = p[i * 3 + 2] - q[2];
  return dx * dx + dy * dy + dz * dz;
};

const at = (p: Float32Array, i: number): [number, number, number] => [p[i * 3], p[i * 3 + 1], p[i * 3 + 2]];

/**
 * Spread cluster anchors over their region (farthest-point sampling, seeded by cluster id) and
 * put each record onto real surface particles of its cluster's cell, nearest to the anchor
 * first. Deterministic for the same inputs.
 */
export function placeClusters(
  cloud: BrainCloud,
  clusters: { id: string; region: RegionId; record_ids: string[] }[],
  ownRecords: (clusterId: string) => string[],
): Placement {
  const anchors = new Map<string, [number, number, number]>();
  const nodes = new Map<string, [number, number, number]>();
  const byRegion = new Map<number, typeof clusters>();
  for (const c of clusters) {
    const code = regionCode(c.region);
    if (!byRegion.has(code)) byRegion.set(code, []);
    byRegion.get(code)!.push(c);
  }
  for (const [code, list] of byRegion) {
    const pool = cloud.regionIndices[code].filter((i) => cloud.brightness[i] > 0.6 || code === regionCode("hippocampus"));
    if (pool.length === 0) continue;
    const sorted = [...list].sort((a, b) => (a.id < b.id ? -1 : 1));
    const stride = Math.max(1, Math.floor(pool.length / 700));
    const candidates = pool.filter((_, k) => k % stride === 0);
    const chosen: number[] = [candidates[hashString(sorted[0].id) % candidates.length]];
    const best = candidates.map((i) => dist2(cloud.positions, i, at(cloud.positions, chosen[0])));
    while (chosen.length < sorted.length) {
      let arg = 0;
      for (let k = 1; k < candidates.length; k++) if (best[k] > best[arg]) arg = k;
      chosen.push(candidates[arg]);
      const q = at(cloud.positions, candidates[arg]);
      for (let k = 0; k < candidates.length; k++) best[k] = Math.min(best[k], dist2(cloud.positions, candidates[k], q));
    }
    const anchorPos = chosen.map((i) => at(cloud.positions, i));
    sorted.forEach((c, k) => anchors.set(c.id, anchorPos[k]));

    // Voronoi cells over the region's particles, then fill each cluster outward from its anchor.
    const cells: number[][] = sorted.map(() => []);
    for (const i of pool) {
      let arg = 0;
      let bestD = Infinity;
      for (let k = 0; k < anchorPos.length; k++) {
        const d = dist2(cloud.positions, i, anchorPos[k]);
        if (d < bestD) {
          bestD = d;
          arg = k;
        }
      }
      cells[arg].push(i);
    }
    sorted.forEach((c, k) => {
      const members = ownRecords(c.id);
      if (!members.length) return;
      const cell = cells[k].sort((a, b) => dist2(cloud.positions, a, anchorPos[k]) - dist2(cloud.positions, b, anchorPos[k]));
      const step = Math.max(1, Math.min(6, Math.floor(cell.length / (members.length * 1.5))));
      members.forEach((rid, m) => {
        const idx = cell.length ? cell[Math.min(cell.length - 1, m * step)] : chosen[k];
        const p = at(cloud.positions, idx);
        const nx = cloud.normals[idx * 3], ny = cloud.normals[idx * 3 + 1], nz = cloud.normals[idx * 3 + 2];
        nodes.set(rid, [p[0] + nx * 0.012, p[1] + ny * 0.012, p[2] + nz * 0.012]);
      });
    });
  }
  return { anchors, nodes };
}

/** Mean position of a region's particles; `rightOnly` keeps the hemisphere facing the default camera. */
export function regionCentroid(cloud: BrainCloud, code: number, rightOnly = false): [number, number, number] {
  const idx = rightOnly ? cloud.regionIndices[code].filter((i) => cloud.positions[i * 3] > 0) : cloud.regionIndices[code];
  let x = 0, y = 0, z = 0;
  for (const i of idx) {
    x += cloud.positions[i * 3];
    y += cloud.positions[i * 3 + 1];
    z += cloud.positions[i * 3 + 2];
  }
  const k = Math.max(idx.length, 1);
  return [x / k, y / k, z / k];
}
