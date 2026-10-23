import { useEffect, useRef } from "react";

import type { RegionId } from "../types";
import { type BrainCloud, type Placement, regionCode } from "./brainModel";
import type { ViewPreset } from "./BrainView";

const SIZE = 132;

type Plane = "axial" | "sagittal" | "coronal";

/** Screen axes per plane: axial looks down (anterior up), sagittal from the right (anterior left), coronal from the front. */
const PROJECT: Record<Plane, (x: number, y: number, z: number) => [number, number]> = {
  axial: (x, _y, z) => [x, -z],
  sagittal: (_x, y, z) => [-z, -y],
  coronal: (x, y) => [-x, -y],
};

function draw(canvas: HTMLCanvasElement, plane: Plane, cloud: BrainCloud, placement: Placement, selected: number, highlight: Set<string>) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const scale = SIZE / 2.3;
  const project = PROJECT[plane];
  const white = new Float32Array(SIZE * SIZE);
  const lime = new Float32Array(SIZE * SIZE);
  const p = cloud.positions;
  for (let i = 0; i < cloud.count; i++) {
    const [u, v] = project(p[i * 3], p[i * 3 + 1] + 0.08, p[i * 3 + 2]);
    const px = Math.floor(SIZE / 2 + u * scale);
    const py = Math.floor(SIZE / 2 + v * scale);
    if (px < 0 || py < 0 || px >= SIZE || py >= SIZE) continue;
    const b = cloud.brightness[i] * 0.07;
    if (selected >= 0 && cloud.region[i] === selected) lime[py * SIZE + px] += b * 1.6;
    else white[py * SIZE + px] += b * (selected >= 0 ? 0.55 : 1);
  }
  for (const [id, q] of placement.nodes) {
    const [u, v] = project(q[0], q[1] + 0.08, q[2]);
    const px = Math.floor(SIZE / 2 + u * scale);
    const py = Math.floor(SIZE / 2 + v * scale);
    if (px < 0 || py < 0 || px >= SIZE || py >= SIZE) continue;
    if (highlight.has(id)) lime[py * SIZE + px] += 2;
    else white[py * SIZE + px] += 0.6;
  }
  const img = ctx.createImageData(SIZE, SIZE);
  for (let k = 0; k < SIZE * SIZE; k++) {
    const w = Math.min(1, white[k]);
    const l = Math.min(1, lime[k]);
    img.data[k * 4] = Math.min(255, (w * 0.92 + l * 0.76) * 255);
    img.data[k * 4 + 1] = Math.min(255, (w * 0.94 + l * 1.0) * 255);
    img.data[k * 4 + 2] = Math.min(255, (w * 0.97 + l * 0.24) * 255);
    img.data[k * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(SIZE / 2 + 0.5, 0);
  ctx.lineTo(SIZE / 2 + 0.5, SIZE);
  ctx.moveTo(0, SIZE / 2 + 0.5);
  ctx.lineTo(SIZE, SIZE / 2 + 0.5);
  ctx.stroke();
}

export default function SliceViews({
  cloud,
  placement,
  selectedRegion,
  highlightIds,
  view,
  onView,
}: {
  cloud: BrainCloud;
  placement: Placement;
  selectedRegion: RegionId | null;
  highlightIds: Set<string>;
  view: ViewPreset;
  onView: (v: ViewPreset) => void;
}) {
  const refs = useRef<Record<Plane, HTMLCanvasElement | null>>({ axial: null, sagittal: null, coronal: null });
  useEffect(() => {
    const selected = selectedRegion ? regionCode(selectedRegion) : -1;
    (Object.keys(refs.current) as Plane[]).forEach((plane) => {
      const canvas = refs.current[plane];
      if (canvas) draw(canvas, plane, cloud, placement, selected, highlightIds);
    });
  }, [cloud, placement, selectedRegion, highlightIds]);
  return (
    <div className="slices">
      {(["axial", "sagittal", "coronal"] as Plane[]).map((plane) => (
        <button key={plane} className={`slice${view === plane ? " active" : ""}`} onClick={() => onView(plane)} title={`${plane} view`}>
          <canvas
            ref={(el) => {
              refs.current[plane] = el;
            }}
            width={SIZE}
            height={SIZE}
          />
          <span>{plane.toUpperCase()}</span>
        </button>
      ))}
    </div>
  );
}
