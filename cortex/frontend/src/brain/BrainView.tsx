import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { type RefObject, useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

import type { BrainMap, Cluster, RegionId } from "../types";
import { REGION_META, REGION_ORDER } from "../types";
import { type BrainCloud, type Placement, REGION_CODES, UNMAPPED, regionCentroid, regionCode } from "./brainModel";
import { brainVertex, nodeVertex, pointFragment } from "./shaders";

export type ViewPreset = "3d" | "axial" | "sagittal" | "coronal";

/** Unit directions; the distance is fitted to the viewport's aspect ratio. */
export const VIEW_DIRECTIONS: Record<ViewPreset, [number, number, number]> = {
  "3d": [0.68, 0.32, 0.66],
  axial: [0, 1, -0.0001],
  sagittal: [1, 0.015, 0],
  coronal: [0, 0.015, 1],
};
const TARGET = new THREE.Vector3(0, -0.08, 0);

export interface BrainViewProps {
  cloud: BrainCloud;
  map: BrainMap | null;
  placement: Placement;
  view: ViewPreset;
  viewNonce: number;
  selectedRegion: RegionId | null;
  hoverRegion: RegionId | null;
  selectedClusterId: string | null;
  selectedRecordId: string | null;
  highlightIds: Set<string>;
  onHoverRegion: (r: RegionId | null) => void;
  onSelectRegion: (r: RegionId | null) => void;
  onSelectCluster: (c: Cluster) => void;
  onSelectRecord: (id: string) => void;
}

interface Label {
  key: string;
  position: [number, number, number];
  kind: "region" | "cluster" | "axis";
  text: string;
  sub?: string;
  count?: number;
  active?: boolean;
  hover?: boolean;
  onClick?: () => void;
}

function useGeometry(build: () => THREE.BufferGeometry, deps: unknown[]): THREE.BufferGeometry {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const geometry = useMemo(build, deps);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return geometry;
}

function BrainPoints({ cloud, selectedRegion, hoverRegion, onHoverRegion, onSelectRegion }: Pick<BrainViewProps, "cloud" | "selectedRegion" | "hoverRegion" | "onHoverRegion" | "onSelectRegion">) {
  const geometry = useGeometry(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(cloud.positions, 3));
    g.setAttribute("normal", new THREE.BufferAttribute(cloud.normals, 3));
    g.setAttribute("aBrightness", new THREE.BufferAttribute(cloud.brightness, 1));
    g.setAttribute("aRegion", new THREE.BufferAttribute(cloud.region, 1));
    g.setAttribute("aSize", new THREE.BufferAttribute(cloud.size, 1));
    g.computeBoundingSphere();
    return g;
  }, [cloud]);
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: brainVertex,
        fragmentShader: pointFragment,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uSelected: { value: -1 },
          uHover: { value: -1 },
          uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
          uScale: { value: 9 },
        },
      }),
    [],
  );
  useEffect(() => {
    material.uniforms.uSelected.value = selectedRegion ? regionCode(selectedRegion) : -1;
    material.uniforms.uHover.value = hoverRegion ? regionCode(hoverRegion) : -1;
  }, [material, selectedRegion, hoverRegion]);
  useEffect(() => () => material.dispose(), [material]);

  const regionAt = (e: ThreeEvent<PointerEvent | MouseEvent>): RegionId | null => {
    const code = e.index !== undefined ? cloud.region[e.index] : UNMAPPED;
    return code === UNMAPPED ? null : REGION_CODES[code];
  };
  return (
    <points
      geometry={geometry}
      material={material}
      onPointerMove={(e) => {
        e.stopPropagation();
        onHoverRegion(regionAt(e));
      }}
      onPointerOut={() => onHoverRegion(null)}
      onClick={(e) => {
        e.stopPropagation();
        const r = regionAt(e);
        if (r) onSelectRegion(r);
      }}
    />
  );
}

function NodePoints({ map, placement, selectedRegion, highlightIds, onSelectRecord }: Pick<BrainViewProps, "map" | "placement" | "selectedRegion" | "highlightIds" | "onSelectRecord">) {
  const nodes = useMemo(() => (map?.nodes ?? []).filter((n) => placement.nodes.has(n.id)), [map, placement]);
  const geometry = useGeometry(() => {
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(nodes.length * 3);
    const reg = new Float32Array(nodes.length);
    nodes.forEach((n, i) => {
      pos.set(placement.nodes.get(n.id)!, i * 3);
      reg[i] = regionCode(n.region);
    });
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("aRegion", new THREE.BufferAttribute(reg, 1));
    g.setAttribute("aHighlight", new THREE.BufferAttribute(new Float32Array(nodes.length), 1));
    g.computeBoundingSphere();
    return g;
  }, [nodes, placement]);
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: nodeVertex,
        fragmentShader: pointFragment,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uSelected: { value: -1 },
          uAnyHighlight: { value: 0 },
          uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
          uScale: { value: 30 },
          uTime: { value: 0 },
        },
      }),
    [],
  );
  useEffect(() => () => material.dispose(), [material]);
  useEffect(() => {
    const attr = geometry.getAttribute("aHighlight") as THREE.BufferAttribute;
    nodes.forEach((n, i) => attr.setX(i, highlightIds.has(n.id) ? 1 : 0));
    attr.needsUpdate = true;
    material.uniforms.uAnyHighlight.value = highlightIds.size > 0 ? 1 : 0;
    material.uniforms.uSelected.value = selectedRegion ? regionCode(selectedRegion) : -1;
  }, [geometry, nodes, highlightIds, selectedRegion, material]);
  useFrame(({ clock }) => {
    material.uniforms.uTime.value = clock.elapsedTime;
  });
  return (
    <points
      geometry={geometry}
      material={material}
      renderOrder={2}
      onClick={(e) => {
        e.stopPropagation();
        if (e.index !== undefined && nodes[e.index]) onSelectRecord(nodes[e.index].id);
      }}
      onPointerOver={() => (document.body.style.cursor = "pointer")}
      onPointerOut={() => (document.body.style.cursor = "")}
    />
  );
}

/**
 * Relationship arcs, bent through the interior so they read as connections, not chords.
 * Unfocused links stay a faint haze; links of the open record or cluster light up.
 */
function EdgeLines({ map, placement, selectedRecordId, highlightIds, extra }: Pick<BrainViewProps, "map" | "placement" | "selectedRecordId" | "highlightIds"> & { extra: [number, number, number][][] }) {
  const geometry = useGeometry(() => {
    const segments = 12;
    const edges = (map?.edges ?? []).filter((e) => placement.nodes.has(e.source) && placement.nodes.has(e.target));
    const pos: number[] = [];
    const col: number[] = [];
    const curve = (a: [number, number, number], b: [number, number, number], color: [number, number, number]) => {
      const c: [number, number, number] = [(a[0] + b[0]) * 0.3, (a[1] + b[1]) * 0.3 - 0.05, (a[2] + b[2]) * 0.3];
      let prev = a;
      for (let s = 1; s <= segments; s++) {
        const t = s / segments;
        const u = 1 - t;
        const p: [number, number, number] = [
          u * u * a[0] + 2 * u * t * c[0] + t * t * b[0],
          u * u * a[1] + 2 * u * t * c[1] + t * t * b[1],
          u * u * a[2] + 2 * u * t * c[2] + t * t * b[2],
        ];
        // Brighter near the endpoints, dim through the middle of the arc.
        const k = 0.45 + 0.55 * Math.abs(1 - 2 * t);
        pos.push(...prev, ...p);
        col.push(color[0] * k, color[1] * k, color[2] * k, color[0] * k, color[1] * k, color[2] * k);
        prev = p;
      }
    };
    const anyFocus = selectedRecordId !== null || highlightIds.size > 0;
    for (const e of edges) {
      const touches = e.source === selectedRecordId || e.target === selectedRecordId;
      const both = highlightIds.has(e.source) && highlightIds.has(e.target);
      const color: [number, number, number] = touches ? [0.62, 0.9, 0.17] : both ? [0.26, 0.37, 0.08] : anyFocus ? [0.005, 0.005, 0.006] : [0.011, 0.012, 0.014];
      curve(placement.nodes.get(e.source)!, placement.nodes.get(e.target)!, color);
    }
    for (const [a, b] of extra) curve(a, b, [0.62, 0.9, 0.17]);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    return g;
  }, [map, placement, selectedRecordId, highlightIds, extra]);
  return (
    <lineSegments geometry={geometry} renderOrder={1}>
      <lineBasicMaterial vertexColors transparent depthWrite={false} blending={THREE.AdditiveBlending} />
    </lineSegments>
  );
}

function ClusterMarkers({ map, placement, selectedRegion, selectedClusterId, onSelectCluster }: Pick<BrainViewProps, "map" | "placement" | "selectedRegion" | "selectedClusterId" | "onSelectCluster">) {
  const region = map?.regions.find((r) => r.region === selectedRegion);
  if (!region) return null;
  return (
    <>
      {region.clusters.map((c) => {
        const p = placement.anchors.get(c.id);
        if (!p) return null;
        const active = c.id === selectedClusterId;
        return (
          <mesh
            key={c.id}
            position={p}
            onClick={(e) => {
              e.stopPropagation();
              onSelectCluster(c);
            }}
          >
            <sphereGeometry args={[0.02 + Math.min(0.03, Math.sqrt(c.count) * 0.004), 12, 12]} />
            <meshBasicMaterial color={active ? "#c6ff3d" : "#86b02a"} transparent opacity={active ? 0.55 : 0.28} depthWrite={false} blending={THREE.AdditiveBlending} />
          </mesh>
        );
      })}
    </>
  );
}

function Frame() {
  const geometry = useGeometry(() => {
    const pos: number[] = [];
    const y = -1.02;
    for (let i = -10; i <= 10; i++) {
      const v = i * 0.12;
      pos.push(-1.2, y, v, 1.2, y, v, v, y, -1.2, v, y, 1.2);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    return g;
  }, []);
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#ffffff" transparent opacity={0.05} depthWrite={false} />
    </lineSegments>
  );
}

/** Keeps the DOM labels glued to their 3D anchors; labels on the far side are dimmed. */
function LabelTracker({ labels, elements }: { labels: Label[]; elements: RefObject<Map<string, HTMLElement>> }) {
  const v = useMemo(() => new THREE.Vector3(), []);
  const toCam = useMemo(() => new THREE.Vector3(), []);
  const center = useMemo(() => new THREE.Vector3(), []);
  useFrame(({ camera, size }) => {
    if (size.width < 2 || size.height < 2) return;
    center.copy(TARGET).project(camera);
    const placed: { x: number; y: number; w: number; h: number }[] = [];
    for (const label of labels) {
      const el = elements.current?.get(label.key);
      if (!el) continue;
      v.set(...label.position);
      const behind = label.kind !== "axis" && v.clone().sub(TARGET).normalize().dot(toCam.copy(camera.position).sub(TARGET).normalize()) < -0.15;
      v.project(camera);
      if (v.z > 1) {
        el.style.visibility = "hidden";
        continue;
      }
      // Region labels are pushed outward from the brain's centre on screen so they do not stack.
      const push = label.kind === "region" ? 1.4 : 1;
      const halfW = el.offsetWidth / 2 + 6;
      const halfH = el.offsetHeight / 2 + 4;
      const px = Math.min(size.width - halfW, Math.max(halfW, ((center.x + (v.x - center.x) * push + 1) / 2) * size.width));
      let py = Math.min(size.height - halfH, Math.max(halfH + 30, ((1 - (center.y + (v.y - center.y) * push)) / 2) * size.height));
      if (label.kind !== "axis") {
        // Greedy de-overlap: earlier labels (active, then largest) keep their spot; later ones step up/down.
        const w = el.offsetWidth;
        const h = el.offsetHeight;
        const collides = (y: number) => placed.some((b) => Math.abs(b.x - px) < (b.w + w) / 2 + 2 && Math.abs(b.y - y) < (b.h + h) / 2 + 1);
        const base = py;
        for (let k = 1; k <= 12 && collides(py); k++) py = base + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * (h + 3);
        placed.push({ x: px, y: py, w, h });
      }
      el.style.visibility = "visible";
      el.style.transform = `translate(-50%, -50%) translate(${px}px, ${py}px)`;
      el.style.opacity = behind && !label.active ? "0.35" : "1";
      el.style.zIndex = behind ? "1" : "2";
    }
  });
  return null;
}

function CameraRig({ view, nonce }: { view: ViewPreset; nonce: number }) {
  const { camera, size } = useThree();
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null;
  const target = useRef<THREE.Vector3 | null>(null);
  useEffect(() => {
    // The canvas can measure 0×0 before layout settles; an aspect of 0 would send the camera to infinity.
    if (size.width < 2 || size.height < 2) return;
    // Narrow viewports need the camera further back so the whole brain fits.
    // fov 38°: visible height ≈ 0.69·d; the brain is ~2.3 long and ~2.1 tall with margins.
    const aspect = size.width / size.height;
    const distance = Math.max(3.3, 3.4 / aspect) * (view === "3d" ? 1.12 : 1);
    target.current = new THREE.Vector3(...VIEW_DIRECTIONS[view]).normalize().multiplyScalar(distance).add(TARGET);
    if (controls) controls.autoRotate = view === "3d" && nonce === 0;
  }, [view, nonce, controls, size.width, size.height]);
  useFrame(() => {
    if (!target.current) return;
    if (![camera.position.x, camera.position.y, camera.position.z].every(Number.isFinite)) camera.position.copy(target.current);
    camera.position.lerp(target.current, 0.12);
    controls?.target.lerp(TARGET, 0.12);
    controls?.update();
    if (camera.position.distanceTo(target.current) < 0.002) target.current = null;
  });
  return null;
}

const AXES: [string, [number, number, number]][] = [
  ["A", [0, -0.08, 1.15]],
  ["P", [0, -0.08, -1.15]],
  ["S", [0, 0.82, 0]],
  ["I", [0, -1.12, 0]],
  ["R", [1.0, -0.08, 0]],
  ["L", [-1.0, -0.08, 0]],
];

export default function BrainView(props: BrainViewProps) {
  const { cloud, map, placement, selectedRegion, hoverRegion, selectedClusterId, onSelectRegion, onSelectCluster } = props;

  // A selected conflict cluster reaches out to the records it involves, wherever they live.
  const extra = useMemo(() => {
    const cluster = map?.regions.flatMap((r) => r.clusters).find((c) => c.id === selectedClusterId);
    const from = cluster && placement.anchors.get(cluster.id);
    if (!cluster || !from || cluster.region !== "frontal") return [];
    return cluster.record_ids.filter((id) => placement.nodes.has(id)).map((id) => [from, placement.nodes.get(id)!]);
  }, [map, placement, selectedClusterId]);

  const centroids = useMemo(
    () =>
      Object.fromEntries(
        REGION_ORDER.map((r) => {
          const c = regionCentroid(cloud, regionCode(r), true);
          const out: [number, number, number] = r === "hippocampus" ? [c[0], c[1] - 0.04, c[2]] : [c[0] * 1.12, c[1] * 1.12 + 0.03, c[2] * 1.12];
          return [r, out];
        }),
      ) as Record<RegionId, [number, number, number]>,
    [cloud],
  );

  const labels = useMemo<Label[]>(() => {
    const out: Label[] = AXES.map(([text, position]) => ({ key: `axis-${text}`, kind: "axis", text, position }));
    const region = map?.regions.find((r) => r.region === selectedRegion);
    if (region) {
      // The largest clusters get labels on the brain; every cluster is listed in the side panel.
      const shown = [...region.clusters]
        .sort((a, b) => Number(b.id === selectedClusterId) - Number(a.id === selectedClusterId) || b.count - a.count)
        .slice(0, 8);
      for (const c of shown) {
        const p = placement.anchors.get(c.id);
        if (p)
          out.push({
            key: c.id,
            kind: "cluster",
            text: c.label,
            count: c.count,
            position: [p[0] * 1.06, p[1] * 1.06 + 0.03, p[2] * 1.06],
            active: c.id === selectedClusterId,
            onClick: () => onSelectCluster(c),
          });
      }
    } else {
      for (const r of REGION_ORDER) {
        out.push({
          key: `region-${r}`,
          kind: "region",
          text: REGION_META[r].label,
          sub: REGION_META[r].anatomy,
          count: map?.regions.find((x) => x.region === r)?.count ?? 0,
          position: centroids[r],
          hover: hoverRegion === r,
          onClick: () => onSelectRegion(r),
        });
      }
    }
    return out;
  }, [map, selectedRegion, selectedClusterId, placement, centroids, hoverRegion, onSelectCluster, onSelectRegion]);

  const elements = useRef(new Map<string, HTMLElement>());

  return (
    <div className="brain-wrap">
      <Canvas
        camera={{ position: [2.8, 1.3, 2.7], fov: 38, near: 0.05, far: 50 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: false }}
        raycaster={{ params: { Points: { threshold: 0.018 }, Line: { threshold: 0 }, Mesh: {}, LOD: {}, Sprite: {} } }}
        onPointerMissed={() => onSelectRegion(null)}
      >
        <color attach="background" args={["#000000"]} />
        <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={1.3} maxDistance={8} autoRotate autoRotateSpeed={0.35} target={TARGET} />
        <CameraRig view={props.view} nonce={props.viewNonce} />
        <BrainPoints {...props} />
        <EdgeLines {...props} extra={extra} />
        <NodePoints {...props} />
        <ClusterMarkers {...props} />
        <Frame />
        <LabelTracker labels={labels} elements={elements} />
      </Canvas>
      <div className="brain-labels">
        {labels.map((l) => {
          const ref = (el: HTMLElement | null) => {
            if (el) elements.current.set(l.key, el);
            else elements.current.delete(l.key);
          };
          if (l.kind === "axis")
            return (
              <span key={l.key} ref={ref} className="axis-mark">
                {l.text}
              </span>
            );
          return (
            <button key={l.key} ref={ref} className={`brain-label ${l.kind}${l.active ? " active" : ""}${l.hover ? " hover" : ""}`} onClick={l.onClick}>
              {l.sub && <b>{l.sub}</b>} {l.text} <span>{l.count}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
