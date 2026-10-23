import { describe, expect, it } from "vitest";

import { REGION_CODES, UNMAPPED, classifyCortex, generateBrain, placeClusters, regionCode } from "./brainModel";

const small = () => generateBrain({ surface: 6000, hippocampus: 400, dust: 200, seed: 3 });

describe("generateBrain", () => {
  it("is deterministic for a seed", () => {
    const a = small();
    const b = small();
    expect(a.count).toBe(b.count);
    expect(Array.from(a.positions.slice(0, 300))).toEqual(Array.from(b.positions.slice(0, 300)));
  });

  it("fills every memory region and stays inside the brain's bounding box", () => {
    const cloud = small();
    expect(cloud.count).toBe(6600);
    for (const code of REGION_CODES.map(regionCode)) expect(cloud.regionIndices[code].length).toBeGreaterThan(50);
    for (let i = 0; i < cloud.count; i++) {
      const [x, y, z] = [cloud.positions[i * 3], cloud.positions[i * 3 + 1], cloud.positions[i * 3 + 2]];
      expect(Math.abs(x)).toBeLessThan(1);
      expect(y).toBeGreaterThan(-1.15);
      expect(y).toBeLessThan(0.75);
      expect(Math.abs(z)).toBeLessThan(1.1);
    }
  });

  it("keeps the two hemispheres apart at the longitudinal fissure", () => {
    const cloud = small();
    let nearMidlineOnTop = 0;
    for (let i = 0; i < cloud.count; i++) {
      if (Math.abs(cloud.positions[i * 3]) < 0.01 && cloud.positions[i * 3 + 1] > 0.4) nearMidlineOnTop++;
    }
    expect(nearMidlineOnTop).toBe(0);
  });

  it("puts the hippocampus inside the brain, symmetric left and right", () => {
    const cloud = small();
    const hippo = cloud.regionIndices[regionCode("hippocampus")];
    const left = hippo.filter((i) => cloud.positions[i * 3] < 0).length;
    expect(left / hippo.length).toBeGreaterThan(0.4);
    expect(left / hippo.length).toBeLessThan(0.6);
    for (const i of hippo) expect(cloud.normals[i * 3] + cloud.normals[i * 3 + 1] + cloud.normals[i * 3 + 2]).toBe(0);
  });
});

describe("classifyCortex", () => {
  it("maps landmark positions to the expected regions", () => {
    expect(classifyCortex(0.3, 0.1, 0.85)).toBe(regionCode("prefrontal"));
    expect(classifyCortex(0.3, 0.45, 0.25)).toBe(regionCode("frontal"));
    expect(classifyCortex(0.55, -0.3, 0.1)).toBe(regionCode("temporal"));
    expect(classifyCortex(0.3, 0.0, -0.85)).toBe(regionCode("occipital"));
    expect(classifyCortex(0.3, 0.5, -0.2)).toBe(UNMAPPED); // parietal: not a memory region
  });
});

describe("placeClusters", () => {
  const cloud = small();
  const clusters = [
    { id: "hippocampus:fact", region: "hippocampus" as const, record_ids: ["a", "b", "c"] },
    { id: "hippocampus:literature", region: "hippocampus" as const, record_ids: ["d"] },
    { id: "prefrontal:p", region: "prefrontal" as const, record_ids: ["e", "f"] },
  ];
  const members = (id: string) => clusters.find((c) => c.id === id)!.record_ids;

  it("is deterministic and gives each cluster its own anchor", () => {
    const a = placeClusters(cloud, clusters, members);
    const b = placeClusters(cloud, clusters, members);
    expect([...a.anchors.entries()]).toEqual([...b.anchors.entries()]);
    expect(a.anchors.get("hippocampus:fact")).not.toEqual(a.anchors.get("hippocampus:literature"));
  });

  it("places every record on a distinct spot inside its region", () => {
    const { nodes } = placeClusters(cloud, clusters, members);
    expect([...nodes.keys()].sort()).toEqual(["a", "b", "c", "d", "e", "f"]);
    const spots = new Set([...nodes.values()].map((p) => p.map((v) => v.toFixed(4)).join(",")));
    expect(spots.size).toBe(6);
    const [x, , z] = nodes.get("e")!;
    expect(z).toBeGreaterThan(0.3); // prefrontal records sit at the front
    expect(Math.abs(x)).toBeLessThan(1);
  });
});
