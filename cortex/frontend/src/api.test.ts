import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, buildUrl } from "./api";

afterEach(() => vi.unstubAllGlobals());

function stubFetch(status: number, body: unknown) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: status < 400,
    status,
    statusText: "x",
    json: () => Promise.resolve(body),
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("buildUrl", () => {
  it("repeats array params and drops empty ones", () => {
    expect(buildUrl("/search", { q: "imk", category: ["facts", "sessions"], tag: [], project_id: undefined, verified: false })).toBe(
      "/api/search?q=imk&category=facts&category=sessions&verified=false",
    );
  });
});

describe("request", () => {
  it("marks every call with the client header and sends JSON bodies", async () => {
    const fetchMock = stubFetch(201, { id: "r1" });
    await api.createRecord({ title: "T", category: "inbox" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/records");
    expect(init.method).toBe("POST");
    expect(init.headers["X-Cortex-Client"]).toBe("1");
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(init.body)).toEqual({ title: "T", category: "inbox" });
  });

  it("surfaces the server's validation message", async () => {
    stubFetch(422, { detail: [{ loc: ["body", "category"], msg: "Input should be 'facts'" }] });
    await expect(api.createRecord({ title: "T", category: "inbox" })).rejects.toThrow("category: Input should be 'facts'");
  });

  it("reports stale writes as 409 errors", async () => {
    stubFetch(409, { detail: "record was changed elsewhere; reload before saving" });
    const err = await api.updateRecord("r1", { title: "x", expected_updated_at: "t" }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(409);
  });

  it("asks for confirmation id on permanent deletion", async () => {
    const fetchMock = stubFetch(200, { purged: "abc" });
    await api.purgeRecord("abc");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/records/abc/purge?confirm=abc");
    expect(fetchMock.mock.calls[0][1].method).toBe("DELETE");
  });
});
