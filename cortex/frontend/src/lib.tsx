import { type DependencyList, Fragment, useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "./api";

export function useAsync<T>(fn: () => Promise<T>, deps: DependencyList) {
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  useEffect(() => {
    let live = true;
    setLoading(true);
    fnRef
      .current()
      .then((value) => {
        if (live) {
          setData(value);
          setError(null);
        }
      })
      .catch((e: unknown) => live && setError(errorText(e)))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);
  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { data, error, loading, reload, setData };
}

export function errorText(e: unknown): string {
  if (e instanceof ApiError) return e.status === 409 ? `Not saved: ${e.message}` : e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

export function fmtDate(iso: string | null | undefined, withTime = false): string {
  if (!iso) return "—";
  if (/^\d{4}-\d{2}-\d{2}$/.test(iso)) return iso;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, "0");
  const day = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  return withTime ? `${day} ${pad(d.getHours())}:${pad(d.getMinutes())}` : day;
}

export function ago(iso: string, now = Date.now()): string {
  const s = Math.max(0, Math.round((now - new Date(iso).getTime()) / 1000));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
}

export const SNIPPET_OPEN = "\u0002";
export const SNIPPET_CLOSE = "\u0003";

/** Search snippet with FTS match markers -> text with <mark>; never interpreted as HTML. */
export function Snippet({ text }: { text: string }) {
  const parts: { text: string; hit: boolean }[] = [];
  let hit = false;
  let buf = "";
  for (const ch of text) {
    if (ch === SNIPPET_OPEN || ch === SNIPPET_CLOSE) {
      if (buf) parts.push({ text: buf, hit });
      buf = "";
      hit = ch === SNIPPET_OPEN;
    } else buf += ch;
  }
  if (buf) parts.push({ text: buf, hit });
  return (
    <>
      {parts.map((p, i) => (p.hit ? <mark key={i}>{p.text}</mark> : <Fragment key={i}>{p.text}</Fragment>))}
    </>
  );
}

export function confirmTyped(message: string, word: string): boolean {
  const answer = window.prompt(`${message}\n\nType "${word}" to confirm.`);
  return answer?.trim().toLowerCase() === word.toLowerCase();
}
