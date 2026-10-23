// Small shared pieces.
import { useEffect, useState } from "react";

export function useLoad<T>(fn: () => Promise<T>, deps: unknown[]): { data?: T; error?: string; loading: boolean } {
  const [s, set] = useState<{ data?: T; error?: string; loading: boolean }>({ loading: true });
  useEffect(() => {
    let live = true;
    set((p) => ({ ...p, loading: true }));
    fn().then((data) => live && set({ data, loading: false }))
      .catch((e) => live && set({ error: String(e), loading: false }));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return s;
}

export function Card({ title, children, className = "" }: { title?: string; children: React.ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 ${className}`}>
      {title && <h2 className="mb-3 text-sm font-medium text-zinc-300">{title}</h2>}
      {children}
    </section>
  );
}

export function Loading({ error }: { error?: string }) {
  return <div className={`text-sm ${error ? "text-red-400" : "text-zinc-500"}`}>{error ?? "Loading…"}</div>;
}

export function Bar({ value, max }: { value: number; max: number }) {
  return (
    <div className="h-1.5 w-full rounded bg-zinc-800">
      <div className="h-1.5 rounded bg-violet-500" style={{ width: `${Math.min(100, (100 * value) / Math.max(max, 1e-9))}%` }} />
    </div>
  );
}

export const fmt = (n: number | null | undefined, d = 1) => (n == null ? "—" : n.toFixed(d));
export const clock = (t: number | null | undefined) =>
  t == null ? "—" : `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

export const SPLIT_COLOR: Record<string, string> = {
  train: "bg-zinc-700 text-zinc-200", heldback: "bg-amber-900/60 text-amber-200", test: "bg-red-900/60 text-red-200",
  heldout: "bg-rose-950 text-rose-200", dev: "bg-sky-900/60 text-sky-200",
};
export function Pill({ text, cls = "bg-zinc-800 text-zinc-300" }: { text: string; cls?: string }) {
  return <span className={`rounded px-1.5 py-0.5 text-xs ${cls}`}>{text}</span>;
}
