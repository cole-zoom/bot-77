// Browse matches: filter by source, split, text (match id, opponent, card in deck).
import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, get } from "../api/client";
import { Card, Loading, Pill, SPLIT_COLOR, useLoad } from "../ui";

const SOURCES = ["", "replay_tv", "replay_bl", "replay_bot", "pov_cole", "pov_friend", "youtube", "bot"];
const SPLITS = ["", "train", "heldback", "test", "heldout", "dev"];

export default function Matches() {
  const [sp, setSp] = useSearchParams();
  const source = sp.get("source") ?? "", split = sp.get("split") ?? "", text = sp.get("text") ?? "";
  const page = Number(sp.get("page") ?? 0);
  const [draft, setDraft] = useState(text);
  const set = (k: string, v: string) => { const n = new URLSearchParams(sp); v ? n.set(k, v) : n.delete(k); n.delete("page"); setSp(n); };
  const { data, error, loading } = useLoad(() => get(api.GET("/api/matches", {
    params: { query: { source: source || undefined, split: split || undefined, text: text || undefined, limit: 50, offset: page * 50 } },
  })), [source, split, text, page]);
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Matches</h1>
      <Card>
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <select className="rounded bg-zinc-800 px-2 py-1" value={source} onChange={(e) => set("source", e.target.value)}>
            {SOURCES.map((s) => <option key={s} value={s}>{s || "any source"}</option>)}
          </select>
          <select className="rounded bg-zinc-800 px-2 py-1" value={split} onChange={(e) => set("split", e.target.value)}>
            {SPLITS.map((s) => <option key={s} value={s}>{s || "any split"}</option>)}
          </select>
          <form onSubmit={(e) => { e.preventDefault(); set("text", draft); }}>
            <input className="w-64 rounded bg-zinc-800 px-2 py-1" placeholder="match id, opponent or card…" value={draft} onChange={(e) => setDraft(e.target.value)} />
          </form>
          {data && <span className="text-zinc-500">{data.total.toLocaleString()} matches</span>}
        </div>
      </Card>
      {!data ? <Loading error={error} /> : (
        <Card className={loading ? "opacity-60" : ""}>
          <table className="w-full text-sm">
            <thead className="text-left text-zinc-500"><tr><th className="py-1">match</th><th>source</th><th>split</th><th>league</th><th>result</th><th>deck (bottom player)</th></tr></thead>
            <tbody>
              {data.rows.map((m) => (
                <tr key={m.match_id} className="border-t border-zinc-800 hover:bg-zinc-800/40">
                  <td className="py-1.5 pr-3"><Link className="font-mono text-xs text-violet-300 hover:underline" to={`/matches/${m.match_id}`}>{m.match_id}</Link></td>
                  <td className="pr-3"><Pill text={m.source ?? "?"} /></td>
                  <td className="pr-3"><Pill text={m.split ?? "?"} cls={SPLIT_COLOR[m.split ?? ""]} /></td>
                  <td className="num pr-3 text-zinc-400">{m.league ?? ""}</td>
                  <td className="pr-3">{m.result && <Pill text={m.result} cls={m.result === "win" ? "bg-emerald-900/60 text-emerald-200" : "bg-red-900/60 text-red-200"} />}</td>
                  <td className="text-xs text-zinc-400">{(m.deck_names ?? []).join(" · ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-3 flex gap-2 text-sm">
            <button disabled={page === 0} className="rounded bg-zinc-800 px-2 py-1 disabled:opacity-40" onClick={() => setSp((p) => { const n = new URLSearchParams(p); n.set("page", String(page - 1)); return n; })}>← prev</button>
            <button disabled={(page + 1) * 50 >= data.total} className="rounded bg-zinc-800 px-2 py-1 disabled:opacity-40" onClick={() => setSp((p) => { const n = new URLSearchParams(p); n.set("page", String(page + 1)); return n; })}>next →</button>
          </div>
        </Card>
      )}
    </div>
  );
}
