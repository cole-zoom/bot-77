// A's training rows exactly as the model receives them, and each feature's distribution per split: train vs held-back
// vs test should look alike (a feature that differs a lot by split is a leak or a broken feature).
import { useState } from "react";
import { api, get } from "../api/client";
import { Card, Loading, Pill, SPLIT_COLOR, fmt, useLoad } from "../ui";

export default function Training() {
  const [split, setSplit] = useState("heldback");
  const { data, error } = useLoad(() => get(api.GET("/api/training/card", { params: { query: { sample_split: split, sample_n: 25 } } })), [split]);
  if (!data) return <Loading error={error} />;
  const splits = data.decisions_by_split.map((d) => d.key);
  const feats = [...new Set(data.stats.map((s) => s.name))];
  const cols = data.sample[0] ? Object.keys(data.sample[0]) : [];
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Training data: card model</h1>
      <Card>
        <div className="flex flex-wrap gap-4 text-sm">
          <span className="text-zinc-400">table <span className="font-mono text-zinc-200">{data.table}</span> (rebuilt from moments)</span>
          {data.decisions_by_split.map((d) => <span key={d.key}><Pill text={d.key} cls={SPLIT_COLOR[d.key]} /> <span className="num">{d.n.toLocaleString()}</span> decisions</span>)}
        </div>
      </Card>
      <Card title="Feature distributions by split (p5 / median / p95, % missing)">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-left text-zinc-500"><tr><th className="py-1">feature</th>{splits.map((s) => <th key={s} className="px-2">{s}</th>)}</tr></thead>
            <tbody>
              {feats.map((f) => (
                <tr key={f} className="border-t border-zinc-800">
                  <td className="py-1 pr-3 font-mono">{f}</td>
                  {splits.map((s) => {
                    const st = data.stats.find((x) => x.name === f && x.split === s);
                    return <td key={s} className="num px-2 text-zinc-300">{!st ? "" : st.categorical ? `${st.distinct} cards · top ${st.top ?? "—"}` : `${fmt(st.p5)} / ${fmt(st.p50)} / ${fmt(st.p95)}`}
                      {st && st.missing > 0 ? <span className="text-amber-400"> · {Math.round(st.missing * 100)}%</span> : null}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <Card title="Rows as the model gets them (4 per decision; chosen = the card played)">
        <div className="mb-2 flex gap-2 text-sm">{splits.map((s) => <button key={s} onClick={() => setSplit(s)} className={`rounded px-2 py-0.5 ${s === split ? "bg-violet-700" : "bg-zinc-800"}`}>{s}</button>)}</div>
        <div className="max-h-[50vh] overflow-auto">
          <table className="text-[11px]">
            <thead className="sticky top-0 bg-zinc-900 text-left text-zinc-500"><tr>{cols.map((c) => <th key={c} className="px-1.5 py-1 font-mono">{c}</th>)}</tr></thead>
            <tbody>
              {data.sample.map((r, i) => (
                <tr key={i} className={`border-t border-zinc-800 ${(r as any).chosen ? "bg-emerald-950/40" : ""} ${i % 4 === 0 ? "border-t-zinc-600" : ""}`}>
                  {cols.map((c) => { const v = (r as any)[c]; return <td key={c} className="num px-1.5 py-0.5 text-zinc-300">{typeof v === "number" ? +v.toFixed(2) : String(v)}</td>; })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
