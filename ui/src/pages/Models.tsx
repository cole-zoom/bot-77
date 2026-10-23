// The models registry (bot77.models): every model version, what it was trained on, its scores, and which one each
// role uses. Old versions stay listed as the history.
import { Fragment, useState } from "react";
import { api, get } from "../api/client";
import { Card, Loading, Pill, useLoad } from "../ui";

const STATUS: Record<string, string> = {
  adopted: "bg-emerald-900/70 text-emerald-200", superseded: "bg-zinc-800 text-zinc-400",
  "not adopted": "bg-amber-950 text-amber-300", reference: "bg-sky-950 text-sky-300",
};

export default function Models() {
  const { data, error } = useLoad(() => get(api.GET("/api/models")), []);
  const [open, setOpen] = useState<string | null>(null);
  if (!data) return <Loading error={error} />;
  const families = [...new Set(data.map((m) => m.family))];
  const inUse = data.filter((m) => m.adopted);
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Models</h1>
      <Card title="In use (adopted for each role)">
        <div className="flex flex-wrap gap-2 text-sm">
          {inUse.map((m) => <span key={m.model_id} className="rounded bg-zinc-800 px-2 py-1"><span className="text-zinc-400">{m.role}</span> <span className="font-mono">{m.version}</span></span>)}
        </div>
      </Card>
      {families.map((f) => (
        <Card key={f} title={f}>
          <table className="w-full text-sm">
            <thead className="text-left text-zinc-500"><tr><th className="py-1">version</th><th>status</th><th>role</th><th>trained</th><th>what</th><th>scores</th></tr></thead>
            <tbody>
              {data.filter((m) => m.family === f).map((m) => (
                <Fragment key={m.model_id}>
                  <tr className="cursor-pointer border-t border-zinc-800 align-top hover:bg-zinc-800/40" onClick={() => setOpen(open === m.model_id ? null : m.model_id)}>
                    <td className="py-1.5 pr-3 font-mono text-xs">{m.version}{!m.on_disk && <span className="text-red-400"> (missing)</span>}</td>
                    <td className="pr-3"><Pill text={m.status} cls={STATUS[m.status]} /></td>
                    <td className="pr-3 text-xs text-zinc-400">{m.role}</td>
                    <td className="num pr-3 text-xs text-zinc-400">{m.trained_at}</td>
                    <td className="pr-3 text-zinc-300">{m.what}</td>
                    <td className="text-xs text-zinc-300">
                      {Object.entries(m.metrics).slice(0, 4).map(([k, v]) => <div key={k}><span className="text-zinc-500">{k}</span> {typeof v === "number" ? +v.toFixed(4) : String(v)}</div>)}
                    </td>
                  </tr>
                  {open === m.model_id && (
                    <tr><td colSpan={6} className="pb-3">
                      <div className="grid gap-3 rounded bg-zinc-950/60 p-3 text-xs md:grid-cols-3">
                        <div><div className="mb-1 text-zinc-500">all scores</div>{Object.entries(m.metrics).map(([k, v]) => <div key={k} className="flex justify-between gap-2"><span>{k}</span><span className="num">{typeof v === "number" ? +v.toFixed(4) : String(v)}</span></div>)}</div>
                        <div><div className="mb-1 text-zinc-500">trained on</div><pre className="whitespace-pre-wrap text-zinc-300">{JSON.stringify(m.data, null, 1)}</pre>
                          <div className="mt-2 mb-1 text-zinc-500">file</div><div className="font-mono break-all">{m.path}</div>
                          {m.notes && <><div className="mt-2 mb-1 text-zinc-500">notes</div><div className="text-zinc-300">{m.notes}</div></>}</div>
                        <div><div className="mb-1 text-zinc-500">features ({m.features.length})</div><div className="font-mono text-zinc-400">{m.features.join(", ") || "—"}</div>
                          <div className="mt-2 mb-1 text-zinc-500">params</div><pre className="whitespace-pre-wrap text-zinc-400">{JSON.stringify(m.params, null, 1)}</pre></div>
                      </div>
                    </td></tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </Card>
      ))}
    </div>
  );
}
