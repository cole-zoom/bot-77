// The feature registry (bot77.features.registry): every feature the models see, what it means, since when.
import { useState } from "react";
import { api, get } from "../api/client";
import { Card, Loading, Pill, useLoad } from "../ui";

export default function Features() {
  const { data, error } = useLoad(() => get(api.GET("/api/features")), []);
  const [f, setF] = useState("");
  if (!data) return <Loading error={error} />;
  const groups = [...new Set(data.map((x) => x.group))];
  const shown = data.filter((x) => !f || `${x.name} ${x.doc} ${x.group}`.toLowerCase().includes(f.toLowerCase()));
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <h1 className="text-xl font-semibold">Features</h1>
        <span className="text-sm text-zinc-500">{data.length} registered</span>
        <input className="ml-auto w-64 rounded bg-zinc-800 px-2 py-1 text-sm" placeholder="filter…" value={f} onChange={(e) => setF(e.target.value)} />
      </div>
      {groups.map((g) => {
        const rows = shown.filter((x) => x.group === g);
        return rows.length === 0 ? null : (
          <Card key={g} title={g}>
            <table className="w-full text-sm">
              <tbody>
                {rows.map((x) => (
                  <tr key={x.name} className="border-t border-zinc-800 first:border-0">
                    <td className="w-56 py-1 pr-3 font-mono text-xs text-violet-200">{x.name}</td>
                    <td className="w-20 pr-3"><Pill text={x.level} /></td>
                    <td className="pr-3 text-zinc-300">{x.doc}</td>
                    <td className="w-20 text-right text-xs text-zinc-500">{x.since}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        );
      })}
    </div>
  );
}
