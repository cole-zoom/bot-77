// What's in the data: tables, matches by source / split, moments, labels, detector runs, models.
import { api, get } from "../api/client";
import { Bar, Card, Loading, useLoad } from "../ui";

function Counts({ title, rows }: { title: string; rows: { key: string; n: number }[] }) {
  const max = Math.max(...rows.map((r) => r.n), 1);
  return (
    <Card title={title}>
      <div className="space-y-2">
        {rows.map((r) => (
          <div key={r.key}>
            <div className="flex justify-between text-sm"><span>{r.key}</span><span className="num text-zinc-400">{r.n.toLocaleString()}</span></div>
            <Bar value={r.n} max={max} />
          </div>
        ))}
      </div>
    </Card>
  );
}

export default function Overview() {
  const { data, error } = useLoad(() => get(api.GET("/api/overview")), []);
  if (!data) return <Loading error={error} />;
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Overview</h1>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Counts title="Matches by source" rows={data.matches_by_source} />
        <Counts title="Matches by split" rows={data.matches_by_split} />
        <Counts title="Moments by kind" rows={data.moments_by_kind} />
        <Counts title="Labels by source" rows={data.labels_by_source} />
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Detector runs" className="lg:col-span-2">
          <table className="w-full text-sm">
            <thead className="text-left text-zinc-500"><tr><th className="py-1">run</th><th>what</th><th>role</th><th className="text-right">matches</th></tr></thead>
            <tbody>
              {data.runs.map((r: Record<string, unknown>) => (
                <tr key={String(r.run_id)} className="border-t border-zinc-800 align-top">
                  <td className="py-1.5 pr-3 font-mono text-xs">{String(r.run_id)}</td>
                  <td className="pr-3 text-zinc-300">{String(r.what)}</td>
                  <td className="pr-3 text-zinc-400">{String(r.role)}</td>
                  <td className="num text-right">{Number(r.matches).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        <Card title="Models on disk">
          <div className="flex flex-wrap gap-1.5">{data.models.map((m) => <span key={m} className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-xs">{m}</span>)}</div>
        </Card>
      </div>
      <Card title="LanceDB tables">
        <div className="grid gap-x-8 gap-y-1 text-sm sm:grid-cols-2 lg:grid-cols-3">
          {data.tables.map((t) => (
            <div key={t.name} className="flex justify-between border-b border-zinc-800/60 py-1">
              <span className="font-mono text-xs">{t.name}</span><span className="num text-zinc-400">{t.rows.toLocaleString()}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
