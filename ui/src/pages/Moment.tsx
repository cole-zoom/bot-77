// One moment as the models see it: the frame, the board in the deciding player's frame (with the placement model's
// candidate spots and the real spot), A v1's score for each card in hand with every feature, and the labels.
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, get, type S } from "../api/client";
import { Card, Loading, Pill, SPLIT_COLOR, clock, fmt, useLoad } from "../ui";

const W = 18, H = 27;  // arena tiles shown (player's frame: their towers at the bottom)

function Arena({ units, spots, truth, towers }: {
  units: Record<string, unknown>[]; spots: S["SpotScore"][]; truth?: { x: number; y: number; kind: string | null };
  towers: Record<string, boolean>;
}) {
  const top = [...spots].sort((a, b) => b.score - a.score);
  const best = top[0]?.score ?? 1;
  const tower = (x: number, y: number, up: boolean | undefined, own: boolean) => (
    <rect x={x - 1.5} y={y - 1.5} width={3} height={3} rx={0.4} fill={up ? (own ? "#1d4ed8" : "#b91c1c") : "none"}
      stroke={own ? "#60a5fa" : "#f87171"} strokeWidth={0.12} opacity={up ? 0.55 : 0.35} />
  );
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg bg-[#1f2a1c]">
      <rect x={0} y={14.3} width={W} height={1.2} fill="#1e3a5f" />
      <line x1={0} x2={W} y1={15.2} y2={15.2} stroke="#fbbf24" strokeWidth={0.05} strokeDasharray="0.3 0.3" />
      {tower(3.55, 5.68, towers.opp_l, false)}{tower(14.5, 5.68, towers.opp_r, false)}
      {tower(3.55, 20.78, towers.own_l, true)}{tower(14.5, 20.78, towers.own_r, true)}
      {top.map((s, i) => (
        <circle key={i} cx={s.x} cy={s.y} r={0.25 + 0.7 * (s.score / best)} fill="#a78bfa" opacity={i < 3 ? 0.85 : 0.25} />
      ))}
      {top[0] && <text x={top[0].x + 0.9} y={top[0].y + 0.3} fontSize={0.7} fill="#ddd6fe">model</text>}
      {units.map((u, i) => (
        <g key={i}>
          <circle cx={Number(u.x)} cy={Number(u.y)} r={0.45} fill={u.team === "own" ? "#3b82f6" : "#ef4444"} />
          <text x={Number(u.x) + 0.55} y={Number(u.y) + 0.25} fontSize={0.6} fill="#e4e4e7">{String(u.card)}</text>
        </g>
      ))}
      {truth && (
        <g>
          <path d={`M${truth.x - 0.6} ${truth.y - 0.6} L${truth.x + 0.6} ${truth.y + 0.6} M${truth.x + 0.6} ${truth.y - 0.6} L${truth.x - 0.6} ${truth.y + 0.6}`}
            stroke="#34d399" strokeWidth={0.18} />
          <text x={truth.x + 0.8} y={truth.y - 0.5} fontSize={0.7} fill="#6ee7b7">real ({truth.kind})</text>
        </g>
      )}
    </svg>
  );
}

export default function Moment() {
  const { id = "" } = useParams();
  const { data, error } = useLoad(() => get(api.GET("/api/moments/{moment_id}", { params: { path: { moment_id: id } } })), [id]);
  const [open, setOpen] = useState<number | null>(null);
  const [boxes, setBoxes] = useState<S["Box"][] | null>(null);
  if (!data) return <Loading error={error} />;
  const m = data.moment as Record<string, any>;
  const replayLabel = data.labels.find((l: any) => l.source === "replay" || l.source === "bot") as any;
  const truth = replayLabel?.x != null ? { x: replayLabel.x, y: replayLabel.y, kind: replayLabel.xy_kind } : undefined;
  const maxScore = Math.max(...data.cards.map((c) => c.score), 1e-9);
  const loadBoxes = () => get(api.GET("/api/detections/{match_id}", { params: { path: { match_id: m.game_id }, query: { t: m.t } } }))
    .then((d) => setBoxes(d.boxes)).catch(() => setBoxes([]));

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="font-mono text-base">{id}</h1>
        <Pill text={m.kind} cls={m.kind === "play" ? "bg-violet-900/60 text-violet-200" : "bg-zinc-800 text-zinc-300"} />
        <Pill text={`${m.side} player`} /><Pill text={m.source} /><Pill text={m.split ?? "?"} cls={SPLIT_COLOR[m.split]} />
        <span className="num text-sm text-zinc-400">clock {clock(m.t_match)} · elixir {fmt(m.elixir, 1)} · x{m.multiplier ?? 1}</span>
        <Link className="text-sm text-violet-300 hover:underline" to={`/matches/${m.match_id}`}>← match</Link>
      </div>
      <div className="grid gap-4 lg:grid-cols-[300px_340px_1fr]">
        <Card title="Frame">
          <div className="relative overflow-hidden rounded bg-black">
            <img src={m.source === "bot" ? `/api/botframe/${m.game_id}/${m.decision}?which=at` : `/api/frame/${m.video_id}?t=${Number(m.t).toFixed(2)}&width=540`}
              className="w-full" alt="frame"
              onError={(e) => { (e.currentTarget as HTMLImageElement).alt = "video not available (LaCie?)"; }} />
            {boxes && (
              <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none">
                {boxes.filter((b) => b.card).map((b, i) => <rect key={i} x={b.x0} y={b.y0} width={b.x1 - b.x0} height={b.y1 - b.y0} fill="none"
                  stroke={b.team === "friendly" ? "#60a5fa" : "#f87171"} strokeWidth={0.004} vectorEffect="non-scaling-stroke" />)}
              </svg>
            )}
          </div>
          <div className="mt-2 flex items-center gap-2 text-xs text-zinc-500">
            {m.source === "bot" ? <span>the bot's own screenshot at this play</span> :
              <button className="rounded bg-zinc-800 px-2 py-0.5 text-zinc-300" onClick={loadBoxes}>show boxes</button>}
            {m.side === "top" && <span>the video is the bottom player's screen</span>}
          </div>
        </Card>
        <Card title="Board as the model sees it">
          <Arena units={m.units} spots={data.spots} truth={truth} towers={m.towers ?? {}} />
          <div className="mt-2 text-xs text-zinc-500">
            This player's towers at the bottom. Purple: placement candidates for <b className="text-zinc-300">{data.spot_card ?? "—"}</b> (size = model score, top 3 bright).
            Green ×: where it really went. Yellow dashes: "crossed" line (y 15.2).
          </div>
        </Card>
        <div className="space-y-4">
          <Card title="Labels">
            {data.labels.length === 0 ? <div className="text-sm text-zinc-500">none</div> : (
              <div className="space-y-1.5 text-sm">
                {data.labels.map((l: any) => (
                  <div key={l.label_id} className="flex flex-wrap items-baseline gap-2">
                    <Pill text={l.source} cls={l.source.startsWith("review") ? "bg-emerald-900/60 text-emerald-200" : undefined} />
                    {l.verdict && <Pill text={l.verdict} cls={l.verdict === "bad" ? "bg-red-900/60 text-red-200" : "bg-emerald-900/60 text-emerald-200"} />}
                    <span className="font-medium">{l.action ?? "—"}</span>
                    {l.x != null && <span className="num text-xs text-zinc-500">at {fmt(l.x)}, {fmt(l.y)} ({l.xy_kind})</span>}
                    {l.note && <span className="text-zinc-400">— {l.note}</span>}
                  </div>
                ))}
              </div>
            )}
          </Card>
          <Card title="Card model (A v1): score per card in hand">
            {data.cards.length === 0 ? <div className="text-sm text-zinc-500">no clean hand read at this moment</div> : (
              <div className="space-y-1">
                {data.cards.map((c) => (
                  <div key={c.slot}>
                    <button className="flex w-full items-center gap-3 text-left text-sm" onClick={() => setOpen(open === c.slot ? null : c.slot)}>
                      <span className="w-36 truncate">{c.card}{replayLabel?.action === c.card ? " ✓" : ""}</span>
                      <div className="h-2 flex-1 rounded bg-zinc-800"><div className="h-2 rounded bg-violet-500" style={{ width: `${(100 * c.score) / maxScore}%` }} /></div>
                      <span className="num w-12 text-right text-zinc-400">{c.score.toFixed(3)}</span>
                    </button>
                    {open === c.slot && (
                      <div className="mt-1 mb-2 grid grid-cols-2 gap-x-6 rounded bg-zinc-950/60 p-2 font-mono text-[11px] text-zinc-400 md:grid-cols-3">
                        {Object.entries(c.features).map(([k, v]) => <div key={k} className="flex justify-between gap-2"><span>{k}</span><span className="text-zinc-200">{typeof v === "number" ? +v.toFixed(3) : String(v)}</span></div>)}
                      </div>
                    )}
                  </div>
                ))}
                <div className="pt-1 text-xs text-zinc-500">✓ = the card actually played. Click a card for its features.</div>
              </div>
            )}
          </Card>
          <Card title="Board features">
            <div className="grid grid-cols-2 gap-x-6 font-mono text-[11px] text-zinc-400 md:grid-cols-3">
              {Object.entries(data.board).map(([k, v]) => <div key={k} className="flex justify-between gap-2"><span>{k}</span><span className="text-zinc-200">{typeof v === "number" ? +v.toFixed(2) : String(v)}</span></div>)}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
