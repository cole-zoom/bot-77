// One match: the video (or single frames when the browser can't play / seek the file) with the detector's boxes drawn
// on top, both players' plays and the moments on a timeline, and the play list. Click a play to seek, a moment to
// inspect it.
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, get, type S } from "../api/client";
import { Card, Loading, Pill, SPLIT_COLOR, clock, fmt, useLoad } from "../ui";

const SIDE_COLOR: Record<string, string> = { bottom: "#a78bfa", top: "#fb923c" };
const SRC_LABEL: Record<string, string> = { hand: "hand", detected: "C", bot: "bot" };

function Overlay({ boxes, show }: { boxes: S["Box"][]; show: boolean }) {
  if (!show) return null;
  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none">
      {boxes.filter((b) => b.card).map((b, i) => (
        <g key={i}>
          <rect x={b.x0} y={b.y0} width={b.x1 - b.x0} height={b.y1 - b.y0} fill="none"
            stroke={b.team === "friendly" ? "#60a5fa" : "#f87171"} strokeWidth={0.003} vectorEffect="non-scaling-stroke" />
        </g>
      ))}
    </svg>
  );
}

function Labels({ boxes, show }: { boxes: S["Box"][]; show: boolean }) {
  if (!show) return null;
  return (
    <div className="pointer-events-none absolute inset-0">
      {boxes.filter((b) => b.card).map((b, i) => (
        <span key={i} className={`absolute -translate-y-full rounded px-1 text-[10px] leading-4 ${b.team === "friendly" ? "bg-blue-500/80" : "bg-red-500/80"}`}
          style={{ left: `${b.x0 * 100}%`, top: `${b.y0 * 100}%` }}>{b.card} {Math.round(b.conf * 100)}</span>
      ))}
    </div>
  );
}

export default function Match() {
  const { id = "" } = useParams();
  const { data, error } = useLoad(() => get(api.GET("/api/matches/{match_id}", { params: { path: { match_id: id } } })), [id]);
  const [t, setT] = useState(0);
  const [mode, setMode] = useState<"video" | "frame">("video");
  const [overlay, setOverlay] = useState(true);
  const [run, setRun] = useState("");
  const [sides, setSides] = useState({ bottom: true, top: true });
  const [sources, setSources] = useState<Record<string, boolean>>({ hand: true, detected: true, bot: true });
  const [boxes, setBoxes] = useState<S["Box"][]>([]);
  const vid = useRef<HTMLVideoElement>(null);

  const range = useMemo(() => {
    if (!data) return [0, 1];
    const ts = [...data.plays.map((p) => p.t), ...data.moments.map((m) => m.t)];
    return ts.length ? [Math.max(0, Math.min(...ts) - 5), Math.max(...ts) + 5] : [0, 200];
  }, [data]);
  useEffect(() => { if (data) { setT(range[0]); setRun(data.runs.includes("v5l_e15") ? "v5l_e15" : data.runs[0] ?? ""); } }, [data, range]);
  useEffect(() => {  // boxes for the current time (debounced)
    if (!run || !overlay) return;
    const h = setTimeout(() => {
      get(api.GET("/api/detections/{match_id}", { params: { path: { match_id: id }, query: { t, run_id: run } } }))
        .then((d) => setBoxes(d.boxes)).catch(() => setBoxes([]));
    }, 120);
    return () => clearTimeout(h);
  }, [t, run, overlay, id]);

  if (!data) return <Loading error={error} />;
  const m = data.match as Record<string, unknown>;
  const videoId = String(m.video_id ?? "");
  const available = Boolean(data.media && (data.media as Record<string, unknown>).available);
  const seek = (x: number) => { setT(x); if (mode === "video" && vid.current) vid.current.currentTime = x; };
  const plays = data.plays.filter((p) => sides[p.side as "bottom" | "top"] && sources[p.source]);
  const span = range[1] - range[0];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="font-mono text-lg">{id}</h1>
        <Pill text={String(m.source)} /><Pill text={String(m.split)} cls={SPLIT_COLOR[String(m.split)]} />
        {m.league != null && <Pill text={`league ${m.league}`} />}
        {m.result != null && <Pill text={String(m.result)} />}
        {data.pair && <Link className="text-sm text-violet-300 hover:underline" to={`/matches/${(data.pair as Record<string, string>).match_id}`}>other side →</Link>}
      </div>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,420px)_1fr]">
        <Card>
          {!available ? (
            <div className="text-sm text-amber-300">Video file not available{String((data.media as Record<string, unknown> | null)?.real_path ?? "").includes("LaCie") ? " — it's on the LaCie: plug it in." : "."}</div>
          ) : (
            <div className="relative overflow-hidden rounded-lg bg-black">
              {mode === "video" ? (
                <video ref={vid} src={`/api/video/${videoId}`} className="w-full" controls={false} muted
                  onTimeUpdate={(e) => setT(e.currentTarget.currentTime)} onError={() => setMode("frame")}
                  onLoadedMetadata={(e) => { if (!isFinite(e.currentTarget.duration)) setMode("frame"); else e.currentTarget.currentTime = t; }} />
              ) : (
                <img src={`/api/frame/${videoId}?t=${t.toFixed(2)}&width=540`} className="w-full" alt="frame" />
              )}
              <Overlay boxes={boxes} show={overlay} />
              <Labels boxes={boxes} show={overlay} />
            </div>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
            {mode === "video" && <button className="rounded bg-zinc-800 px-2 py-1" onClick={() => vid.current && (vid.current.paused ? vid.current.play() : vid.current.pause())}>play / pause</button>}
            <button className="rounded bg-zinc-800 px-2 py-1" onClick={() => seek(Math.max(range[0], t - 1))}>−1 s</button>
            <button className="rounded bg-zinc-800 px-2 py-1" onClick={() => seek(t + 1)}>+1 s</button>
            <span className="num text-zinc-400">t {fmt(t, 2)} s</span>
            <label className="flex items-center gap-1"><input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} /> boxes</label>
            <select className="rounded bg-zinc-800 px-1 py-0.5 text-xs" value={run} onChange={(e) => setRun(e.target.value)}>
              {data.runs.map((r) => <option key={r}>{r}</option>)}
            </select>
            <span className="text-xs text-zinc-500">{mode === "frame" ? "single frames (the browser can't seek this file)" : ""}</span>
          </div>
        </Card>
        <div className="space-y-4">
          <Card title="Timeline">
            <div className="mb-2 flex flex-wrap gap-3 text-xs">
              {(["bottom", "top"] as const).map((s) => (
                <label key={s} className="flex items-center gap-1"><input type="checkbox" checked={sides[s]} onChange={(e) => setSides({ ...sides, [s]: e.target.checked })} />
                  <span style={{ color: SIDE_COLOR[s] }}>{s} player</span></label>
              ))}
              {Object.keys(sources).map((s) => (
                <label key={s} className="flex items-center gap-1"><input type="checkbox" checked={sources[s]} onChange={(e) => setSources({ ...sources, [s]: e.target.checked })} />{SRC_LABEL[s]} plays</label>
              ))}
            </div>
            <div className="relative h-20 cursor-pointer rounded bg-zinc-800/60"
              onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); seek(range[0] + ((e.clientX - r.left) / r.width) * span); }}>
              {data.moments.filter((x) => x.kind === "idle" && sides[x.side as "bottom" | "top"]).map((x) => (
                <div key={x.moment_id} className="absolute h-1 w-px bg-zinc-600" style={{ left: `${((x.t - range[0]) / span) * 100}%`, top: x.side === "bottom" ? 66 : 8 }} />
              ))}
              {plays.map((p, i) => (
                <div key={i} title={`${clock(p.t_match)} ${p.card} (${SRC_LABEL[p.source]})`} className="absolute w-1 rounded"
                  style={{ left: `${((p.t - range[0]) / span) * 100}%`, top: p.side === "bottom" ? 44 : 16, height: 18,
                    background: SIDE_COLOR[p.side], opacity: p.source === "hand" ? 1 : 0.5 }} />
              ))}
              <div className="absolute top-0 h-full w-0.5 bg-white" style={{ left: `${((t - range[0]) / span) * 100}%` }} />
            </div>
            <div className="mt-1 flex justify-between text-xs text-zinc-500"><span>top player ↑ · bottom player ↓ · grey ticks: could-have-played waits</span><span>click to seek</span></div>
          </Card>
          <Card title={`Plays (${plays.length})`}>
            <div className="max-h-[52vh] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-zinc-900 text-left text-zinc-500"><tr><th className="py-1">clock</th><th>player</th><th>card</th><th>from</th><th>tile</th><th>moment</th></tr></thead>
                <tbody>
                  {plays.map((p, i) => {
                    const mom = data.moments.find((x) => x.kind === "play" && x.side === p.side && Math.abs(x.t - p.t) < 0.01);
                    return (
                      <tr key={i} className={`cursor-pointer border-t border-zinc-800 hover:bg-zinc-800/40 ${Math.abs(p.t - t) < 0.6 ? "bg-violet-950/40" : ""}`} onClick={() => seek(p.t)}>
                        <td className="num py-1 pr-2 text-zinc-400">{clock(p.t_match)}</td>
                        <td className="pr-2" style={{ color: SIDE_COLOR[p.side] }}>{p.side}</td>
                        <td className="pr-2">{p.card}{p.kind === "ability" ? " (ability)" : ""}</td>
                        <td className="pr-2 text-xs text-zinc-500">{SRC_LABEL[p.source]}{p.reason ? ` · ${p.reason}` : ""}</td>
                        <td className="num pr-2 text-xs text-zinc-400">{p.x == null ? "" : `${fmt(p.x)}, ${fmt(p.y)}`}</td>
                        <td>{mom && <Link onClick={(e) => e.stopPropagation()} className="text-xs text-violet-300 hover:underline" to={`/moments/${encodeURIComponent(mom.moment_id)}`}>inspect</Link>}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
