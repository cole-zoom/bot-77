"""Terminal output for the bot (2026-10-21, Cole: "make sure it outputs logs to the cli ... for visuals"): a banner,
one coloured line per play, a status line every few seconds, and a scoreboard per game. Plain text when stdout isn't a
terminal (log files)."""
from __future__ import annotations

import os
import sys

TTY = sys.stdout.isatty() or os.environ.get("BOT77_COLOR") == "1"
C = {"red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36", "grey": "90", "bold": "1"}


def c(text: str, *styles: str) -> str:
    if not TTY or not styles:
        return text
    return "".join(f"\033[{C[s]}m" for s in styles) + text + "\033[0m"


def banner(device: str, layout: str, games: int, hours: float) -> str:
    line = "─" * 60
    return "\n".join([c(line, "magenta"),
                      c("  bot-77 · B v0 — Clash Royale bot", "bold", "magenta"),
                      c(f"  device {device} · layout {layout} · up to {games} games / {hours:g} h", "grey"),
                      c("  playbook rules (from Cole's reviews) + move predictor (A v1) · stop: touch data/live/STOP", "grey"),
                      c(line, "magenta")])


def where(x: float, y: float) -> str:
    """A spot in words (arena tiles, our half y > 15)."""
    lane = "left" if x < 9 else "right"
    if y < 9:
        return f"{lane} enemy tower"
    if y < 14.9:
        return f"{lane} enemy side"
    if y < 17.5:
        return f"{lane} bridge"
    if 7 <= x <= 11 and y < 22:
        return "centre"
    if y < 21:
        return f"in front of {lane} tower"
    return f"behind {lane} tower"


def elixir_bar(e: float | None) -> str:
    if e is None:
        return ""
    n = max(0, min(10, int(e)))
    return c("▮" * n, "magenta") + c("▯" * (10 - n), "grey") + f" {e:4.1f}"


def clock(time_left_s: float | None) -> str:  # the fast reader gives a float
    return "--:--" if time_left_s is None else f"{int(time_left_s) // 60}:{int(time_left_s) % 60:02d}"


def play_line(st, act) -> str:
    reason = act.reason.split(" (")[0]
    how = act.reason.split("(")[-1].rstrip(")") if "(" in act.reason else ""
    col = {"defend": "red", "full elixir": "yellow", "idle": "grey", "support": "green"}.get(reason, "cyan")
    return (f"  {c(clock(st.time_left_s), 'grey')}  {c(f'{act.card:16s}', 'bold', 'cyan')} → {where(act.x, act.y):24s} "
            f"{c(f'{reason:11s}', col)} {c(how, 'grey'):9s} {elixir_bar(st.elixir)}")


def status_line(st, names: dict) -> str:
    hand = " ".join(names.get(h, "?")[:10] for h in st.hand)
    th = st.threats()
    t = c(f"{len(th)} threat{'s' if len(th) != 1 else ''}", "red") if th else c("no threats", "green")
    return c(f"  · {clock(st.time_left_s)}  hand: {hand}  ·  ", "grey") + t + "  " + elixir_bar(st.elixir)


def game_start(n: int, opp: str) -> str:
    return "\n" + c(f"▶ Game {n}", "bold", "green") + (c(f"  vs {opp}", "bold") if opp else "")


def game_end(n: int, result: str, plays: int, secs: float, wins: int, losses: int) -> str:
    col = {"win": "green", "loss": "red"}.get(result, "yellow")
    res = {"win": "WIN", "loss": "LOSS"}.get(result, "result ?")
    return (c(f"■ Game {n}: ", "bold") + c(res, "bold", col) + c(f"  · {plays} plays · {secs / 60:.1f} min", "grey")
            + c(f"   session: {wins}W {losses}L", "bold"))
