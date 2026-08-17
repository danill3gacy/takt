"""Профили игроков: что каждый реально делает на поле."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..model import Capability as C
from ..model import MatchSet
from .common import RUN_RU, engagements, episodes, off_ball_runs, on_ball
from .registry import metric


@metric("player_table", "Игроки", "Игроки",
        requires=(C.EVENTS,),
        note="Всё на 90 минут. Взломы — передачи и ведения за линию обороны, "
             "забегания — открывания под передачу, отборы — оборонительные единоборства.")
def player_table(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id

    minutes: dict[int, float] = {}
    names: dict[int, str] = {}
    positions: dict[int, str] = {}
    for m in ms.matches:
        for p in m.players.values():
            if p.team_id != tid:
                continue
            minutes[p.id] = minutes.get(p.id, 0.0) + p.minutes
            names[p.id] = p.name
            positions[p.id] = p.position or ""

    own = on_ball(ev, tid)
    runs = off_ball_runs(ev, tid)
    eng = engagements(ev, tid)

    rows = []
    for pid, mins in minutes.items():
        if mins < 60:  # меньше 60 минут суммарно — статистика не читается
            continue
        o = own[own["player_id"] == pid]
        r = runs[runs["player_id"] == pid]
        e = eng[eng["player_id"] == pid]
        k = 90.0 / mins

        breaks = o["furthest_line_break"].notna().sum() if "furthest_line_break" in o else 0
        xt = float(o["xt"].fillna(0).sum()) if "xt" in o else 0.0
        passes = int((o["end_type"] == "pass").sum())
        lost = int((o["end_type"] == "possession_loss").sum())
        shots = int((o["end_type"] == "shot").sum())
        behind = int((r["subtype"] == "behind").sum())
        regains = int(e["end_type"].isin(["direct_regain", "indirect_regain"]).sum()) \
            if "end_type" in e else 0
        speed = float(r["speed_avg"].dropna().max()) if "speed_avg" in r and len(r) else None

        rows.append({
            "id": int(pid),
            "name": names.get(pid, str(pid)),
            "pos": positions.get(pid, ""),
            "minutes": round(mins, 0),
            "touches90": round(len(o) * k, 1),
            "passes90": round(passes * k, 1),
            "xthreat90": round(xt * k, 2),
            "breaks90": round(breaks * k, 2),
            "shots90": round(shots * k, 2),
            "losses90": round(lost * k, 2),
            "runs90": round(len(r) * k, 1),
            "behind90": round(behind * k, 2),
            "press90": round(len(e) * k, 1),
            "regains90": round(regains * k, 2),
            "top_speed": round(speed, 1) if speed else None,
        })
    rows.sort(key=lambda x: -x["minutes"])
    return {"rows": rows}


@metric("key_players", "Ключевые игроки", "Игроки",
        requires=(C.XTHREAT, C.OFF_BALL_RUNS),
        note="Автоматический разбор трёх профилей: кто создаёт угрозу, "
             "кто продвигает мяч и кто больше всех бежит за спину.")
def key_players(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    own = on_ball(ev, tid)
    runs = off_ball_runs(ev, tid)
    if own.empty:
        return {}

    minutes: dict[int, float] = {}
    for m in ms.matches:
        for p in m.players.values():
            if p.team_id == tid:
                minutes[p.id] = minutes.get(p.id, 0.0) + p.minutes
    ok = {pid for pid, mn in minutes.items() if mn >= 120}

    def top(df, col, agg="sum", n=3):
        d = df[df["player_id"].isin(ok)]
        if d.empty:
            return []
        s = d.groupby(["player_id", "player_name"])[col]
        s = s.sum() if agg == "sum" else s.size()
        s = s.sort_values(ascending=False).head(n)
        out = []
        for (pid, nm), v in s.items():
            mn = minutes.get(int(pid), 90) or 90
            out.append({"id": int(pid), "name": nm,
                        "value": round(float(v), 2),
                        "per90": round(float(v) * 90 / mn, 2),
                        "minutes": round(mn)})
        return out

    threat = top(own, "xt")
    breakers = own.dropna(subset=["furthest_line_break"]) if "furthest_line_break" in own else own.iloc[0:0]
    behind = runs[runs["subtype"] == "behind"]

    def counts(df, n=3):
        d = df[df["player_id"].isin(ok)]
        if d.empty:
            return []
        s = d.groupby(["player_id", "player_name"]).size().sort_values(ascending=False).head(n)
        return [{"id": int(pid), "name": nm, "value": int(v),
                 "per90": round(v * 90 / (minutes.get(int(pid), 90) or 90), 2),
                 "minutes": round(minutes.get(int(pid), 0))}
                for (pid, nm), v in s.items()]

    profiles = []
    if threat:
        p = threat[0]
        clips = episodes(own[own["player_id"] == p["id"]].sort_values("xt", ascending=False),
                         lambda r: f"{r['player_name']} — {r.get('phase','')}, xT {float(r.get('xt') or 0):.3f}", 20)
        profiles.append({"role": "Главный источник угрозы", **p, "clips": clips})
    b = counts(breakers)
    if b:
        p = b[0]
        clips = episodes(breakers[breakers["player_id"] == p["id"]].sort_values("t"),
                         lambda r: f"Взлом линии — {r['player_name']}, {r.get('phase','')}", 20)
        profiles.append({"role": "Продвигает мяч через линии", **p, "clips": clips})
    r3 = counts(behind)
    if r3:
        p = r3[0]
        clips = episodes(behind[behind["player_id"] == p["id"]].sort_values("t"),
                         lambda r: f"Забегание за спину — {r['player_name']}", 20)
        profiles.append({"role": "Бежит за спину чаще всех", **p, "clips": clips})

    return {"profiles": profiles, "threat": threat, "breakers": b, "behind": r3}
