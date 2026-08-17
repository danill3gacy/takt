"""Структура команды: блок, линии, расстановка. Считается из трекинга."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..model import Capability as C
from ..model import MatchSet
from .common import episodes, phase_episodes, share
from .registry import metric

DEF_PHASE_ORDER = ["высокий блок", "средний блок", "низкий блок",
                   "против длинных", "против стандарта", "переход в оборону", "хаос"]


@metric("block_shape", "Форма оборонительного блока", "Оборона",
        requires=(C.TEAM_SHAPE, C.PHASES),
        note="Ширина и длина блока — расстояние между крайними игроками "
             "поперёк и вдоль поля в момент завершения фазы.")
def block_shape(ms: MatchSet) -> dict:
    ph = ms.phases()
    tid = ms.subject_team.id
    if ph.empty:
        return {}

    # Фазы, где разбираемая команда БЕЗ мяча.
    out = ph[ph["team_id"] != tid].copy()
    out["area"] = out["width_out"] * out["length_out"]

    rows = []
    for name, g in out.groupby("def_phase"):
        if len(g) < 5:
            continue
        rows.append({
            "phase": name,
            "n": int(len(g)),
            "share": round(100 * len(g) / len(out), 1),
            "width": round(float(g["width_out"].mean()), 1),
            "length": round(float(g["length_out"].mean()), 1),
            "area": round(float(g["area"].mean()), 0),
        })
    rows.sort(key=lambda r: DEF_PHASE_ORDER.index(r["phase"])
              if r["phase"] in DEF_PHASE_ORDER else 99)

    # То же в атаке — для сравнения «как растягиваются».
    inp = ph[ph["team_id"] == tid]
    return {
        "rows": rows,
        "mean_width": round(float(out["width_out"].mean()), 1),
        "mean_length": round(float(out["length_out"].mean()), 1),
        "mean_area": round(float(out["area"].mean()), 0),
        "attack_width": round(float(inp["width_in"].mean()), 1) if len(inp) else None,
        "attack_length": round(float(inp["length_in"].mean()), 1) if len(inp) else None,
        "n_phases_out": int(len(out)),
        "clips_low": phase_episodes(
            out[out["def_phase"] == "низкий блок"].nsmallest(30, "width_out"),
            lambda r: f"Низкий блок, ширина {r['width_out']:.0f} м", 30),
        "clips": phase_episodes(
            out.nlargest(30, "length_out"),
            lambda r: f"{r['def_phase']}, блок {r['width_out']:.0f}×{r['length_out']:.0f} м", 30),
    }


@metric("defensive_line", "Высота последней линии обороны", "Оборона",
        requires=(C.DEFENSIVE_LINE,),
        note="Расстояние от своих ворот до последнего полевого игрока, метры.")
def defensive_line(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    # Когда соперник владеет мячом, last_defensive_line_* описывает линию
    # разбираемой команды.
    d = ev[(ev["type"] == "player_possession") & (ev["possession_team_id"] != tid)]
    d = d.dropna(subset=["last_defensive_line_height_start"])
    d = d[~d["def_phase"].isin(["срыв атаки соперника", "хаос"])]
    if d.empty:
        return {}
    h = d["last_defensive_line_height_start"]
    by_phase = []
    for name, g in d.groupby("def_phase"):
        if len(g) < 10:
            continue
        by_phase.append({
            "phase": name,
            "n": int(len(g)),
            "height": round(float(g["last_defensive_line_height_start"].mean()), 1),
        })
    by_phase.sort(key=lambda r: -r["height"])

    # Динамика по 15-минуткам — падает ли линия к концу матча.
    d = d.assign(bucket=np.minimum((d["minute"] // 15).astype(int), 5))
    by_time = [{"bucket": f"{int(b)*15}–{int(b)*15+15}",
                "height": round(float(g["last_defensive_line_height_start"].mean()), 1),
                "n": int(len(g))}
               for b, g in d.groupby("bucket")]
    late = d[d["minute"] >= 75]
    return {
        "mean": round(float(h.mean()), 1),
        "p25": round(float(h.quantile(0.25)), 1),
        "p75": round(float(h.quantile(0.75)), 1),
        "by_phase": by_phase,
        "by_time": by_time,
        "clips": episodes(
            d.nlargest(30, "last_defensive_line_height_start"),
            lambda r: f"Линия на {r['last_defensive_line_height_start']:.0f} м — {r.get('def_phase','')}", 30),
        "clips_late": episodes(
            late.nsmallest(30, "last_defensive_line_height_start"),
            lambda r: f"{int(r['minute'])+1}′, линия {r['last_defensive_line_height_start']:.0f} м", 30),
    }


@metric("formations", "Расстановка в обороне", "Оборона",
        requires=(C.LINE_BREAKS,),
        note="Схема распознана по расположению игроков на поле, а не взята из протокола.")
def formations(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    base = ev[(ev["type"] == "player_possession") & (ev["possession_team_id"] != tid)]
    d = base.dropna(subset=["defensive_structure"])
    if d.empty:
        return {}
    s = d["defensive_structure"].astype(int).astype(str).map(
        lambda v: "-".join(list(v)))
    rows = share(s)[:6]
    lines = d["n_defensive_lines"].dropna()
    org = base["organised_defense"].dropna()
    return {
        "rows": rows,
        "n_lines_mean": round(float(lines.mean()), 2) if len(lines) else None,
        "organised_share": round(100 * float(org.mean()), 1) if len(org) else None,
        "n": int(len(d)),
        "n_observed": int(len(base)),
        "clips": episodes(
            d[s == rows[0]["key"]].sort_values("t") if rows else d.iloc[0:0],
            lambda r: f"Оборона {rows[0]['key']} — {r.get('def_phase','')}", 30),
    }


@metric("avg_positions", "Средние позиции и связи", "Атака",
        requires=(C.EVENTS,),
        note="Позиция игрока усреднена по всем его действиям с мячом и открываниям; "
             "связи — по фактическим передачам между игроками.")
def avg_positions(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id

    own = ev[(ev["possession_team_id"] == tid) &
             (ev["type"].isin(["player_possession", "passing_option"]))]
    own = own.dropna(subset=["player_id", "x", "y"])
    if own.empty:
        return {}

    names = {p.id: p.name for m in ms.matches for p in m.players.values()}
    positions = {p.id: p.position for m in ms.matches for p in m.players.values()}

    nodes: list[dict[str, Any]] = []
    for pid, g in own.groupby("player_id"):
        if len(g) < 25:
            continue
        nodes.append({
            "id": int(pid),
            "name": names.get(int(pid), str(pid)),
            "pos": positions.get(int(pid)) or "",
            "x": round(float(g["x"].mean()), 1),
            "y": round(float(g["y"].mean()), 1),
            "n": int(len(g)),
            "touches": int((g["type"] == "player_possession").sum()),
        })
    nodes.sort(key=lambda r: -r["touches"])
    nodes = nodes[:14]
    keep = {n["id"] for n in nodes}

    # Связи: успешные передачи адресату.
    pas = ev[(ev["possession_team_id"] == tid) &
             (ev["type"] == "player_possession") &
             (ev["end_type"] == "pass")]
    links: dict[tuple[int, int], int] = {}
    if "player_targeted_id" in pas.columns:
        p = pas.dropna(subset=["player_id", "player_targeted_id"])
        for a, b in zip(p["player_id"].astype(int), p["player_targeted_id"].astype(int)):
            if a in keep and b in keep and a != b:
                k = (min(a, b), max(a, b))
                links[k] = links.get(k, 0) + 1
    edges = [{"a": a, "b": b, "n": n} for (a, b), n in links.items() if n >= 4]
    edges.sort(key=lambda r: -r["n"])
    return {"nodes": nodes, "edges": edges[:28]}
