"""Оборонительное поведение: прессинг, цепочки, зоны отбора."""

from __future__ import annotations

import pandas as pd

from ..model import Capability as C
from ..model import MatchSet
from .common import (
    CHANNEL_ORDER,
    CHANNEL_RU,
    PRESS_RU,
    THIRD_ORDER,
    THIRD_RU,
    engagements,
    episodes,
    on_ball,
    opponent_on_ball,
    share,
    zone_grid,
)
from .registry import metric


@metric(
    "pressing_profile",
    "Профиль прессинга",
    "Оборона",
    requires=(C.PRESSING_CHAINS,),
    note="Оборонительные единоборства и давление на игрока с мячом, "
    "нормировано на 100 владений соперника.",
)
def pressing_profile(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    eng = engagements(ev, tid)
    opp = opponent_on_ball(ev, tid)
    if eng.empty or opp.empty:
        return {}

    denom = len(opp) / 100.0
    by_type = []
    for k, g in eng.groupby("subtype"):
        by_type.append(
            {
                "key": PRESS_RU.get(k, k),
                "n": int(len(g)),
                "per100": round(len(g) / denom, 1),
                "share": round(100 * len(g) / len(eng), 1),
            }
        )
    by_type.sort(key=lambda r: -r["n"])

    # Где вступают в отбор — в системе координат обороняющейся команды
    # x < 0 у нас = ближе к своим воротам (мы уже во фрейме subject).
    thirds = share(eng["third_start"].map(THIRD_RU), [THIRD_RU[t] for t in THIRD_ORDER])
    channels = share(eng["channel_start"].map(CHANNEL_RU), [CHANNEL_RU[c] for c in CHANNEL_ORDER])

    # Исход: сорвали угрозу / заставили назад / их обыграли.
    def rate(col: str) -> float | None:
        s = eng[col].dropna()
        return round(100 * float(s.mean()), 1) if len(s) else None

    return {
        "total": int(len(eng)),
        "per100": round(len(eng) / denom, 1),
        "by_type": by_type,
        "thirds": thirds,
        "channels": channels,
        "stop_danger": rate("stop_possession_danger"),
        "force_backward": rate("force_backward"),
        "beaten_possession": rate("beaten_by_possession"),
        "beaten_movement": rate("beaten_by_movement"),
        "resolved": int(eng["end_type"].notna().sum()),
        "mean_danger_faced": round(float(eng["danger_faced"].dropna().mean()), 3)
        if "danger_faced" in eng and eng["danger_faced"].notna().any()
        else None,
        "regains": int(eng["end_type"].isin(["direct_regain", "indirect_regain"]).sum()),
        "goal_side": round(100 * float(eng["goal_side_start"].dropna().mean()), 1)
        if "goal_side_start" in eng and eng["goal_side_start"].notna().any()
        else None,
        "clips": episodes(
            eng[eng["end_type"].notna()].sort_values("t"),
            lambda r: (
                f"{PRESS_RU.get(r['subtype'], r['subtype'])} — {r['player_name']}, "
                f"{THIRD_RU.get(r.get('third_start'), '')}"
            ),
            30,
        ),
    }


@metric(
    "pressing_chains",
    "Цепочки прессинга",
    "Оборона",
    requires=(C.PRESSING_CHAINS,),
    note="Цепочка — серия оборонительных действий подряд по одному владению "
    "соперника. Исход: regain (отобрали) или disruption (сбили атаку).",
)
def pressing_chains(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    eng = engagements(ev, tid)
    if eng.empty or "pressing_chain" not in eng.columns:
        return {}
    ch = eng[eng["pressing_chain"] == True]  # noqa: E712
    if ch.empty:
        return {}
    heads = ch[ch["index_in_pressing_chain"] == 1] if "index_in_pressing_chain" in ch else ch
    ends = share(ch["pressing_chain_end_type"].dropna())
    lens = ch["pressing_chain_length"].dropna()

    regain = ch[ch["pressing_chain_end_type"] == "regain"]
    clips = episodes(
        regain.sort_values("t"),
        lambda r: f"Цепочка прессинга, отбор — {r['player_name']}, {r.get('def_phase', '')}",
        limit=30,
    )
    return {
        "n_chains": int(heads["pressing_chain_index"].nunique())
        if "pressing_chain_index" in heads
        else int(len(heads)),
        "mean_length": round(float(lens.mean()), 1) if len(lens) else None,
        "max_length": int(lens.max()) if len(lens) else None,
        "ends": ends,
        "regain_share": round(100 * (ch["pressing_chain_end_type"] == "regain").mean(), 1),
        "clips": clips,
    }


@metric(
    "ppda",
    "PPDA",
    "Оборона",
    requires=(C.EVENTS,),
    note="Передачи соперника на одно оборонительное действие в его атакующих "
    "двух третях. Меньше — агрессивнее.",
)
def ppda(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    opp = opponent_on_ball(ev, tid)
    # Во фрейме разбираемой команды её ворота на x = -52. Зона расчёта PPDA —
    # всё, что выше её защитной трети, то есть x > -105/6.
    cut = -105 / 6
    opp_pass = opp[(opp["end_type"] == "pass") & (opp["x"] > cut)]
    eng = engagements(ev, tid)
    # Оборонительным действием считаем то, что имело исход: отбор, срыв, фол.
    # Простое «давление» без исхода в знаменатель PPDA не входит.
    eng = eng[eng["end_type"].notna()]
    eng_hi = eng[eng["x"] > cut]
    if len(eng_hi) == 0:
        return {}
    val = len(opp_pass) / len(eng_hi)
    per_match = []
    for mid, g in ev.groupby("match_id"):
        o = g[
            (g["type"] == "player_possession")
            & (g["possession_team_id"] != tid)
            & (g["end_type"] == "pass")
            & (g["x"] > cut)
        ]
        d = g[
            (g["type"] == "on_ball_engagement")
            & (g["team_id"] == tid)
            & (g["x"] > cut)
            & (g["end_type"].notna())
        ]
        if len(d):
            per_match.append({"match": g["match_label"].iloc[0], "ppda": round(len(o) / len(d), 2)})
    return {
        "value": round(val, 2),
        "per_match": per_match,
        "clips": episodes(
            eng_hi.sort_values("t"), lambda r: f"Отбор выше своей трети — {r['player_name']}", 30
        ),
    }


@metric(
    "regain_zones",
    "Зоны отбора",
    "Оборона",
    requires=(C.EVENTS,),
    note="Где команда возвращает мяч. Сетка 6×5, вид со стороны её атаки.",
)
def regain_zones(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    eng = engagements(ev, tid)
    regains = (
        eng[eng["end_type"].isin(["direct_regain", "indirect_regain"])]
        if "end_type" in eng
        else eng.iloc[0:0]
    )
    # Плюс перехваты и подборы, зафиксированные как начало владения.
    own = on_ball(ev, tid)
    starts = own[
        own["start_type"].isin(
            [
                "pass_interception",
                "recovery",
                "corner_interception",
                "throw_in_interception",
                "goal_kick_interception",
            ]
        )
    ]
    allr = pd.concat([regains, starts], ignore_index=True)
    if allr.empty:
        return {}
    grid = zone_grid(allr, length=105.0, width=68.0)
    thirds = share(allr["third_start"].map(THIRD_RU), [THIRD_RU[t] for t in THIRD_ORDER])
    high = allr[allr["x"] > 105 / 6]
    return {
        "grid": grid.tolist(),
        "total": int(len(allr)),
        "thirds": thirds,
        "high_regains": int(len(high)),
        "high_share": round(100 * len(high) / len(allr), 1),
        "clips": episodes(
            high.sort_values("t"), lambda r: f"Отбор на чужой половине — {r['player_name']}", 30
        ),
    }
