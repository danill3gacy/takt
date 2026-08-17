"""Игра с мячом: фазы, продвижение, взломы линий, угроза, потери, стандарты."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..model import Capability as C
from ..model import MatchSet
from .common import (
    CHANNEL_ORDER,
    CHANNEL_RU,
    SET_PIECE_RU,
    THIRD_ORDER,
    THIRD_RU,
    episodes,
    on_ball,
    opponent_on_ball,
    phase_episodes,
    share,
    zone_grid,
)
from .registry import metric


@metric(
    "possession_profile",
    "Владение и фазы атаки",
    "Атака",
    requires=(C.PHASES,),
    note="Фаза — отрезок владения с однородным намерением: начало атаки, "
    "созидание, завершение, длинная передача, стандарт.",
)
def possession_profile(ms: MatchSet) -> dict:
    ph = ms.phases()
    ev = ms.events()
    tid = ms.subject_team.id
    if ph.empty:
        return {}
    own = ph[ph["team_id"] == tid]
    if own.empty:
        return {}

    rows = []
    for name, g in own.groupby("phase"):
        rows.append(
            {
                "phase": name,
                "n": int(len(g)),
                "share": round(100 * len(g) / len(own), 1),
                "duration": round(float(g["duration"].mean()), 1),
                "possessions": round(float(g["n_possessions"].mean()), 1),
                "shot_rate": round(100 * float(g["lead_to_shot"].mean()), 1),
            }
        )
    rows.sort(key=lambda r: -r["n"])

    own_p = on_ball(ev, tid)
    opp_p = opponent_on_ball(ev, tid)
    tot = len(own_p) + len(opp_p)
    return {
        "rows": rows,
        "possession_share": round(100 * len(own_p) / tot, 1) if tot else None,
        "own_touches": int(len(own_p)),
        "mean_phase_duration": round(float(own["duration"].mean()), 1),
        "mean_possessions_per_phase": round(float(own["n_possessions"].mean()), 1),
        "n_phases": int(len(own)),
        "shot_rate": round(100 * float(own["lead_to_shot"].mean()), 1),
        "clips": phase_episodes(
            own[own["lead_to_shot"] == True].nlargest(30, "duration"),  # noqa: E712
            lambda r: (
                f"{r['phase']}, {r['duration']:.0f} с, {int(r['n_possessions'])} владений → удар"
            ),
            30,
        ),
    }


@metric(
    "progression",
    "Продвижение и взлом линий",
    "Атака",
    requires=(C.LINE_BREAKS,),
    note="Взлом линии — передача или ведение, после которого мяч оказывается "
    "за линией обороны соперника. «Через» — сквозь линию, «вокруг» — в обход.",
)
def progression(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    own = on_ball(ev, tid)
    if own.empty or "furthest_line_break" not in own.columns:
        return {}
    br = own.dropna(subset=["furthest_line_break"])
    per_match = len(br) / max(len(ms), 1)

    kinds = {"first": "первая линия", "second_last": "предпоследняя", "last": "последняя"}
    rows = []
    for k, ru in kinds.items():
        g = br[br["furthest_line_break"] == k]
        if not len(g):
            continue
        types = g["furthest_line_break_type"].value_counts()
        rows.append(
            {
                "line": ru,
                "n": int(len(g)),
                "through": int(types.get("through", 0)),
                "around": int(types.get("around", 0)),
                "share": round(100 * len(g) / len(br), 1),
            }
        )

    last = br[br["furthest_line_break"] == "last"]
    channels = share(last["channel_start"].map(CHANNEL_RU), [CHANNEL_RU[c] for c in CHANNEL_ORDER])
    by_player = br.groupby("player_name").size().sort_values(ascending=False).head(8)
    bypass = own["n_opponents_bypassed"].dropna()

    clips = episodes(
        last.sort_values("xt", ascending=False) if "xt" in last else last,
        lambda r: f"Взлом последней линии — {r['player_name']}, {r.get('phase', '')}",
        30,
    )
    return {
        "rows": rows,
        "total": int(len(br)),
        "per_match": round(per_match, 1),
        "channels": channels,
        "top_players": [{"name": k, "n": int(v)} for k, v in by_player.items()],
        "mean_bypassed": round(float(bypass.mean()), 2) if len(bypass) else None,
        "clips": clips,
    }


@metric(
    "threat_map",
    "Карта угрозы",
    "Атака",
    requires=(C.XTHREAT,),
    note="xThreat — модельный прирост вероятности гола от действия. "
    "Сетка 6×5, атака слева направо.",
)
def threat_map(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    opts = ev[(ev["type"] == "passing_option") & (ev["possession_team_id"] == tid)]
    own = on_ball(ev, tid).dropna(subset=["xt"])
    opp = opponent_on_ball(ev, tid).dropna(subset=["xt"])
    if own.empty:
        return {}
    grid = zone_grid(own, "xt")
    n_matches = max(len(ms), 1)
    ch = own.groupby(own["channel_start"].map(CHANNEL_RU))["xt"].sum()
    ch = [
        {"key": CHANNEL_RU[c], "value": round(float(ch.get(CHANNEL_RU[c], 0)) / n_matches, 2)}
        for c in CHANNEL_ORDER
    ]
    return {
        "grid": (grid / n_matches).round(3).tolist(),
        "total": round(float(own["xt"].sum()) / n_matches, 2),
        "conceded": round(float(opp["xt"].sum()) / n_matches, 2) if len(opp) else None,
        "by_channel": ch,
        "potential": round(float(opts["xthreat"].fillna(0).sum()) / n_matches, 2),
        "potential_taken": round(
            100
            * float(opts[opts["targeted"] == True]["xthreat"].fillna(0).sum())  # noqa: E712
            / max(float(opts["xthreat"].fillna(0).sum()), 1e-9),
            1,
        ),
        "top_players": [
            {"name": k, "value": round(float(v) / n_matches, 2)}
            for k, v in own.groupby("player_name")["xt"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
            .items()
        ],
        "clips": episodes(
            own.nlargest(30, "xt"),
            lambda r: f"{r['player_name']}, xT {float(r['xt']):.3f} — {r.get('phase', '')}",
            30,
        ),
    }


@metric(
    "shots",
    "Удары",
    "Атака",
    requires=(C.EVENTS,),
    note="Владения, закончившиеся ударом, с разбивкой по фазе атаки "
    "и по тому, как команда получила мяч перед этим.",
)
def shots(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    own = on_ball(ev, tid)
    sh = own[own["end_type"] == "shot"]
    if sh.empty:
        return {}
    n_matches = max(len(ms), 1)
    goals_rows = own[own["lead_to_goal"] == True] if "lead_to_goal" in own else own.iloc[0:0]  # noqa: E712

    # Из какой фазы рождается удар.
    by_phase = share(sh["phase"])
    # После какого начала владения.
    by_start = share(sh["start_type"])
    return {
        "total": int(len(sh)),
        "per_match": round(len(sh) / n_matches, 1),
        "by_phase": by_phase,
        "by_start": by_start[:6],
        "top_players": [
            {"name": k, "n": int(v)}
            for k, v in sh.groupby("player_name")
            .size()
            .sort_values(ascending=False)
            .head(8)
            .items()
        ],
        "clips": episodes(
            sh.sort_values("t"), lambda r: f"Удар — {r['player_name']}, {r.get('phase', '')}", 40
        ),
        "goal_episodes": episodes(
            goals_rows.sort_values("t"), lambda r: f"Гол — атака через {r.get('phase', '')}", 20
        ),
    }


@metric(
    "losses",
    "Потери мяча",
    "Атака",
    requires=(C.EVENTS,),
    note="Потеря — владение, закончившееся отбором соперника или неточным действием.",
)
def losses(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    own = on_ball(ev, tid)
    lost = (
        own[own["end_type"].isin(["possession_loss"]) | (own["pass_outcome"] == "unsuccessful")]
        if "pass_outcome" in own
        else own[own["end_type"] == "possession_loss"]
    )
    if lost.empty:
        return {}
    grid = zone_grid(lost)
    thirds = share(lost["third_start"].map(THIRD_RU), [THIRD_RU[t] for t in THIRD_ORDER])
    own_third = lost[lost["third_start"] == "defensive_third"]
    return {
        "grid": grid.tolist(),
        "total": int(len(lost)),
        "per_match": round(len(lost) / max(len(ms), 1), 1),
        "thirds": thirds,
        "own_third": int(len(own_third)),
        "own_third_share": round(100 * len(own_third) / len(lost), 1),
        "top_players": [
            {"name": k, "n": int(v)}
            for k, v in lost.groupby("player_name")
            .size()
            .sort_values(ascending=False)
            .head(6)
            .items()
        ],
        "clips": episodes(
            own_third.sort_values("t"), lambda r: f"Потеря в своей трети — {r['player_name']}", 30
        ),
    }


@metric(
    "set_pieces",
    "Стандарты",
    "Стандарты",
    requires=(C.EVENTS,),
    note="Владения, начавшиеся с углового, штрафного, аута или удара от ворот.",
)
def set_pieces(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    own = on_ball(ev, tid)
    if "game_interruption_before" not in own.columns:
        return {}
    sp = own[own["game_interruption_before"].isin(SET_PIECE_RU.keys())]
    if sp.empty:
        return {}
    n_matches = max(len(ms), 1)
    rows = []
    for k, ru in SET_PIECE_RU.items():
        g = sp[sp["game_interruption_before"] == k]
        if not len(g):
            continue
        rows.append(
            {
                "kind": ru,
                "n": int(len(g)),
                "per_match": round(len(g) / n_matches, 1),
                "shot_rate": round(100 * float(g["lead_to_shot"].fillna(False).mean()), 1),
                "xthreat": round(float(g["xt"].fillna(0).sum()) / n_matches, 2),
            }
        )
    corners = sp[sp["game_interruption_before"] == "corner_for"]
    targets = []
    if "player_targeted_y_reception" in corners.columns:
        c = corners.dropna(subset=["player_targeted_x_reception", "player_targeted_y_reception"])
        targets = [
            {
                "x": round(float(r["player_targeted_x_reception"]), 1),
                "y": round(float(r["player_targeted_y_reception"]), 1),
                "player": str(r.get("player_targeted_name") or ""),
                "shot": bool(r.get("lead_to_shot")),
            }
            for _, r in c.iterrows()
        ][:60]
    return {
        "rows": rows,
        "total": int(len(sp)),
        "corner_targets": targets,
        "clips": episodes(
            sp[sp["lead_to_shot"] == True].sort_values("t"),  # noqa: E712
            lambda r: f"Стандарт с ударом — {SET_PIECE_RU.get(r['game_interruption_before'], '')}",
            30,
        ),
    }


@metric(
    "time_profile",
    "Профиль по времени матча",
    "Ритм",
    requires=(C.XTHREAT,),
    note="Как распределены угроза, удары и потери по 15-минутным отрезкам.",
)
def time_profile(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    own = on_ball(ev, tid).copy()
    opp = opponent_on_ball(ev, tid).copy()
    if own.empty:
        return {}
    n_matches = max(len(ms), 1)

    def bucketize(df: pd.DataFrame) -> pd.DataFrame:
        b = np.minimum((df["minute"].fillna(0).astype(int) // 15), 5)
        return df.assign(bucket=b)

    own, opp = bucketize(own), bucketize(opp)
    labels = ["0–15", "15–30", "30–45", "45–60", "60–75", "75–90"]
    rows: list[dict[str, Any]] = []
    for i, lab in enumerate(labels):
        o = own[own["bucket"] == i]
        d = opp[opp["bucket"] == i]
        rows.append(
            {
                "bucket": lab,
                "xthreat": round(float(o["xt"].fillna(0).sum()) / n_matches, 2),
                "xthreat_against": round(float(d["xt"].fillna(0).sum()) / n_matches, 2),
                "shots": int((o["end_type"] == "shot").sum()),
                "shots_against": int((d["end_type"] == "shot").sum()),
                "losses": int((o["end_type"] == "possession_loss").sum()),
            }
        )
    peak = max(rows, key=lambda r: r["xthreat"])
    vuln = max(rows, key=lambda r: r["xthreat_against"])
    pi = labels.index(peak["bucket"])
    vi = labels.index(vuln["bucket"])
    return {
        "rows": rows,
        "clips_peak": episodes(
            own[(own["bucket"] == pi) & (own["end_type"] == "shot")].sort_values("t"),
            lambda r: f"{int(r['minute']) + 1}′ удар — {r['player_name']}",
            30,
        ),
        "clips_vuln": episodes(
            opp[(opp["bucket"] == vi) & (opp["end_type"] == "shot")].sort_values("t"),
            lambda r: f"{int(r['minute']) + 1}′ удар соперника — {r['player_name']}",
            30,
        ),
    }
