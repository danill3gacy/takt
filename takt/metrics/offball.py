"""Движение без мяча: забегания, варианты передачи, растягивание обороны.

Это тот блок, который невозможно получить из событийной разметки руками —
он существует только там, где есть трекинг всех 22 игроков.
"""

from __future__ import annotations

import pandas as pd

from ..model import Capability as C
from ..model import MatchSet
from .common import (
    CHANNEL_ORDER,
    CHANNEL_RU,
    RUN_RU,
    episodes,
    off_ball_runs,
    on_ball,
    passing_options,
    share,
)
from .registry import metric


@metric("off_ball_runs", "Забегания без мяча", "Движение без мяча",
        requires=(C.OFF_BALL_RUNS,),
        note="Забегание засчитывается, когда игрок открывается под передачу "
             "партнёра с мячом. «Использовано» — партнёр отдал именно туда.")
def off_ball_runs_metric(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    runs = off_ball_runs(ev, tid)
    if runs.empty:
        return {}
    n_matches = max(len(ms), 1)

    rows = []
    for k, g in runs.groupby("subtype"):
        targeted = g["targeted"].fillna(False)
        received = g["received"].fillna(False)
        rows.append({
            "kind": RUN_RU.get(k, k),
            "raw": k,
            "n": int(len(g)),
            "per_match": round(len(g) / n_matches, 1),
            "share": round(100 * len(g) / len(runs), 1),
            "used": round(100 * float(targeted.mean()), 1),
            "received": round(100 * float(received.mean()), 1),
            "dangerous": round(100 * float(g["dangerous"].fillna(False).mean()), 1)
            if "dangerous" in g else None,
            "distance": round(float(g["distance_covered"].dropna().mean()), 1)
            if "distance_covered" in g else None,
            "speed": round(float(g["speed_avg"].dropna().mean()), 1)
            if "speed_avg" in g else None,
        })
    rows.sort(key=lambda r: -r["n"])

    behind = runs[runs["subtype"] == "behind"]
    breaking = runs[runs["break_defensive_line"] == True] if "break_defensive_line" in runs else runs.iloc[0:0]  # noqa: E712
    pushing = runs[runs["push_defensive_line"] == True] if "push_defensive_line" in runs else runs.iloc[0:0]  # noqa: E712

    by_player = (runs.groupby("player_name")
                 .agg(n=("event_id", "size"),
                      used=("targeted", lambda s: 100 * s.fillna(False).mean()))
                 .sort_values("n", ascending=False).head(10))

    sep = runs["separation_gain"].dropna() if "separation_gain" in runs else pd.Series(dtype=float)
    return {
        "rows": rows,
        "total": int(len(runs)),
        "per_match": round(len(runs) / n_matches, 1),
        "used_share": round(100 * float(runs["targeted"].fillna(False).mean()), 1),
        "behind": int(len(behind)),
        "behind_per_match": round(len(behind) / n_matches, 1),
        "behind_used": round(100 * float(behind["targeted"].fillna(False).mean()), 1)
        if len(behind) else None,
        "break_line": int(len(breaking)),
        "push_line": int(len(pushing)),
        "mean_separation_gain": round(float(sep.mean()), 2) if len(sep) else None,
        "top_players": [{"name": k, "n": int(r["n"]), "used": round(float(r["used"]), 1)}
                        for k, r in by_player.iterrows()],
        "clips": episodes(behind.sort_values("t"),
                          lambda r: f"Забегание за спину — {r['player_name']}, {r.get('phase','')}", 30),
        "clips_all": episodes(
            runs[runs["dangerous"] == True].sort_values("t"),  # noqa: E712
            lambda r: f"{RUN_RU.get(r['subtype'], r['subtype'])} — {r['player_name']}", 30),
    }


@metric("passing_options", "Варианты передачи", "Движение без мяча",
        requires=(C.PASSING_OPTIONS,),
        note="Сколько открытых партнёров у игрока с мячом и как часто команда "
             "выбирает самый опасный из доступных.")
def passing_options_metric(ms: MatchSet) -> dict:
    ev = ms.events()
    tid = ms.subject_team.id
    own = on_ball(ev, tid)
    opts = passing_options(ev, tid)
    if own.empty or opts.empty:
        return {}

    n_opts = own["n_passing_options"].dropna()
    n_break = own["n_passing_options_line_break"].dropna()
    n_ahead = own["n_passing_options_ahead"].dropna()

    # Выбирают ли опасный вариант, когда он есть.
    dangerous = opts[opts["dangerous"] == True]  # noqa: E712
    dang_taken = dangerous["targeted"].fillna(False).mean() if len(dangerous) else None
    safe = opts[opts["dangerous"] != True]  # noqa: E712
    safe_taken = safe["targeted"].fillna(False).mean() if len(safe) else None

    return {
        "mean_options": round(float(n_opts.mean()), 2) if len(n_opts) else None,
        "mean_line_break_options": round(float(n_break.mean()), 2) if len(n_break) else None,
        "mean_forward_options": round(float(n_ahead.mean()), 2) if len(n_ahead) else None,
        "dangerous_taken": round(100 * float(dang_taken), 1) if dang_taken is not None else None,
        "safe_taken": round(100 * float(safe_taken), 1) if safe_taken is not None else None,
        "n_options": int(len(opts)),
        "channels": share(opts[opts["targeted"] == True]["channel_start"].map(CHANNEL_RU),  # noqa: E712
                          [CHANNEL_RU[c] for c in CHANNEL_ORDER]),
        "clips": episodes(
            dangerous[dangerous["targeted"] != True].nlargest(30, "xthreat"),  # noqa: E712
            lambda r: f"Открыт опасный вариант, передача не туда — {r['player_name']}, "
                      f"xT {float(r['xthreat'] or 0):.3f}", 30),
    }
