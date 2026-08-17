"""Общие помощники для метрик: фильтры, зоны, сборка эпизодов."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ..model import MatchSet

CHANNEL_RU = {
    "wide_left": "левый фланг",
    "half_space_left": "левый полуфланг",
    "center": "центр",
    "half_space_right": "правый полуфланг",
    "wide_right": "правый фланг",
}
CHANNEL_ORDER = ["wide_left", "half_space_left", "center", "half_space_right", "wide_right"]

THIRD_RU = {
    "defensive_third": "своя треть",
    "middle_third": "средняя треть",
    "attacking_third": "чужая треть",
}
THIRD_ORDER = ["defensive_third", "middle_third", "attacking_third"]

RUN_RU = {
    "behind": "забегание за спину",
    "coming_short": "открывание в ноги",
    "cross_receiver": "под навес",
    "dropping_off": "оттяжка",
    "overlap": "обегание снаружи",
    "underlap": "забегание внутрь",
    "pulling_half_space": "смещение в полуфланг",
    "pulling_wide": "растягивание на фланг",
    "run_ahead_of_the_ball": "рывок вперёд мяча",
    "support": "поддержка",
}

PRESS_RU = {
    "pressing": "прессинг",
    "pressure": "давление",
    "counter_press": "контрпрессинг",
    "recovery_press": "возврат в отбор",
    "other": "прочее",
}

END_RU = {
    "pass": "передача",
    "shot": "удар",
    "possession_loss": "потеря",
    "clearance": "вынос",
    "foul_suffered": "фол на нём",
    "foul_committed": "фол",
    "direct_regain": "прямой отбор",
    "indirect_regain": "отбор через партнёра",
    "direct_disruption": "срыв атаки",
    "indirect_disruption": "срыв через партнёра",
    "unknown": "не определено",
}

SET_PIECE_RU = {
    "corner_for": "угловой",
    "free_kick_for": "штрафной",
    "throw_in_for": "аут",
    "goal_kick_for": "удар от ворот",
}


# --------------------------------------------------------------------------- #
# Срезы
# --------------------------------------------------------------------------- #


def on_ball(ev: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """Владения мячом игроками команды."""
    return ev[(ev["type"] == "player_possession") & (ev["possession_team_id"] == team_id)]


def off_ball_runs(ev: pd.DataFrame, team_id: int) -> pd.DataFrame:
    return ev[(ev["type"] == "off_ball_run") & (ev["possession_team_id"] == team_id)]


def passing_options(ev: pd.DataFrame, team_id: int) -> pd.DataFrame:
    return ev[(ev["type"] == "passing_option") & (ev["possession_team_id"] == team_id)]


def engagements(ev: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """Оборонительные действия команды (она БЕЗ мяча)."""
    return ev[(ev["type"] == "on_ball_engagement") & (ev["team_id"] == team_id)]


def opponent_on_ball(ev: pd.DataFrame, team_id: int) -> pd.DataFrame:
    return ev[(ev["type"] == "player_possession") & (ev["possession_team_id"] != team_id)]


# --------------------------------------------------------------------------- #
# Зоны
# --------------------------------------------------------------------------- #


def zone_grid(
    df: pd.DataFrame,
    value: str | None = None,
    nx: int = 6,
    ny: int = 5,
    length: float = 105.0,
    width: float = 68.0,
) -> np.ndarray:
    """Сетка nx×ny: сумма value (или количество) по зонам поля."""
    if df.empty:
        return np.zeros((ny, nx))
    xi = np.clip(((df["x"] + length / 2) / length * nx).astype(int), 0, nx - 1)
    yi = np.clip(((df["y"] + width / 2) / width * ny).astype(int), 0, ny - 1)
    grid = np.zeros((ny, nx))
    vals = df[value].fillna(0).to_numpy() if value else np.ones(len(df))
    for a, b, v in zip(yi.to_numpy(), xi.to_numpy(), vals):
        grid[a, b] += v
    return grid


def share(series: pd.Series, order: list[str] | None = None) -> list[dict]:
    """Распределение категориальной величины в долях, отсортированное."""
    if series.empty:
        return []
    vc = series.value_counts()
    total = vc.sum()
    keys = order if order else vc.index.tolist()
    out = []
    for k in keys:
        if k in vc.index:
            out.append({"key": k, "n": int(vc[k]), "share": round(100 * vc[k] / total, 1)})
    if not order:
        out.sort(key=lambda r: -r["n"])
    return out


# --------------------------------------------------------------------------- #
# Эпизоды (клипы)
# --------------------------------------------------------------------------- #


def episodes(
    df: pd.DataFrame, title_fn: Callable[..., str] | None = None, limit: int = 40
) -> list[dict]:
    """Превратить строки событий в список клипов с тайм-кодами."""
    out = []
    for _, r in df.head(limit).iterrows():
        t = float(r.get("t", 0) or 0)
        out.append(
            {
                "match_id": str(r.get("match_id", "")),
                "match": str(r.get("match_label", "")),
                "minute": int(r["minute"]) + 1 if pd.notna(r.get("minute")) else 0,
                "clock": f"{int(r['minute']) + 1}′" if pd.notna(r.get("minute")) else "",
                "t": round(t, 1),
                "timecode": f"{int(t // 60):02d}:{int(t % 60):02d}",
                "frame": int(r["frame"]) if pd.notna(r.get("frame")) else 0,
                "player": str(r.get("player_name") or ""),
                "phase": str(r.get("phase") or ""),
                "def_phase": str(r.get("def_phase") or ""),
                "title": title_fn(r) if title_fn else "",
            }
        )
    return out


def phase_episodes(
    df: pd.DataFrame, title_fn: Callable[..., str] | None = None, limit: int = 30
) -> list[dict]:
    """Клипы из таблицы фаз (у фазы есть кадр начала и длительность)."""
    out = []
    for _, r in df.head(limit).iterrows():
        t = float(r.get("t", 0) or 0)
        out.append(
            {
                "match_id": str(r.get("match_id", "")),
                "match": str(r.get("match_label", "")),
                "minute": int(r["minute"]) + 1 if pd.notna(r.get("minute")) else 0,
                "clock": f"{int(r['minute']) + 1}\u2032" if pd.notna(r.get("minute")) else "",
                "t": round(t, 1),
                "timecode": f"{int(t // 60):02d}:{int(t % 60):02d}",
                "frame": int(r["frame"]) if pd.notna(r.get("frame")) else 0,
                "player": "",
                "phase": str(r.get("phase") or ""),
                "def_phase": str(r.get("def_phase") or ""),
                "title": title_fn(r) if title_fn else "",
            }
        )
    return out


def per90(n: float, minutes: float) -> float:
    return round(n / minutes * 90, 2) if minutes else 0.0


def league_baseline(all_sets: list[MatchSet], fn: Callable[[MatchSet], float]) -> float:
    """Среднее по всем командам датасета — контекст «много это или мало»."""
    vals = [fn(s) for s in all_sets]
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")
