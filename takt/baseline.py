"""Лига-бенчмарк.

Без контекста число «высота линии 36 метров» ничего не говорит тренеру.
Оно начинает работать только рядом с «в лиге 33». Поэтому те же метрики
считаются для всех команд, по которым есть данные, и берётся медиана.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .metrics import Computed, compute_all
from .model import Match, MatchSet, Team

# Что сравниваем с лигой: ключ метрики -> путь к скаляру -> подпись.
SCALARS: dict[str, tuple[str, str, str]] = {
    "ppda": ("ppda", "value", "PPDA"),
    "line_height": ("defensive_line", "mean", "высота линии, м"),
    "block_width": ("block_shape", "mean_width", "ширина блока, м"),
    "block_length": ("block_shape", "mean_length", "длина блока, м"),
    "possession": ("possession_profile", "possession_share", "владение, %"),
    "phase_duration": ("possession_profile", "mean_phase_duration", "длина фазы, с"),
    "shot_rate": ("possession_profile", "shot_rate", "фаз с ударом, %"),
    "threat": ("threat_map", "total", "xT за матч"),
    "threat_taken": ("threat_map", "potential_taken", "реализация вариантов, %"),
    "runs": ("off_ball_runs", "per_match", "забеганий за матч"),
    "runs_used": ("off_ball_runs", "used_share", "забеганий использовано, %"),
    "behind": ("off_ball_runs", "behind_per_match", "забеганий за спину"),
    "options": ("passing_options", "mean_options", "вариантов передачи"),
    "breaks": ("progression", "per_match", "взломов линий за матч"),
    "losses": ("losses", "per_match", "потерь за матч"),
    "high_regains": ("regain_zones", "high_share", "отборов на чужой половине, %"),
}


@dataclass
class Baseline:
    """Медиана по всем командам выборки + отдельные значения по каждой."""

    median: dict[str, float] = field(default_factory=dict)
    spread: dict[str, float] = field(default_factory=dict)  # межквартильный размах
    by_team: dict[str, dict[str, float]] = field(default_factory=dict)
    n_teams: int = 0
    n_matches: int = 0

    def compare(self, key: str, value: float | None) -> dict | None:
        """Насколько значение отличается от лиги. delta в единицах разброса."""
        if value is None or key not in self.median:
            return None
        med = self.median[key]
        sp = self.spread.get(key) or 0.0
        delta = value - med
        z = delta / sp if sp else 0.0
        return {
            "value": round(value, 2),
            "median": round(med, 2),
            "delta": round(delta, 2),
            "z": round(z, 2),
            "label": SCALARS[key][2],
            "rank": self._rank(key, value),
        }

    def _rank(self, key: str, value: float) -> str:
        vals = sorted(v[key] for v in self.by_team.values() if key in v)
        if not vals:
            return ""
        below = sum(1 for v in vals if v < value)
        pct = 100 * below / len(vals)
        if pct >= 80:
            return "заметно выше лиги"
        if pct >= 60:
            return "выше лиги"
        if pct <= 20:
            return "заметно ниже лиги"
        if pct <= 40:
            return "ниже лиги"
        return "на уровне лиги"


def _extract(computed: Computed, path: tuple[str, str]) -> float | None:
    block = computed.get(path[0])
    if not isinstance(block, dict):
        return None
    v = block.get(path[1])
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def build_baseline(matches: list[Match], min_matches: int = 1) -> Baseline:
    """Посчитать те же метрики для каждой команды выборки."""
    by_team_matches: dict[int, list[Match]] = {}
    teams: dict[int, Team] = {}
    for m in matches:
        for t in (m.home, m.away):
            by_team_matches.setdefault(t.id, []).append(m)
            teams[t.id] = t

    per_team: dict[str, dict[str, float]] = {}
    for tid, ms_list in by_team_matches.items():
        if len(ms_list) < min_matches:
            continue
        ms = MatchSet(subject_team=teams[tid], matches=ms_list)
        c = compute_all(ms)
        vals = {}
        for key, (metric_key, field_name, _) in SCALARS.items():
            v = _extract(c, (metric_key, field_name))
            if v is not None:
                vals[key] = v
        per_team[teams[tid].short_name] = vals

    median: dict[str, float] = {}
    spread: dict[str, float] = {}
    for key in SCALARS:
        series = [v[key] for v in per_team.values() if key in v]
        if len(series) >= 3:
            median[key] = statistics.median(series)
            q = sorted(series)
            lo = statistics.median(q[: len(q) // 2])
            hi = statistics.median(q[(len(q) + 1) // 2 :])
            spread[key] = max(hi - lo, 1e-6)

    return Baseline(
        median=median,
        spread=spread,
        by_team=per_team,
        n_teams=len(per_team),
        n_matches=len(matches),
    )
