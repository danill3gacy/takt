"""Сборка отчёта: метрики + тезисы + паттерны + плейлист → одна модель данных."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .baseline import SCALARS, Baseline
from .insights import Insight, build_insights
from .metrics import REGISTRY, Computed, compute_all
from .metrics.common import CHANNEL_RU, RUN_RU, episodes, phase_episodes
from .model import Capability, MatchSet

# --------------------------------------------------------------------------- #
# Поиск по тактическим паттернам
# --------------------------------------------------------------------------- #
# Каждый паттерн — детерминированный фильтр по событиям, а не запрос к модели.
# Словарь синонимов нужен только чтобы понять, какой фильтр применить.
# В боевом контуре сюда встаёт разбор фразы языковой моделью, но сам поиск
# остаётся тем же: он должен быть воспроизводимым.

PATTERNS: list[dict] = [
    {
        "id": "high_press",
        "name": "Высокий прессинг на чужой половине",
        "words": ["высокий прессинг", "прессинг", "давление", "отбор высоко",
                  "прессинг на чужой", "агрессия", "накрывают"],
        "hint": "Оборонительное действие выше средней линии",
    },
    {
        "id": "counter_press",
        "name": "Контрпрессинг сразу после потери",
        "words": ["контрпрессинг", "после потери", "сразу отбор", "5 секунд",
                  "возврат мяча", "gegenpressing"],
        "hint": "Отбор в первые секунды после потери владения",
    },
    {
        "id": "run_behind",
        "name": "Забегание за спину защитникам",
        "words": ["за спину", "забегание", "рывок за спину", "в разрез",
                  "глубина", "убегает"],
        "hint": "Движение без мяча за последнюю линию обороны",
    },
    {
        "id": "line_break",
        "name": "Взлом последней линии обороны",
        "words": ["взлом", "разрезающая", "передача между линиями", "вскрытие",
                  "проникающая", "последняя линия"],
        "hint": "Передача или ведение за последнюю линию",
    },
    {
        "id": "switch",
        "name": "Смена фланга",
        "words": ["смена фланга", "перевод", "длинный перевод", "с фланга на фланг",
                  "диагональ"],
        "hint": "Передача с поперечным смещением больше 25 метров",
    },
    {
        "id": "loss_own_third",
        "name": "Потеря в своей трети",
        "words": ["потеря", "потеря сзади", "ошибка при выходе", "обрез",
                  "потеря в своей трети"],
        "hint": "Владение потеряно в защитной трети",
    },
    {
        "id": "corner",
        "name": "Угловые",
        "words": ["угловой", "корнер", "подача с углового", "стандарт"],
        "hint": "Владение, начавшееся с углового",
    },
    {
        "id": "low_block",
        "name": "Низкий блок",
        "words": ["низкий блок", "оборона у своей штрафной", "садятся",
                  "автобус", "глубокая оборона"],
        "hint": "Фаза обороны в низком блоке",
    },
    {
        "id": "quick_break",
        "name": "Быстрый выпад и переход",
        "words": ["контратака", "быстрый выпад", "переход", "быстрая атака",
                  "переход из обороны в атаку"],
        "hint": "Фаза быстрого перехода в атаку",
    },
    {
        "id": "cross",
        "name": "Открывание под навес",
        "words": ["навес", "подача", "кросс", "под навес", "с фланга в штрафную"],
        "hint": "Забегание под подачу в штрафную",
    },
    {
        "id": "shot",
        "name": "Удары",
        "words": ["удар", "момент", "завершение", "по воротам"],
        "hint": "Владение, закончившееся ударом",
    },
]


def _build_patterns(ms: MatchSet, c: Computed) -> list[dict]:
    ev = ms.events()
    ph = ms.phases()
    tid = ms.subject_team.id
    own = ev[(ev["type"] == "player_possession") & (ev["possession_team_id"] == tid)]
    runs = ev[(ev["type"] == "off_ball_run") & (ev["possession_team_id"] == tid)]
    eng = ev[(ev["type"] == "on_ball_engagement") & (ev["team_id"] == tid)]

    sel: dict[str, list[dict]] = {}

    sel["high_press"] = episodes(
        eng[(eng["x"] > 0) & (eng["end_type"].notna())].sort_values("t"),
        lambda r: f"{r['player_name']} — отбор на чужой половине, {r.get('def_phase','')}", 60)

    sel["counter_press"] = episodes(
        eng[eng["subtype"] == "counter_press"].sort_values("t"),
        lambda r: f"Контрпрессинг — {r['player_name']}, {r.get('def_phase','')}", 60)

    sel["run_behind"] = episodes(
        runs[runs["subtype"] == "behind"].sort_values("t"),
        lambda r: f"Забегание за спину — {r['player_name']}"
                  + (", адресовано" if r.get("targeted") else ", передачи не последовало"), 60)

    sel["line_break"] = episodes(
        own[own["furthest_line_break"] == "last"].sort_values("t"),
        lambda r: f"Взлом последней линии — {r['player_name']}, {r.get('phase','')}", 60)

    sw = own[(own["end_type"] == "pass") & ((own["y_end"] - own["y"]).abs() > 25)]
    sel["switch"] = episodes(
        sw.sort_values("t"),
        lambda r: f"Смена фланга — {r['player_name']}, {abs(r['y_end']-r['y']):.0f} м поперёк", 60)

    sel["loss_own_third"] = episodes(
        own[(own["end_type"] == "possession_loss") &
            (own["third_start"] == "defensive_third")].sort_values("t"),
        lambda r: f"Потеря в своей трети — {r['player_name']}", 60)

    sel["corner"] = episodes(
        own[own["game_interruption_before"] == "corner_for"].sort_values("t"),
        lambda r: f"Угловой — {r['player_name']}"
                  + (" → удар" if r.get("lead_to_shot") else ""), 60)

    if not ph.empty:
        low = ph[(ph["team_id"] != tid) & (ph["def_phase"] == "низкий блок")]
        sel["low_block"] = phase_episodes(
            low.sort_values("t"),
            lambda r: f"Низкий блок {r['duration']:.0f} с, ширина {r['width_out']:.0f} м", 60)
        qb = ph[(ph["team_id"] == tid) & (ph["phase"].isin(["быстрый выпад", "переход"]))]
        sel["quick_break"] = phase_episodes(
            qb.sort_values("t"),
            lambda r: f"{r['phase']} — {r['duration']:.0f} с"
                      + (" → удар" if r.get("lead_to_shot") else ""), 60)

    sel["cross"] = episodes(
        runs[runs["subtype"] == "cross_receiver"].sort_values("t"),
        lambda r: f"Открывание под навес — {r['player_name']}", 60)

    sel["shot"] = episodes(
        own[own["end_type"] == "shot"].sort_values("t"),
        lambda r: f"Удар — {r['player_name']}, {r.get('phase','')}", 60)

    out = []
    for p in PATTERNS:
        clips = sel.get(p["id"], [])
        if not clips:
            continue
        out.append({**p, "n": len(clips), "clips": clips})
    return out


# --------------------------------------------------------------------------- #

def _capability_matrix(ms: MatchSet, c: Computed) -> list[dict]:
    """Что посчитано, что недоступно и от чего это зависит.

    Это не служебная информация, а часть разговора с клубом: по этой таблице
    видно, какие блоки отчёта появятся сразу после подключения к фиду лиги,
    а какие требуют дополнительных полей.
    """
    caps = ms.capabilities
    rows = []
    for key, spec in REGISTRY.items():
        need = [r.value for r in spec.requires]
        missing = [r.value for r in spec.requires if r not in caps]
        value = c.get(key)
        rows.append({
            "title": spec.title,
            "section": spec.section,
            "requires": need,
            "status": "нет данных" if missing else ("посчитано" if value else "пусто"),
            "missing": missing,
            "note": spec.note,
        })
    return rows


@dataclass
class Report:
    club: str
    club_short: str
    opponent: str
    opponent_full: str
    generated: str
    n_matches: int
    minutes: float
    source: str
    competition: str
    matches: list[dict] = field(default_factory=list)
    insights: list[dict] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    patterns: list[dict] = field(default_factory=list)
    baseline: dict[str, Any] = field(default_factory=dict)
    capabilities: list[dict] = field(default_factory=list)
    caps_present: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_report(ms: MatchSet, bl: Baseline, *, club: str, club_short: str,
                 generated: str) -> Report:
    c = compute_all(ms)
    ins = build_insights(c, bl, ms.subject_team.name)

    matches = []
    for m in ms.matches:
        opp = m.opponent_of(ms.subject_team.id)
        home = m.home.id == ms.subject_team.id
        gf, ga = (m.score if home else (m.score[1], m.score[0]))
        matches.append({
            "id": m.id,
            "date": m.date,
            "opponent": opp.short_name,
            "venue": "дома" if home else "в гостях",
            "score": f"{gf}:{ga}",
            "result": "П" if gf > ga else ("Н" if gf == ga else "Пор"),
            "label": m.label(),
        })

    baseline = {
        "n_teams": bl.n_teams,
        "n_matches": bl.n_matches,
        "rows": [
            {"key": k, "label": SCALARS[k][2],
             "median": round(v, 2),
             "team": round(bl.by_team.get(ms.subject_team.short_name, {}).get(k), 2)
             if bl.by_team.get(ms.subject_team.short_name, {}).get(k) is not None else None}
            for k, v in bl.median.items()
        ],
    }

    return Report(
        club=club,
        club_short=club_short,
        opponent=ms.subject_team.short_name,
        opponent_full=ms.subject_team.name,
        generated=generated,
        n_matches=len(ms),
        minutes=round(ms.total_minutes()),
        source=ms.matches[0].source if ms.matches else "",
        competition=ms.matches[0].competition if ms.matches else "",
        matches=matches,
        insights=[{**asdict(i), "level": i.level} for i in ins],
        metrics=c.values,
        patterns=_build_patterns(ms, c),
        baseline=baseline,
        capabilities=_capability_matrix(ms, c),
        caps_present=sorted(x.value for x in ms.capabilities),
    )
