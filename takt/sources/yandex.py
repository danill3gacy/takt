"""Адаптер фида лиги (Яндекс×VSporte / РУСТАТ) — заготовка с явным контрактом.

Это не заглушка «на будущее», а рабочая точка подключения. Весь остальной
код — модель, метрики, тезисы, отчёт — не знает, откуда пришли данные.
Чтобы отчёт начал считаться на данных РПЛ, нужно заполнить три метода этого
файла и объявить, какие возможности реально даёт фид.

ЧТО НУЖНО УТОЧНИТЬ У ЛИГИ/КЛУБА ДО НАЧАЛА РАБОТЫ
------------------------------------------------
Порядок вопросов важен: первые два определяют, проект это на три месяца или
на полтора года.

1. Отдаётся ли СЫРОЙ ТРЕКИНГ (координаты 22 игроков и мяча, 10–25 Гц) или
   только агрегаты и PDF-отчёты?
     — есть трекинг            → Capability.TRACKING, всё считается у нас;
     — только производные      → смотрим пункт 2;
     — только PDF/CSV-сводки   → пространственный блок отчёта недоступен,
                                 остаётся событийный. Это видно в отчёте
                                 на вкладке «Фид и метод» — врать не придётся.

2. Есть ли ПРОИЗВОДНЫЕ СОБЫТИЯ поверх трекинга: забегания без мяча, цепочки
   прессинга, взломы линий, варианты передачи, ширина/длина блока?
   Если да — большая часть метрик переносится один-в-один, как в адаптере
   SkillCorner. Если нет — их надо считать у себя из сырого трекинга;
   это отдельный этап работ, но он не меняет ни модель, ни отчёт.

3. Формат выгрузки: REST по расписанию, S3, файлы на диск? Частота?
   Сколько матчей в истории доступно (глубина выборки задаёт, сколько
   матчей соперника попадёт в отчёт).

4. Соответствие идентификаторов: id игроков и команд лиги ↔ id клуба.
   Без стабильного маппинга профили игроков не склеятся между матчами.

5. Есть ли тайм-коды видео в той же системе кадров, что и данные?
   Это то, что превращает тезис в клип. Если тайм-коды в другой системе,
   нужен один коэффициент синхронизации на матч.

КАК ЗАПОЛНЯТЬ
-------------
Каноническая таблица событий описана в sources/base.py. Обязательный минимум:
event_id, period, minute, second, t, frame, team_id, player_id, type, x, y.
Всё остальное — по мере наличия; метрики сами разберутся по capabilities.

Координаты приводить к метрам с началом в центре поля, ось x — в сторону
атаки команды, владеющей мячом (см. docstring в model.py). Если фид отдаёт
координаты в фиксированной системе стадиона, разворот делается здесь и
больше нигде.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ..model import Capability, Match
from .base import Source


class LeagueFeedSource(Source):
    """Подключение к фиду лиги. Заполняется после ответа на вопросы выше."""

    name = "Фид лиги (Яндекс×VSporte / РУСТАТ)"

    # Объявляется по факту, а не по надежде. Пустой набор означает, что
    # отчёт покажет все метрики как «нет данных» — это корректное поведение,
    # а не поломка.
    capabilities: frozenset[Capability] = frozenset()

    def __init__(self, base_url: str | None = None, token: str | None = None,
                 cache_dir: str | Path = "data/league_cache"):
        self.base_url = base_url or os.getenv("LEAGUE_FEED_URL", "")
        self.token = token or os.getenv("LEAGUE_FEED_TOKEN", "")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #

    def list_matches(self, team: str | int | None = None) -> list[dict]:
        raise NotImplementedError(
            "Подставить вызов каталога матчей фида. Ожидаемый формат строки: "
            "{'id', 'date', 'home', 'away', 'home_id', 'away_id'}."
        )

    def load(self, match_id: str | int) -> Match:
        raise NotImplementedError(
            "Собрать Match из ответа фида: метаданные, состав, каноническая "
            "таблица событий (см. sources/base.CANONICAL_EVENT_COLUMNS), при "
            "наличии — таблица фаз и кадры трекинга. Разворот координат в "
            "систему атакующей команды выполняется здесь."
        )

    # ------------------------------------------------------------------ #

    def probe(self) -> dict:
        """Диагностика фида: что реально приходит.

        Запускается первой же командой после получения доступа. Возвращает
        отчёт «какие поля есть», по которому заполняется capabilities —
        и по которому сразу видно, какие блоки отчёта появятся у клуба.
        """
        raise NotImplementedError(
            "Скачать один матч, разобрать структуру ответа и вернуть "
            "{'capabilities': [...], 'fields': [...], 'sample_rate_hz': ..., "
            "'has_video_timecodes': bool}."
        )


def capabilities_from_fields(fields: set[str]) -> frozenset[Capability]:
    """Вывести возможности фида из набора пришедших полей.

    Вспомогательная функция для probe(): чтобы решение «что мы умеем на этом
    фиде» принималось по данным, а не на словах.
    """
    caps: set[Capability] = set()
    if {"event_type", "x", "y"} <= fields:
        caps.add(Capability.EVENTS)
    if {"player_id", "frame", "x", "y"} <= fields and "tracking" in fields:
        caps.add(Capability.TRACKING)
    if {"phase_type"} & fields:
        caps.add(Capability.PHASES)
    if {"off_ball_run", "run_type"} & fields:
        caps.add(Capability.OFF_BALL_RUNS)
    if {"pressing_chain"} & fields:
        caps.add(Capability.PRESSING_CHAINS)
    if {"line_break", "defensive_line_count"} & fields:
        caps.add(Capability.LINE_BREAKS)
    if {"passing_option"} & fields:
        caps.add(Capability.PASSING_OPTIONS)
    if {"xthreat", "xt"} & fields:
        caps.add(Capability.XTHREAT)
    if {"team_width", "team_length"} & fields:
        caps.add(Capability.TEAM_SHAPE)
    if {"defensive_line_height"} & fields:
        caps.add(Capability.DEFENSIVE_LINE)
    if {"speed"} & fields:
        caps.add(Capability.SPEEDS)
    return frozenset(caps)
