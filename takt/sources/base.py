"""Базовый контракт источника данных.

Чтобы подключить новый фид (Яндекс×VSporte, РУСТАТ, Sportec, кто угодно),
нужно написать один класс: объявить capabilities и отдать канонические
таблицы. Ни одна метрика при этом не переписывается.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable

from ..model import Capability, Match

# Обязательные колонки канонической таблицы событий.
CANONICAL_EVENT_COLUMNS = [
    "event_id",          # уникальный id события
    "period",            # 1 | 2 | ...
    "minute",            # минута матча (0-based)
    "second",
    "t",                 # секунды от начала матча
    "frame",             # кадр видео — точка входа в клип
    "frame_end",
    "team_id",           # команда игрока-субъекта события
    "player_id",
    "player_name",
    "position",
    "possession_team_id",  # команда, владеющая мячом в этот момент
    "type",              # player_possession | on_ball_engagement | off_ball_run | passing_option
    "subtype",
    "x", "y",            # начало, метры, атака слева направо для владеющей команды
    "x_end", "y_end",
]


class Source(abc.ABC):
    """Адаптер конкретного поставщика данных."""

    name: str = "abstract"
    capabilities: frozenset[Capability] = frozenset()

    @abc.abstractmethod
    def list_matches(self, team: str | int | None = None) -> list[dict]:
        """Перечислить доступные матчи (id, дата, команды, счёт)."""

    @abc.abstractmethod
    def load(self, match_id: str | int) -> Match:
        """Загрузить один матч в каноническую модель."""

    def load_many(self, match_ids: Iterable[str | int]) -> list[Match]:
        return [self.load(mid) for mid in match_ids]

    def describe(self) -> str:
        have = sorted(c.value for c in self.capabilities)
        return f"{self.name}: {', '.join(have)}"
