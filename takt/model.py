"""Единая модель матча.

Ключевое проектное решение: матч = метаданные + события + кадры трекинга,
причём ЛЮБОЙ из двух потоков может отсутствовать. Метрики объявляют, какие
возможности источника им нужны; отчёт собирается из того, что реально есть.

Это сделано ради одного практического сценария: мы не знаем заранее, что
именно отдаёт фид конкретной лиги. Подключение нового поставщика = новый
адаптер + декларация возможностей, логика метрик не меняется.

Система координат после нормализации:
    x ∈ [-L/2, +L/2], y ∈ [-W/2, +W/2], метры, начало — центр поля.
    Ось x направлена в сторону атаки КОМАНДЫ, ВЛАДЕЮЩЕЙ МЯЧОМ.
    То есть x = +52 — ворота обороняющейся команды.
    Для метрик обороняющейся команды знак x инвертируется (см. Match.frame_of).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Capability(str, Enum):
    """Что умеет источник данных.

    Метрика объявляет требуемый набор; если источник его не покрывает,
    метрика не считается и попадает в раздел отчёта «недоступно на этом фиде»
    вместо того, чтобы молча вернуть неверное число.
    """

    EVENTS = "events"  # события с мячом
    TRACKING = "tracking"  # сырые кадры координат
    PHASES = "phases"  # разметка фаз игры
    OFF_BALL_RUNS = "off_ball_runs"  # забегания без мяча
    PRESSING_CHAINS = "pressing_chains"  # цепочки прессинга
    LINE_BREAKS = "line_breaks"  # взломы линий обороны
    PASSING_OPTIONS = "passing_options"  # варианты передачи в момент владения
    XTHREAT = "xthreat"  # модельная оценка угрозы
    TEAM_SHAPE = "team_shape"  # ширина/длина блока
    DEFENSIVE_LINE = "defensive_line"  # высота последней линии
    SPEEDS = "speeds"  # скорости игроков


_THIRD_FLIP = {
    "defensive_third": "attacking_third",
    "attacking_third": "defensive_third",
    "middle_third": "middle_third",
}

_CHANNEL_FLIP = {
    "wide_left": "wide_right",
    "wide_right": "wide_left",
    "half_space_left": "half_space_right",
    "half_space_right": "half_space_left",
    "center": "center",
}


@dataclass(frozen=True)
class Team:
    id: int
    name: str
    short_name: str
    color: str = "#888888"


@dataclass(frozen=True)
class Player:
    id: int
    name: str
    team_id: int
    number: int | None = None
    position: str | None = None
    minutes: float = 0.0


@dataclass
class Match:
    """Один матч в канонической форме."""

    id: str
    date: str
    competition: str
    home: Team
    away: Team
    score: tuple[int, int]
    players: dict[int, Player]
    pitch_length: float = 105.0
    pitch_width: float = 68.0

    # Каноническая таблица событий. Обязательные колонки:
    #   event_id, period, minute, second, t, team_id, player_id,
    #   type, subtype, x, y, x_end, y_end
    # Плюс произвольные колонки источника — метрики обращаются к ним
    # только если объявили соответствующую Capability.
    events: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Фазы игры: period, minute, t, duration, team_id, phase, def_phase,
    #   width_in, length_in, width_out, length_out
    phases: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Кадры трекинга: t, player_id, x, y, vx, vy (может быть пустым)
    tracking: pd.DataFrame = field(default_factory=pd.DataFrame)

    capabilities: frozenset[Capability] = frozenset()
    source: str = "unknown"

    # ------------------------------------------------------------------ #

    def team(self, team_id: int) -> Team:
        return self.home if team_id == self.home.id else self.away

    def opponent_of(self, team_id: int) -> Team:
        return self.away if team_id == self.home.id else self.home

    def has(self, *caps: Capability) -> bool:
        return all(c in self.capabilities for c in caps)

    def label(self) -> str:
        return f"{self.home.short_name} {self.score[0]}:{self.score[1]} {self.away.short_name}"

    def frame_of(self, team_id: int) -> pd.DataFrame:
        """События в системе координат конкретной команды.

        Возвращает копию events, где x > 0 всегда означает «ближе к воротам
        соперника этой команды», независимо от того, владела ли она мячом.
        """
        df = self.events.copy()
        flip = df["possession_team_id"].ne(team_id)
        for col in ("x", "x_end", "player_targeted_x_reception"):
            if col in df.columns:
                df.loc[flip, col] = -df.loc[flip, col]
        for col in ("y", "y_end", "player_targeted_y_reception"):
            if col in df.columns:
                df.loc[flip, col] = -df.loc[flip, col]
        # Категориальные метки зон источник считает во фрейме владеющей команды —
        # их тоже надо развернуть, иначе «чужая треть» соперника окажется
        # нашей защитной третью под тем же именем.
        for col in ("third_start", "third_end"):
            if col in df.columns:
                df.loc[flip, col] = df.loc[flip, col].map(_THIRD_FLIP).fillna(df.loc[flip, col])
        for col in ("channel_start", "channel_end"):
            if col in df.columns:
                df.loc[flip, col] = df.loc[flip, col].map(_CHANNEL_FLIP).fillna(df.loc[flip, col])
        return df

    def minutes_played(self) -> float:
        if self.events.empty:
            return 0.0
        return float(self.events["t"].max() / 60.0)


@dataclass
class MatchSet:
    """Набор матчей одной команды — то, по чему строится отчёт о сопернике."""

    subject_team: Team
    matches: list[Match]

    def __len__(self) -> int:
        return len(self.matches)

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Пересечение возможностей: считаем только то, что есть во ВСЕХ матчах."""
        if not self.matches:
            return frozenset()
        caps = self.matches[0].capabilities
        for m in self.matches[1:]:
            caps &= m.capabilities
        return caps

    def has(self, *caps: Capability) -> bool:
        return all(c in self.capabilities for c in caps)

    def events(self) -> pd.DataFrame:
        """Все события всех матчей в системе координат разбираемой команды."""
        parts = []
        for m in self.matches:
            df = m.frame_of(self.subject_team.id)
            df = df.assign(match_id=m.id, match_label=m.label())
            parts.append(df)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    def phases(self) -> pd.DataFrame:
        parts = []
        for m in self.matches:
            if m.phases.empty:
                continue
            parts.append(m.phases.assign(match_id=m.id, match_label=m.label()))
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    def total_minutes(self) -> float:
        return sum(m.minutes_played() for m in self.matches)
