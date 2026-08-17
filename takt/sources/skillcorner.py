"""Адаптер SkillCorner Game Intelligence (открытый датасет).

Это производный слой поверх оптического трекинга: события с мячом,
забегания без мяча, цепочки прессинга, взломы линий, варианты передачи,
ширина и длина блока по фазам. По типу данных это то же, что отдаёт
оптическая система лиги (4 камеры → нейросетевой трекинг → производные
события), поэтому логика метрик, написанная здесь, переносится на фид лиги
заменой одного файла.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..model import Capability, Match, Player, Team
from .base import Source

FPS = 10.0

# Колонки, которые мы протаскиваем в каноническую таблицу сверх обязательных.
# Каждая привязана к Capability — если фид их не даёт, соответствующие
# метрики просто не считаются.
RICH_COLUMNS = {
    Capability.XTHREAT: [
        "xthreat",
        "xpass_completion",
        "xshot_player_possession_start",
        "xshot_player_possession_end",
        "xshot_player_possession_max",
        "xloss_player_possession_start",
        "xloss_player_possession_max",
        "player_targeted_xthreat",
        "player_targeted_xpass_completion",
    ],
    Capability.LINE_BREAKS: [
        "first_line_break",
        "second_last_line_break",
        "last_line_break",
        "furthest_line_break",
        "furthest_line_break_type",
        "n_defensive_lines",
        "organised_defense",
        "defensive_structure",
        "n_opponents_bypassed",
    ],
    Capability.PRESSING_CHAINS: [
        "pressing_chain",
        "pressing_chain_length",
        "pressing_chain_end_type",
        "pressing_chain_index",
        "index_in_pressing_chain",
        "interplayer_distance",
        "interplayer_distance_min",
        "angle_of_engagement",
        "goal_side_start",
        "goal_side_end",
        "beaten_by_possession",
        "beaten_by_movement",
        "stop_possession_danger",
        "reduce_possession_danger",
        "force_backward",
        "possession_danger",
        "simultaneous_defensive_engagement_same_target",
    ],
    Capability.OFF_BALL_RUNS: [
        "distance_covered",
        "trajectory_direction",
        "in_to_out",
        "out_to_in",
        "separation_start",
        "separation_end",
        "separation_gain",
        "delta_to_last_defensive_line_start",
        "delta_to_last_defensive_line_end",
        "inside_defensive_shape_start",
        "inside_defensive_shape_end",
        "targeted",
        "received",
        "received_in_space",
        "dangerous",
        "intended_run_behind",
        "push_defensive_line",
        "break_defensive_line",
        "give_and_go",
        "n_simultaneous_runs",
    ],
    Capability.PASSING_OPTIONS: [
        "n_passing_options",
        "n_off_ball_runs",
        "n_passing_options_line_break",
        "n_passing_options_ahead",
        "n_passing_options_dangerous_difficult",
        "n_passing_options_dangerous_not_difficult",
        "difficult_pass_target",
        "passing_option_score",
        "player_targeted_id",
        "player_targeted_name",
        "player_targeted_x_reception",
        "player_targeted_y_reception",
    ],
    Capability.DEFENSIVE_LINE: [
        "last_defensive_line_x_start",
        "last_defensive_line_x_end",
        "last_defensive_line_height_start",
        "last_defensive_line_height_end",
    ],
    Capability.SPEEDS: ["speed_avg", "speed_avg_band"],
    Capability.EVENTS: [
        "start_type",
        "end_type",
        "lead_to_shot",
        "lead_to_goal",
        "game_state",
        "team_score",
        "opponent_team_score",
        "third_start",
        "third_end",
        "channel_start",
        "channel_end",
        "penalty_area_start",
        "penalty_area_end",
        "pass_distance",
        "pass_angle",
        "pass_direction",
        "pass_ahead",
        "pass_outcome",
        "high_pass",
        "one_touch",
        "quick_pass",
        "carry",
        "is_header",
        "forward_momentum",
        "duration",
        "game_interruption_before",
        "game_interruption_after",
        "n_player_possessions_in_phase",
        "team_possession_loss_in_phase",
        "first_player_possession_in_team_possession",
        "last_player_possession_in_team_possession",
    ],
    Capability.PHASES: [
        "phase_index",
        "team_in_possession_phase_type",
        "team_out_of_possession_phase_type",
    ],
}

# Типы событий, где team_id — это команда БЕЗ мяча.
_DEFENSIVE_TYPES = {"on_ball_engagement"}

_PHASE_RU = {
    "build_up": "начало атаки",
    "create": "созидание",
    "finish": "завершение",
    "direct": "длинная передача",
    "transition": "переход",
    "quick_break": "быстрый выпад",
    "counter": "контратака",
    "set_play": "стандарт",
    "chaotic": "хаос",
    "disruption": "под давлением",
}

_DEF_PHASE_RU = {
    "high_block": "высокий блок",
    "medium_block": "средний блок",
    "low_block": "низкий блок",
    "defending_direct": "против длинных",
    "defending_transition": "переход в оборону",
    "defending_quick_break": "против быстрого выпада",
    "defending_counter": "против контратаки",
    "defending_set_play": "против стандарта",
    "chaotic": "хаос",
    "disruption": "срыв атаки соперника",
}


class SkillCornerSource(Source):
    name = "SkillCorner Game Intelligence"
    capabilities = frozenset(
        {
            Capability.EVENTS,
            Capability.PHASES,
            Capability.OFF_BALL_RUNS,
            Capability.PRESSING_CHAINS,
            Capability.LINE_BREAKS,
            Capability.PASSING_OPTIONS,
            Capability.XTHREAT,
            Capability.TEAM_SHAPE,
            Capability.DEFENSIVE_LINE,
            Capability.SPEEDS,
        }
    )

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._index = json.loads((self.root / "matches.json").read_text())

    # ------------------------------------------------------------------ #

    def list_matches(self, team: str | int | None = None) -> list[dict]:
        out = []
        for m in self._index:
            row = {
                "id": m["id"],
                "date": m["date_time"][:10],
                "home": m["home_team"]["short_name"],
                "away": m["away_team"]["short_name"],
                "home_id": m["home_team"]["id"],
                "away_id": m["away_team"]["id"],
            }
            if team is None:
                out.append(row)
            elif isinstance(team, int):
                if team in (row["home_id"], row["away_id"]):
                    out.append(row)
            elif str(team).lower() in (row["home"].lower(), row["away"].lower()):
                out.append(row)
        return sorted(out, key=lambda r: r["date"])

    def team_id_by_name(self, name: str) -> int:
        for m in self._index:
            for side in ("home_team", "away_team"):
                if m[side]["short_name"].lower() == name.lower():
                    return m[side]["id"]
        raise KeyError(name)

    # ------------------------------------------------------------------ #

    def load(self, match_id: str | int) -> Match:
        d = self.root / "matches" / str(match_id)
        meta = json.loads((d / f"{match_id}_match.json").read_text())

        home = Team(
            id=meta["home_team"]["id"],
            name=meta["home_team"]["name"],
            short_name=meta["home_team"]["short_name"],
            color=(meta.get("home_team_kit") or {}).get("jersey_color", "#2255cc"),
        )
        away = Team(
            id=meta["away_team"]["id"],
            name=meta["away_team"]["name"],
            short_name=meta["away_team"]["short_name"],
            color=(meta.get("away_team_kit") or {}).get("jersey_color", "#cc3322"),
        )

        players: dict[int, Player] = {}
        for p in meta["players"]:
            pt = (p.get("playing_time") or {}).get("total") or {}
            players[p["id"]] = Player(
                id=p["id"],
                name=p.get("short_name")
                or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                team_id=p["team_id"],
                number=p.get("number"),
                position=(p.get("player_role") or {}).get("acronym"),
                minutes=float(pt.get("minutes_played") or 0.0),
            )

        events = self._load_events(d, match_id, home.id, away.id)
        phases = self._load_phases(d, match_id)

        return Match(
            id=str(match_id),
            date=meta["date_time"][:10],
            competition=(meta.get("competition_edition") or {}).get("name", ""),
            home=home,
            away=away,
            score=(meta.get("home_team_score", 0), meta.get("away_team_score", 0)),
            players=players,
            pitch_length=float(meta.get("pitch_length") or 105.0),
            pitch_width=float(meta.get("pitch_width") or 68.0),
            events=events,
            phases=phases,
            capabilities=self.capabilities,
            source=self.name,
        )

    # ------------------------------------------------------------------ #

    def _load_events(
        self, d: Path, match_id: int | str, home_id: int, away_id: int
    ) -> pd.DataFrame:
        raw = pd.read_csv(d / f"{match_id}_dynamic_events.csv", low_memory=False)

        df = pd.DataFrame(
            {
                "event_id": raw["event_id"],
                "period": raw["period"].astype("Int64"),
                "minute": raw["minute_start"].astype("Int64"),
                "second": raw["second_start"].astype("Int64"),
                "frame": raw["frame_start"].astype("Int64"),
                "frame_end": raw["frame_end"].astype("Int64"),
                "team_id": raw["team_id"].astype("Int64"),
                "player_id": raw["player_id"].astype("Int64"),
                "player_name": raw["player_name"],
                "position": raw["player_position"],
                "type": raw["event_type"],
                "subtype": raw["event_subtype"],
                "x": raw["x_start"].astype(float),
                "y": raw["y_start"].astype(float),
                "x_end": raw["x_end"].astype(float),
                "y_end": raw["y_end"].astype(float),
                "in_possession_player_id": raw["player_in_possession_id"].astype("Int64"),
                "in_possession_player": raw["player_in_possession_name"],
            }
        )

        df["t"] = raw["frame_start"].astype(float) / FPS

        # Команда, владеющая мячом.
        other = np.where(df["team_id"] == home_id, away_id, home_id)
        df["possession_team_id"] = np.where(
            raw["event_type"].isin(_DEFENSIVE_TYPES), other, df["team_id"]
        ).astype("int64")

        # ВАЖНО: источник уже отдаёт координаты в системе атакующей команды —
        # проверено на сырых данных (вратарь всегда около x = -40 независимо от
        # attacking_side, нападающий — в плюсе). Колонка attacking_side описывает
        # физическое направление атаки на видео и для координат значения не имеет.
        # Разворот здесь был бы ошибкой; он нужен только при сведении матчей
        # к системе одной команды — см. Match.frame_of.

        # Богатые колонки источника — одним concat, иначе pandas фрагментирует блок.
        rich = [c for cols in RICH_COLUMNS.values() for c in cols if c in raw.columns]
        df = pd.concat([df, raw[rich]], axis=1)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Единая колонка «угроза действия». Источник кладёт её в разные поля:
        #   player_possession      -> player_targeted_xthreat (угроза сделанной передачи)
        #   passing_option / run   -> xthreat (угроза доступного, но не обязательно
        #                             использованного варианта)
        # Метрики работают с xt как с реализованной угрозой, с xthreat — с потенциальной.
        extra = {}
        if "player_targeted_xthreat" in df.columns and "xthreat" in df.columns:
            extra["xt"] = np.where(
                df["type"].eq("player_possession"), df["player_targeted_xthreat"], np.nan
            )
        if "possession_danger" in df.columns:
            extra["danger_faced"] = np.where(
                df["type"].eq("on_ball_engagement"), df["possession_danger"], np.nan
            )
        if "team_in_possession_phase_type" in df.columns:
            extra["phase"] = (
                df["team_in_possession_phase_type"]
                .map(_PHASE_RU)
                .fillna(df["team_in_possession_phase_type"])
            )
        if "team_out_of_possession_phase_type" in df.columns:
            extra["def_phase"] = (
                df["team_out_of_possession_phase_type"]
                .map(_DEF_PHASE_RU)
                .fillna(df["team_out_of_possession_phase_type"])
            )
        extra["clock"] = (df["minute"].fillna(0).astype(int) + 1).astype(str) + "′"
        df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)
        return df.sort_values("t").reset_index(drop=True)

    def _load_phases(self, d: Path, match_id: int | str) -> pd.DataFrame:
        raw = pd.read_csv(d / f"{match_id}_phases_of_play.csv", low_memory=False)
        df = pd.DataFrame(
            {
                "phase_index": raw["index"],
                "period": raw["period"],
                "minute": raw["minute_start"],
                "t": raw["frame_start"].astype(float) / FPS,
                "frame": raw["frame_start"],
                "frame_end": raw["frame_end"],
                "duration": raw["duration"],
                "team_id": raw["team_in_possession_id"].astype("Int64"),
                "phase_raw": raw["team_in_possession_phase_type"],
                "def_phase_raw": raw["team_out_of_possession_phase_type"],
                "n_possessions": raw["n_player_possessions_in_phase"],
                "lost": raw["team_possession_loss_in_phase"],
                "lead_to_shot": raw["team_possession_lead_to_shot"],
                "lead_to_goal": raw["team_possession_lead_to_goal"],
                "width_in": raw["team_in_possession_width_end"],
                "length_in": raw["team_in_possession_length_end"],
                "width_out": raw["team_out_of_possession_width_end"],
                "length_out": raw["team_out_of_possession_length_end"],
                "x_start": raw["x_start"].astype(float),
                "x_end": raw["x_end"].astype(float),
                "y_start": raw["y_start"].astype(float),
                "third_start": raw["third_start"],
                "third_end": raw["third_end"],
            }
        )
        df["phase"] = df["phase_raw"].map(_PHASE_RU).fillna(df["phase_raw"])
        df["def_phase"] = df["def_phase_raw"].map(_DEF_PHASE_RU).fillna(df["def_phase_raw"])
        return df
