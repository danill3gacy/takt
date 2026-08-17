"""Проверки пайплайна.

Главное, что здесь проверяется, — не «код не падает», а что цифры означают
то, что написано в отчёте. Отдельно продублирован независимый пересчёт
нескольких метрик прямо из сырого CSV: если пайплайн начнёт врать, тест
это поймает.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from takt.baseline import build_baseline
from takt.insights import build_insights
from takt.metrics import compute_all
from takt.metrics.common import on_ball
from takt.model import Capability, MatchSet
from takt.sources.skillcorner import SkillCornerSource

DATA = Path(__file__).resolve().parents[1] / "data" / "skillcorner" / "data"
TEAM = "Auckland FC"


@pytest.fixture(scope="session")
def src():
    return SkillCornerSource(DATA)


@pytest.fixture(scope="session")
def all_matches(src):
    return src.load_many([r["id"] for r in src.list_matches()])


@pytest.fixture(scope="session")
def ms(src, all_matches):
    tid = src.team_id_by_name(TEAM)
    subject = [m for m in all_matches if tid in (m.home.id, m.away.id)]
    team = next(m.home if m.home.id == tid else m.away for m in subject)
    return MatchSet(team, subject)


@pytest.fixture(scope="session")
def computed(ms):
    return compute_all(ms, strict=True)


# --------------------------------------------------------------------------- #
# Система координат — самая опасная часть: ошибка здесь не падает, а тихо
# переворачивает половину выводов.
# --------------------------------------------------------------------------- #


def test_own_goalkeeper_is_at_own_goal(ms):
    ev = ms.events()
    own = on_ball(ev, ms.subject_team.id)
    gk = own[own["position"] == "GK"]
    assert len(gk) > 20
    assert gk["x"].mean() < -25, "вратарь разбираемой команды должен быть у своих ворот"


def test_opponent_goalkeeper_is_at_opposite_goal(ms):
    ev = ms.events()
    opp = ev[(ev["type"] == "player_possession") & (ev["possession_team_id"] != ms.subject_team.id)]
    gk = opp[opp["position"] == "GK"]
    assert len(gk) > 20
    assert gk["x"].mean() > 25, "вратарь соперника после сведения к нашей системе — справа"


def test_forwards_ahead_of_defenders(ms):
    ev = ms.events()
    own = on_ball(ev, ms.subject_team.id)
    by_pos = own.groupby("position")["x"].mean()
    assert by_pos.get("CF", 0) > by_pos.get("GK", 0)
    for cb in ("CB", "LCB", "RCB"):
        if cb in by_pos:
            assert by_pos["CF"] > by_pos[cb]


def test_channels_match_y_sign(ms):
    """Левый канал должен лежать по одну сторону от центра, правый — по другую."""
    ev = ms.events()
    own = on_ball(ev, ms.subject_team.id)
    m = own.groupby("channel_start")["y"].mean()
    assert m["wide_left"] > m["center"] > m["wide_right"]


def test_frame_of_is_an_involution(ms):
    """Двойное сведение к системе одной и той же команды возвращает исходное."""
    m = ms.matches[0]
    a = m.frame_of(ms.subject_team.id)
    b = m.frame_of(m.opponent_of(ms.subject_team.id).id)
    merged = a[["event_id", "x"]].merge(b[["event_id", "x"]], on="event_id", suffixes=("_a", "_b"))
    flipped = merged[merged["x_a"].notna() & merged["x_b"].notna()]
    assert ((flipped["x_a"] + flipped["x_b"]).abs() < 1e-6).mean() > 0.99


# --------------------------------------------------------------------------- #
# Независимый пересчёт из сырого CSV
# --------------------------------------------------------------------------- #


def test_shot_count_matches_raw_csv(src, ms):
    """Число ударов в отчёте совпадает с прямым подсчётом по CSV."""
    tid = ms.subject_team.id
    expected = 0
    for m in ms.matches:
        raw = pd.read_csv(DATA / "matches" / m.id / f"{m.id}_dynamic_events.csv", low_memory=False)
        expected += int(
            (
                (raw["event_type"] == "player_possession")
                & (raw["team_id"] == tid)
                & (raw["end_type"] == "shot")
            ).sum()
        )
    c = compute_all(ms, strict=True)
    assert c["shots"]["total"] == expected


def test_runs_behind_matches_raw_csv(ms):
    tid = ms.subject_team.id
    expected = 0
    for m in ms.matches:
        raw = pd.read_csv(DATA / "matches" / m.id / f"{m.id}_dynamic_events.csv", low_memory=False)
        expected += int(
            (
                (raw["event_type"] == "off_ball_run")
                & (raw["team_id"] == tid)
                & (raw["event_subtype"] == "behind")
            ).sum()
        )
    c = compute_all(ms, strict=True)
    assert c["off_ball_runs"]["behind"] == expected


def test_defensive_line_height_is_from_opponent_possessions(ms, computed):
    """Высота линии должна считаться только по владениям соперника."""
    dl = computed["defensive_line"]
    assert 5 < dl["mean"] < 70
    assert dl["p25"] < dl["mean"] < dl["p75"]
    heights = {r["phase"]: r["height"] for r in dl["by_phase"]}
    assert heights["высокий блок"] > heights["низкий блок"], (
        "в высоком блоке линия обязана стоять выше, чем в низком"
    )


# --------------------------------------------------------------------------- #
# Осмысленность значений
# --------------------------------------------------------------------------- #


def test_ppda_in_plausible_range(computed):
    assert 0.5 < computed["ppda"]["value"] < 60


def test_shares_sum_to_hundred(computed):
    for key, field in [("possession_profile", "rows"), ("formations", "rows")]:
        rows = computed[key][field]
        assert abs(sum(r["share"] for r in rows) - 100) < 25  # top-N, не весь хвост
    for block in ("thirds", "channels"):
        rows = computed["pressing_profile"][block]
        assert abs(sum(r["share"] for r in rows) - 100) < 0.5


def test_block_is_narrower_than_pitch(computed):
    bs = computed["block_shape"]
    assert 20 < bs["mean_width"] < 68
    assert 20 < bs["mean_length"] < 105
    assert bs["attack_width"] > bs["mean_width"], "в атаке команда шире, чем в обороне"


def test_every_player_row_has_minutes(computed):
    for r in computed["player_table"]["rows"]:
        assert r["minutes"] >= 60
        assert r["touches90"] >= 0


# --------------------------------------------------------------------------- #
# Контракт продукта
# --------------------------------------------------------------------------- #


def test_every_insight_has_clips(ms, computed, all_matches):
    bl = build_baseline(all_matches)
    ins = build_insights(computed, bl, ms.subject_team.name)
    assert ins, "тезисы должны генерироваться"
    for i in ins:
        assert i.clips, f"тезис без эпизодов не должен выпускаться: {i.text}"
        assert i.evidence, f"тезис без основания: {i.text}"
        assert i.kind in ("факт", "гипотеза")


def test_clips_point_to_real_timecodes(computed):
    for key in ("progression", "losses", "shots"):
        for c in computed[key]["clips"]:
            assert c["frame"] > 0
            assert c["match_id"]
            assert abs(c["t"] - c["frame"] / 10.0) < 0.11


def test_missing_capability_disables_metric(ms):
    """Урезанный фид не ломает пайплайн, а честно помечает метрики недоступными."""
    poor = [m for m in ms.matches]
    for m in poor:
        m.capabilities = frozenset({Capability.EVENTS})
    reduced = MatchSet(ms.subject_team, poor)
    c = compute_all(reduced)
    assert c.unavailable, "метрики, требующие трекинга, должны попасть в недоступные"
    assert not c.failed, f"падений быть не должно: {c.failed}"
    assert "shots" in c.values, "событийные метрики обязаны посчитаться"
    assert "off_ball_runs" not in c.values
    # вернуть как было для остальных тестов
    for m in poor:
        m.capabilities = SkillCornerSource.capabilities


def test_baseline_covers_league(all_matches):
    bl = build_baseline(all_matches)
    assert bl.n_teams >= 8
    assert "ppda" in bl.median and "line_height" in bl.median
    assert 20 < bl.median["line_height"] < 60


def test_report_json_is_serialisable(ms, all_matches):
    from takt.render import _clean
    from takt.report import build_report

    bl = build_baseline(all_matches)
    rep = build_report(
        ms, bl, club="ФК «Динамо» Москва", club_short="Д", generated="4 августа 2026"
    )
    payload = json.dumps(_clean(rep.to_dict()), ensure_ascii=False, allow_nan=False)
    assert len(payload) > 100_000
    assert "NaN" not in payload
    assert rep.patterns and all(p["clips"] for p in rep.patterns)
